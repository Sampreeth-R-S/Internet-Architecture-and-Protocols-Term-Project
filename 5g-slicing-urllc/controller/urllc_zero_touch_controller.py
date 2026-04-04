"""
Zero-Touch URLLC Slice Resource Controller.

Group B - URLLC slice (SST=2, SD="000002")

How it works:
  - Loads the trained multivariate LSTM model (4 features per step)
  - At every CHECK_INTERVAL the controller builds a feature window from
    recent latency measurements and calls the LSTM for the next HORIZON steps
  - Decision policy (proactive, before SLA breach):
      peak_pred  > SLA_THRESHOLD (10 ms) -> URGENT:    ARP 2->1, GBR +50 %
      peak_pred  > PROACTIVE_THRESHOLD (8 ms) -> PROACTIVE: ARP 2->1, GBR +25 %
      avg_pred   < RESTORE_THRESHOLD   (3 ms) -> RESTORE:   ARP back to 2, GBR baseline
      otherwise  -> HOLD
  - All decisions are logged to data/urllc_controller_decisions.json

Usage:
    python3 urllc_zero_touch_controller.py            # Simulation mode
    python3 urllc_zero_touch_controller.py --live     # Live mode (reads ogstun)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import requests
import time
import json
import csv
import os
import sys
import argparse
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OPEN5GS_API        = "http://127.0.0.1:9999/api"
CHECK_INTERVAL     = 2          # seconds (URLLC needs faster reaction than eMBB)
NUM_SUBSCRIBERS    = 5

# Latency SLA thresholds (ms)
SLA_THRESHOLD      = 10.0       # Hard SLA - breach triggers URGENT action
PROACTIVE_THRESHOLD = 8.0       # Soft threshold - pre-emptive action before breach
RESTORE_THRESHOLD  = 3.0        # Latency is low enough to release extra resources

# GBR (Guaranteed Bit Rate) per slice in Mbps
INITIAL_GBR_DL     = 50
INITIAL_GBR_UL     = 25
MAX_GBR_DL         = 150
MAX_GBR_UL         = 75
MIN_GBR_DL         = 20
MIN_GBR_UL         = 10

# ARP (Allocation and Retention Priority): lower number = higher priority in 5G NR
NORMAL_ARP_PRIORITY  = 2
BOOSTED_ARP_PRIORITY = 1        # Highest schedulable priority

SUBSCRIBER_IMSIS = [
    "999700000000006",
    "999700000000007",
    "999700000000008",
    "999700000000009",
    "999700000000010",
]

# LSTM hyper-parameters - must match lstm_model.py exactly
WINDOW_SIZE  = 10
HORIZON      = 3
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
N_FEATURES   = 4    # [max_lat, mean_lat, std_lat, loss_rate]
SMOOTH_WINDOW = 3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..')


# ---------------------------------------------------------------------------
# LSTM model definition (mirrors lstm_model.py)
# ---------------------------------------------------------------------------
class TrafficLSTM(nn.Module):
    def __init__(self, input_size=N_FEATURES, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, output_size=HORIZON):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
class URLLCSliceController:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(ROOT_DIR, 'saved', 'lstm_urllc.pth')

        # --- Load LSTM model ---
        self.model = TrafficLSTM(N_FEATURES, HIDDEN_SIZE, NUM_LAYERS, HORIZON)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()
        print(f"  Model loaded from: {model_path}")

        # --- Rebuild scalers from training data (same pipeline as lstm_model.py) ---
        self.feat_scaler, self.tgt_scaler = self._build_scalers()
        print("  Scalers fitted from training data.")

        # --- State ---
        self.current_arp_priority = NORMAL_ARP_PRIORITY
        self.current_gbr_dl = INITIAL_GBR_DL
        self.current_gbr_ul = INITIAL_GBR_UL
        self.history_features = []   # list of 4-element feature vectors
        self.decisions = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_scalers(self):
        """Refit MinMaxScaler from training data - identical to lstm_model.py."""
        data_path = os.path.join(ROOT_DIR, 'data', 'Training_data.csv')
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()
        df['latency_ms'] = pd.to_numeric(df['latency_ms'], errors='coerce')
        df.dropna(subset=['latency_ms'], inplace=True)

        agg = (df.groupby('seq')
                 .agg(max_lat=('latency_ms', 'max'),
                      mean_lat=('latency_ms', 'mean'),
                      std_lat=('latency_ms', 'std'),
                      total_recv=('recv', 'sum'),
                      total_lost=('lost', 'sum'))
                 .fillna(0)
                 .sort_index()
                 .reset_index())
        agg['loss_rate'] = agg['total_lost'] / (agg['total_recv'] + 1e-6)

        feat_cols = ['max_lat', 'mean_lat', 'std_lat', 'loss_rate']
        features = agg[feat_cols].values.astype(np.float32)

        raw_target = agg['max_lat'].values.astype(np.float32)
        smooth_target = (pd.Series(raw_target)
                           .rolling(SMOOTH_WINDOW, min_periods=1)
                           .mean()
                           .values
                           .astype(np.float32))

        # Replicate the exact train split to avoid data leakage in scaler
        def _make_sequences(feats, tgt, w, h):
            Xs, ys = [], []
            for i in range(len(tgt) - w - h + 1):
                Xs.append(feats[i:i + w])
                ys.append(tgt[i + w:i + w + h])
            return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

        X, y = _make_sequences(features, smooth_target, WINDOW_SIZE, HORIZON)
        n = len(X)
        train_end = int((1.0 - 0.2 - 0.1) * n)   # TEST_RATIO=0.2, VAL_RATIO=0.1

        X_train = X[:train_end]
        y_train = y[:train_end]

        feat_scaler = MinMaxScaler()
        feat_scaler.fit(X_train.reshape(-1, N_FEATURES))

        tgt_scaler = MinMaxScaler()
        tgt_scaler.fit(y_train.reshape(-1, 1))

        return feat_scaler, tgt_scaler

    def _scale_features(self, window_arr):
        """window_arr: (WINDOW_SIZE, N_FEATURES) -> scaled (WINDOW_SIZE, N_FEATURES)"""
        return self.feat_scaler.transform(window_arr)

    def _inverse_scale_target(self, pred_scaled):
        """pred_scaled: (HORIZON,) -> latency values in ms"""
        return self.tgt_scaler.inverse_transform(
            pred_scaled.reshape(-1, 1)).flatten()

    # ------------------------------------------------------------------
    # Core controller methods
    # ------------------------------------------------------------------
    def predict(self, feature_window):
        """
        feature_window: numpy array (WINDOW_SIZE, N_FEATURES)
        Returns predicted latency in ms for next HORIZON steps.
        """
        scaled = self._scale_features(feature_window)
        x = torch.FloatTensor(scaled).unsqueeze(0)   # (1, window, features)
        with torch.no_grad():
            pred_scaled = self.model(x).numpy()[0]   # (HORIZON,)
        return self._inverse_scale_target(pred_scaled)

    def decide_action(self, predictions):
        """
        Proactive decision based on predicted latency.
        Returns (action, trigger_value) where trigger_value is the peak/avg
        that caused the decision.
        """
        peak_pred = float(np.max(predictions))
        avg_pred  = float(np.mean(predictions))

        if peak_pred > SLA_THRESHOLD:
            return 'prioritize_urgent', peak_pred

        if peak_pred > PROACTIVE_THRESHOLD:
            return 'prioritize', peak_pred

        if avg_pred < RESTORE_THRESHOLD and (
                self.current_arp_priority < NORMAL_ARP_PRIORITY or
                self.current_gbr_dl > INITIAL_GBR_DL):
            return 'restore', avg_pred

        return 'hold', avg_pred

    def apply_action(self, action, trigger_value, predictions):
        """Apply resource reconfiguration and log the decision."""
        timestamp = datetime.now().isoformat()
        prev_arp = self.current_arp_priority
        prev_gbr_dl = self.current_gbr_dl
        prev_gbr_ul = self.current_gbr_ul

        if action == 'prioritize_urgent':
            # SLA breach imminent - max priority + large GBR boost
            self.current_arp_priority = BOOSTED_ARP_PRIORITY
            self.current_gbr_dl = min(MAX_GBR_DL,
                                      int(prev_gbr_dl * 1.50))   # +50 %
            self.current_gbr_ul = min(MAX_GBR_UL,
                                      int(prev_gbr_ul * 1.50))
            print(
                f"[{timestamp}] !! URGENT  - RTT spike predicted: {trigger_value:.2f} ms "
                f"(>{SLA_THRESHOLD} ms SLA) | "
                f"ARP {prev_arp}->{self.current_arp_priority} | "
                f"GBR DL {prev_gbr_dl}->{self.current_gbr_dl} Mbps"
            )

        elif action == 'prioritize':
            # Proactive boost before SLA breach
            self.current_arp_priority = BOOSTED_ARP_PRIORITY
            self.current_gbr_dl = min(MAX_GBR_DL,
                                      int(prev_gbr_dl * 1.25))   # +25 %
            self.current_gbr_ul = min(MAX_GBR_UL,
                                      int(prev_gbr_ul * 1.25))
            print(
                f"[{timestamp}] ^^ PROACTIVE - RTT forecast: {trigger_value:.2f} ms "
                f"(>{PROACTIVE_THRESHOLD} ms) | "
                f"ARP {prev_arp}->{self.current_arp_priority} | "
                f"GBR DL {prev_gbr_dl}->{self.current_gbr_dl} Mbps"
            )

        elif action == 'restore':
            # Latency well below SLA - release extra resources
            self.current_arp_priority = NORMAL_ARP_PRIORITY
            self.current_gbr_dl = max(MIN_GBR_DL,
                                      int(prev_gbr_dl * 0.80))   # -20 %
            self.current_gbr_ul = max(MIN_GBR_UL,
                                      int(prev_gbr_ul * 0.80))
            # Floor at initial baseline so we don't shrink below provisioned
            self.current_gbr_dl = max(self.current_gbr_dl, INITIAL_GBR_DL)
            self.current_gbr_ul = max(self.current_gbr_ul, INITIAL_GBR_UL)
            print(
                f"[{timestamp}] vv RESTORE  - avg RTT: {trigger_value:.2f} ms "
                f"(safe, <{RESTORE_THRESHOLD} ms) | "
                f"ARP {prev_arp}->{self.current_arp_priority} | "
                f"GBR DL {prev_gbr_dl}->{self.current_gbr_dl} Mbps"
            )

        else:  # hold
            print(
                f"[{timestamp}] -- HOLD     - avg RTT: {trigger_value:.2f} ms | "
                f"ARP={self.current_arp_priority}, GBR DL={self.current_gbr_dl} Mbps"
            )

        # Log entry
        self.decisions.append({
            'timestamp':        timestamp,
            'action':           action,
            'trigger_value_ms': round(trigger_value, 3),
            'predicted_ms':     [round(float(p), 3) for p in predictions],
            'arp_priority':     self.current_arp_priority,
            'gbr_dl_mbps':      self.current_gbr_dl,
            'gbr_ul_mbps':      self.current_gbr_ul,
            'num_subscribers':  NUM_SUBSCRIBERS,
        })

        # --- Real deployment: call Open5GS REST API ---
        # In a live 5G core the controller would PUT slice-level QoS here.
        # Uncomment and adapt to your Open5GS WebUI endpoint:
        #
        # for imsi in SUBSCRIBER_IMSIS:
        #     try:
        #         resp = requests.put(
        #             f"{OPEN5GS_API}/subscribers/{imsi}/session",
        #             json={
        #                 "slice": {"sst": 2, "sd": "000002"},
        #                 "qos": {
        #                     "arp": {
        #                         "priority_level": self.current_arp_priority,
        #                         "pre_emption_capability": 1,
        #                         "pre_emption_vulnerability": 0
        #                     },
        #                     "gbr": {
        #                         "downlink": {"value": self.current_gbr_dl,
        #                                      "unit": "Mbps"},
        #                         "uplink":   {"value": self.current_gbr_ul,
        #                                      "unit": "Mbps"}
        #                     }
        #                 }
        #             },
        #             headers={"Content-Type": "application/json"},
        #             timeout=2
        #         )
        #         resp.raise_for_status()
        #     except Exception as e:
        #         print(f"  API Error for {imsi}: {e}")

    def get_current_features(self):
        """
        Live mode: read the latest per-UE latency measurements and compute
        [max_lat, mean_lat, std_lat, loss_rate] from the most recent rows.
        Reads from data/Training_data.csv as a proxy; replace with a live
        ogstun reader in real deployment.
        """
        try:
            csv_path = os.path.join(ROOT_DIR, 'data', 'Training_data.csv')
            df = pd.read_csv(csv_path, nrows=50)   # tail-like: last 50 rows
            df.columns = df.columns.str.strip()
            df['latency_ms'] = pd.to_numeric(df['latency_ms'], errors='coerce')
            df.dropna(subset=['latency_ms'], inplace=True)
            lats = df['latency_ms'].values
            recv = df['recv'].sum() if 'recv' in df.columns else 1
            lost = df['lost'].sum() if 'lost' in df.columns else 0
            return np.array([lats.max(), lats.mean(),
                             lats.std() if len(lats) > 1 else 0.0,
                             lost / (recv + 1e-6)], dtype=np.float32)
        except Exception:
            return np.zeros(N_FEATURES, dtype=np.float32)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, simulation_data=None):
        """
        simulation_data: numpy array of shape (T, N_FEATURES) for offline mode.
        Pass None to run in live mode.
        """
        print("=" * 65)
        print("  Zero-Touch URLLC Slice Controller  --  Group B")
        print("=" * 65)
        print(f"  Slice:               SST=2, SD=000002 (URLLC)")
        print(f"  Subscribers:         {NUM_SUBSCRIBERS} ({', '.join(SUBSCRIBER_IMSIS)})")
        print(f"  LSTM window/horizon: {WINDOW_SIZE} / {HORIZON} steps")
        print(f"  SLA threshold:       {SLA_THRESHOLD} ms (URGENT above this)")
        print(f"  Proactive threshold: {PROACTIVE_THRESHOLD} ms")
        print(f"  Restore threshold:   {RESTORE_THRESHOLD} ms")
        print(f"  ARP priority:        normal={NORMAL_ARP_PRIORITY}, "
              f"boosted={BOOSTED_ARP_PRIORITY}")
        print(f"  Initial GBR:         DL={INITIAL_GBR_DL} Mbps, "
              f"UL={INITIAL_GBR_UL} Mbps")
        print()

        if simulation_data is not None:
            # ---- Simulation / offline mode --------------------------------
            T = len(simulation_data)
            print(f"[SIMULATION MODE]  {T} time steps\n")

            for i in range(WINDOW_SIZE, T):
                window = simulation_data[i - WINDOW_SIZE:i]   # (W, F)
                predictions = self.predict(window)
                action, trigger = self.decide_action(predictions)

                # Always log; only print non-hold actions
                if action != 'hold':
                    self.apply_action(action, trigger, predictions)
                else:
                    self.decisions.append({
                        'timestamp':        datetime.now().isoformat(),
                        'action':           'hold',
                        'trigger_value_ms': round(trigger, 3),
                        'predicted_ms':     [round(float(p), 3) for p in predictions],
                        'arp_priority':     self.current_arp_priority,
                        'gbr_dl_mbps':      self.current_gbr_dl,
                        'gbr_ul_mbps':      self.current_gbr_ul,
                        'num_subscribers':  NUM_SUBSCRIBERS,
                    })

            # Summary
            actions = [d['action'] for d in self.decisions]
            urgent    = actions.count('prioritize_urgent')
            proactive = actions.count('prioritize')
            restores  = actions.count('restore')
            holds     = actions.count('hold')
            sla_risk  = sum(1 for d in self.decisions
                            if d['trigger_value_ms'] > SLA_THRESHOLD)

            print(f"\n{'=' * 65}")
            print(f"  Simulation summary  ({T - WINDOW_SIZE} control cycles)")
            print(f"    URGENT actions  : {urgent}")
            print(f"    PROACTIVE boosts: {proactive}")
            print(f"    RESTORE actions : {restores}")
            print(f"    HOLD            : {holds}")
            print(f"    SLA-risk events : {sla_risk}  "
                  f"(predicted RTT > {SLA_THRESHOLD} ms)")
            return self.decisions

        # ---- Live mode ----------------------------------------------------
        print("[LIVE MODE]  Monitoring URLLC traffic...\n")
        while True:
            feat_vec = self.get_current_features()
            self.history_features.append(feat_vec)

            if len(self.history_features) >= WINDOW_SIZE:
                window = np.array(self.history_features[-WINDOW_SIZE:])
                predictions = self.predict(window)
                action, trigger = self.decide_action(predictions)
                self.apply_action(action, trigger, predictions)

            time.sleep(CHECK_INTERVAL)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Zero-Touch URLLC Slice Controller - Group B')
    parser.add_argument('--live', action='store_true',
                        help='Run in live mode (reads from ogstun)')
    parser.add_argument('--model', default=None,
                        help='Path to lstm_urllc.pth (optional)')
    args = parser.parse_args()

    controller = URLLCSliceController(model_path=args.model)

    if args.live:
        controller.run()
    else:
        # Simulation mode: build feature matrix from Training_data.csv
        data_path = os.path.join(ROOT_DIR, 'data', 'Training_data.csv')
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()
        df['latency_ms'] = pd.to_numeric(df['latency_ms'], errors='coerce')
        df.dropna(subset=['latency_ms'], inplace=True)

        agg = (df.groupby('seq')
                 .agg(max_lat=('latency_ms', 'max'),
                      mean_lat=('latency_ms', 'mean'),
                      std_lat=('latency_ms', 'std'),
                      total_recv=('recv', 'sum'),
                      total_lost=('lost', 'sum'))
                 .fillna(0)
                 .sort_index()
                 .reset_index())
        agg['loss_rate'] = agg['total_lost'] / (agg['total_recv'] + 1e-6)

        feat_cols = ['max_lat', 'mean_lat', 'std_lat', 'loss_rate']
        sim_data = agg[feat_cols].values.astype(np.float32)

        decisions = controller.run(simulation_data=sim_data)

        # Save decisions log
        out_path = os.path.join(ROOT_DIR, 'data', 'urllc_controller_decisions.json')
        with open(out_path, 'w') as f:
            json.dump(decisions, f, indent=2)
        print(f"\nDecisions saved -> {out_path}")
