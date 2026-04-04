"""
Unified Zero-Touch 5G Slice Orchestrator
Manages all 3 slices simultaneously with cross-slice awareness.

Slices:
  - eMBB  (SST=1, SD=000001) : High-bandwidth 4K streaming
  - mMTC  (SST=1, SD=000002) : Massive IoT sensor connectivity
  - URLLC (SST=2, SD=000002) : Ultra-low latency mission-critical

Cross-slice rule:
  If URLLC is URGENT or PROACTIVE → eMBB expansion is BLOCKED.
  URLLC always gets priority because a 1ms latency violation is
  catastrophic (autonomous vehicles, remote surgery), whereas a
  temporary bandwidth reduction for video is just a quality drop.

Usage:
    python3 unified_controller.py           # Simulation mode (uses CSVs)
    python3 unified_controller.py --live    # Live mode (reads from network)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
import os
import argparse
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════
# LSTM MODEL DEFINITIONS
# (Mirrors each slice's model exactly so we can load their .pth files)
# ═══════════════════════════════════════════════════════════════

class UnivariateLSTM(nn.Module):
    """Single-feature LSTM — used by eMBB (throughput) and mMTC (packet_rate)."""
    def __init__(self, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


class MultivariateLSTM(nn.Module):
    """4-feature LSTM — used by URLLC (max_lat, mean_lat, std_lat, loss_rate)."""
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, output_size=3):
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


# ═══════════════════════════════════════════════════════════════
# SHARED STATE  (cross-slice coordination)
# ═══════════════════════════════════════════════════════════════

class SharedState:
    """
    All 3 controllers read/write this object so they are aware of
    each other's current status. No threading needed in simulation
    mode — controllers run sequentially and share this state.
    """
    def __init__(self):
        # URLLC publishes its current action here after every decision
        self.urllc_status = 'hold'   # hold | prioritize | prioritize_urgent | restore

        # Resource snapshot (for the final summary report)
        self.embb_bw_dl = 500
        self.mmtc_bw_dl = 10
        self.urllc_gbr_dl = 50

    def urllc_is_critical(self):
        """Returns True when URLLC needs priority (eMBB must not expand)."""
        return self.urllc_status in ('prioritize_urgent', 'prioritize')


# ═══════════════════════════════════════════════════════════════
# eMBB CONTROLLER  (SST=1, SD=000001)
# KPI: Throughput (Mbps)  |  Action: Expand / Contract slice BW
# ═══════════════════════════════════════════════════════════════

class EmbbController:
    WINDOW        = 24      # 2 min of history (24 × 5s steps)
    HORIZON       = 6       # predict 30 s ahead
    HIDDEN        = 64
    LAYERS        = 2
    INIT_DL       = 500     # Mbps
    INIT_UL       = 250
    MIN_DL        = 250
    MAX_DL        = 2500
    MIN_UL        = 125
    MAX_UL        = 1250
    HIGH_RATIO    = 0.80    # expand when demand > 80% of current BW
    LOW_RATIO     = 0.40    # contract when demand < 40%

    def __init__(self, shared: SharedState):
        self.shared = shared
        saved = os.path.join(BASE_DIR, '5g-slicing-embb', 'models', 'saved')

        self.model = UnivariateLSTM(self.HIDDEN, self.LAYERS, self.HORIZON)
        self.model.load_state_dict(
            torch.load(os.path.join(saved, 'lstm_embb.pth'), weights_only=True))
        self.model.eval()

        self.sc_min   = np.load(os.path.join(saved, 'scaler_min.npy'))
        self.sc_scale = np.load(os.path.join(saved, 'scaler_scale.npy'))

        self.bw_dl    = self.INIT_DL
        self.bw_ul    = self.INIT_UL
        self.decisions = []

    # ── helpers ──────────────────────────────────────────────
    def _scale(self, v):   return (v - self.sc_min[0]) * self.sc_scale[0]
    def _inv(self, v):     return v / self.sc_scale[0] + self.sc_min[0]

    def predict(self, window):
        x = torch.FloatTensor(
            np.array([self._scale(v) for v in window])
        ).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            out = self.model(x).numpy()[0]
        return np.array([self._inv(v) for v in out])

    def decide(self, preds):
        peak = float(max(preds))
        avg  = float(np.mean(preds))

        # ── CROSS-SLICE RULE ──────────────────────────────────
        # URLLC is under latency pressure → freeze eMBB expansion
        if self.shared.urllc_is_critical() and peak > self.bw_dl * self.HIGH_RATIO:
            return 'hold_urllc_priority', peak

        if peak > self.bw_dl * self.HIGH_RATIO and self.bw_dl < self.MAX_DL:
            return 'expand', peak
        if avg  < self.bw_dl * self.LOW_RATIO  and self.bw_dl > self.MIN_DL:
            return 'contract', avg
        return 'hold', avg

    def apply(self, action, value):
        ts = datetime.now().isoformat()

        if action == 'expand':
            step = max(1, int(self.bw_dl * 0.25))
            self.bw_dl = min(self.MAX_DL, self.bw_dl + step)
            self.bw_ul = min(self.MAX_UL, self.bw_ul + max(1, int(self.bw_ul * 0.25)))
            print(f"  [eMBB ] EXPAND   → DL={self.bw_dl} Mbps  (predicted peak: {value:.1f} Mbps)")

        elif action == 'contract':
            step = max(1, int(self.bw_dl * 0.20))
            self.bw_dl = max(self.MIN_DL, self.bw_dl - step)
            self.bw_ul = max(self.MIN_UL, self.bw_ul - max(1, int(self.bw_ul * 0.20)))
            print(f"  [eMBB ] CONTRACT  → DL={self.bw_dl} Mbps  (predicted avg: {value:.1f} Mbps)")

        elif action == 'hold_urllc_priority':
            print(f"  [eMBB ] BLOCKED   → expansion refused (URLLC={self.shared.urllc_status}, "
                  f"predicted: {value:.1f} Mbps)")
        else:
            pass  # hold — silent to reduce noise

        self.shared.embb_bw_dl = self.bw_dl
        self.decisions.append({
            'slice':          'eMBB',
            'timestamp':      ts,
            'action':         action,
            'trigger_mbps':   round(value, 2),
            'bw_dl_mbps':     self.bw_dl,
            'bw_ul_mbps':     self.bw_ul,
            'urllc_status':   self.shared.urllc_status,
        })

    def run_simulation(self, data):
        print("\n[eMBB ] Running simulation ...")
        for i in range(self.WINDOW, len(data)):
            preds  = self.predict(data[i - self.WINDOW:i])
            action, value = self.decide(preds)
            self.apply(action, value)

        actions = [d['action'] for d in self.decisions]
        blocked = actions.count('hold_urllc_priority')
        print(f"  [eMBB ] Done — expand:{actions.count('expand')}  "
              f"contract:{actions.count('contract')}  "
              f"hold:{actions.count('hold')}  "
              f"blocked by URLLC:{blocked}")
        return self.decisions


# ═══════════════════════════════════════════════════════════════
# mMTC CONTROLLER  (SST=1, SD=000002)
# KPI: Packet rate (pps)  |  Action: Expand / Contract capacity
# ═══════════════════════════════════════════════════════════════

class MmtcController:
    WINDOW     = 24
    HORIZON    = 6
    HIDDEN     = 128
    LAYERS     = 3
    INIT_CAP   = 500    # pps baseline
    MIN_CAP    = 200
    MAX_CAP    = 2000
    INIT_DL    = 10     # Mbps (IoT uses very little bandwidth)
    MAX_DL     = 50
    MIN_DL     = 5
    INIT_UL    = 5
    MAX_UL     = 25
    MIN_UL     = 2
    HIGH_RATIO = 0.80
    LOW_RATIO  = 0.30

    def __init__(self, shared: SharedState):
        self.shared = shared
        saved = os.path.join(BASE_DIR, '5g-slicing-mmtc', 'models', 'saved')

        self.model = UnivariateLSTM(self.HIDDEN, self.LAYERS, self.HORIZON)
        self.model.load_state_dict(
            torch.load(os.path.join(saved, 'lstm_mmtc.pth'), weights_only=True))
        self.model.eval()

        self.sc_min   = np.load(os.path.join(saved, 'scaler_min.npy'))
        self.sc_scale = np.load(os.path.join(saved, 'scaler_scale.npy'))

        self.cap      = self.INIT_CAP
        self.bw_dl    = self.INIT_DL
        self.bw_ul    = self.INIT_UL
        self.decisions = []

    def _scale(self, v):   return (v - self.sc_min[0]) * self.sc_scale[0]
    def _inv(self, v):     return v / self.sc_scale[0] + self.sc_min[0]

    def predict(self, window):
        x = torch.FloatTensor(
            np.array([self._scale(v) for v in window])
        ).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            out = self.model(x).numpy()[0]
        return np.array([self._inv(v) for v in out])

    def decide(self, preds):
        peak = float(max(preds))
        avg  = float(np.mean(preds))
        if peak > self.cap * self.HIGH_RATIO and self.cap < self.MAX_CAP:
            return 'expand', peak
        if avg  < self.cap * self.LOW_RATIO  and self.cap > self.MIN_CAP:
            return 'contract', avg
        return 'hold', avg

    def apply(self, action, value):
        ts = datetime.now().isoformat()

        if action == 'expand':
            self.cap   = min(self.MAX_CAP, self.cap + max(1, int(self.cap * 0.30)))
            self.bw_dl = min(self.MAX_DL,  self.bw_dl + max(1, int(self.bw_dl * 0.30)))
            self.bw_ul = min(self.MAX_UL,  self.bw_ul + max(1, int(self.bw_ul * 0.30)))
            print(f"  [mMTC ] EXPAND   → cap={self.cap} pps, DL={self.bw_dl} Mbps  "
                  f"(peak: {value:.1f} pps)")
        elif action == 'contract':
            self.cap   = max(self.MIN_CAP, self.cap - max(1, int(self.cap * 0.20)))
            self.bw_dl = max(self.MIN_DL,  self.bw_dl - max(1, int(self.bw_dl * 0.20)))
            self.bw_ul = max(self.MIN_UL,  self.bw_ul - max(1, int(self.bw_ul * 0.20)))
            print(f"  [mMTC ] CONTRACT  → cap={self.cap} pps, DL={self.bw_dl} Mbps  "
                  f"(avg: {value:.1f} pps)")
        else:
            pass  # hold — silent

        self.shared.mmtc_bw_dl = self.bw_dl
        self.decisions.append({
            'slice':        'mMTC',
            'timestamp':    ts,
            'action':       action,
            'trigger_pps':  round(value, 2),
            'capacity_pps': self.cap,
            'bw_dl_mbps':   self.bw_dl,
            'bw_ul_mbps':   self.bw_ul,
        })

    def run_simulation(self, data):
        print("\n[mMTC ] Running simulation ...")
        for i in range(self.WINDOW, len(data)):
            preds  = self.predict(data[i - self.WINDOW:i])
            action, value = self.decide(preds)
            self.apply(action, value)

        actions = [d['action'] for d in self.decisions]
        print(f"  [mMTC ] Done — expand:{actions.count('expand')}  "
              f"contract:{actions.count('contract')}  "
              f"hold:{actions.count('hold')}")
        return self.decisions


# ═══════════════════════════════════════════════════════════════
# URLLC CONTROLLER  (SST=2, SD=000002)
# KPI: Latency (ms)  |  Action: ARP priority + GBR boost
# ═══════════════════════════════════════════════════════════════

class UrllcController:
    WINDOW        = 10
    HORIZON       = 3
    HIDDEN        = 128
    LAYERS        = 2
    N_FEAT        = 4
    SMOOTH_WIN    = 3
    SLA           = 10.0    # ms — hard SLA, breach is unacceptable
    PROACTIVE     = 8.0     # ms — soft threshold, act before breach
    RESTORE       = 3.0     # ms — safe zone, release extra resources
    INIT_GBR_DL   = 50      # Mbps
    INIT_GBR_UL   = 25
    MAX_GBR_DL    = 150
    MAX_GBR_UL    = 75
    MIN_GBR_DL    = 20
    MIN_GBR_UL    = 10

    def __init__(self, shared: SharedState):
        self.shared = shared
        model_path = os.path.join(BASE_DIR, '5g-slicing-urllc', 'saved', 'lstm_urllc.pth')
        data_path  = os.path.join(BASE_DIR, '5g-slicing-urllc', 'data', 'Training_data.csv')

        self.model = MultivariateLSTM(self.N_FEAT, self.HIDDEN, self.LAYERS, self.HORIZON)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

        self.feat_sc, self.tgt_sc = self._build_scalers(data_path)
        self.arp     = 2               # normal ARP priority
        self.gbr_dl  = self.INIT_GBR_DL
        self.gbr_ul  = self.INIT_GBR_UL
        self.decisions = []

    def _build_scalers(self, data_path):
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()
        df['latency_ms'] = pd.to_numeric(df['latency_ms'], errors='coerce')
        df.dropna(subset=['latency_ms'], inplace=True)

        agg = (df.groupby('seq')
                 .agg(max_lat=('latency_ms','max'), mean_lat=('latency_ms','mean'),
                      std_lat=('latency_ms','std'),
                      total_recv=('recv','sum'), total_lost=('lost','sum'))
                 .fillna(0).sort_index().reset_index())
        agg['loss_rate'] = agg['total_lost'] / (agg['total_recv'] + 1e-6)

        feats = agg[['max_lat','mean_lat','std_lat','loss_rate']].values.astype(np.float32)
        tgt   = (pd.Series(agg['max_lat'].values.astype(np.float32))
                   .rolling(self.SMOOTH_WIN, min_periods=1).mean().values.astype(np.float32))

        # Build sequences and fit scalers on train partition only
        Xs, ys = [], []
        for i in range(len(tgt) - self.WINDOW - self.HORIZON + 1):
            Xs.append(feats[i:i+self.WINDOW])
            ys.append(tgt[i+self.WINDOW:i+self.WINDOW+self.HORIZON])
        X = np.array(Xs, dtype=np.float32)
        y = np.array(ys, dtype=np.float32)
        t = int(0.7 * len(X))

        fs = MinMaxScaler().fit(X[:t].reshape(-1, self.N_FEAT))
        ts_ = MinMaxScaler().fit(y[:t].reshape(-1, 1))
        return fs, ts_

    def predict(self, window):
        scaled = self.feat_sc.transform(window)
        x = torch.FloatTensor(scaled).unsqueeze(0)
        with torch.no_grad():
            out = self.model(x).numpy()[0]
        return self.tgt_sc.inverse_transform(out.reshape(-1, 1)).flatten()

    def decide(self, preds):
        peak = float(np.max(preds))
        avg  = float(np.mean(preds))
        if peak > self.SLA:
            return 'prioritize_urgent', peak
        if peak > self.PROACTIVE:
            return 'prioritize', peak
        if avg < self.RESTORE and (self.arp < 2 or self.gbr_dl > self.INIT_GBR_DL):
            return 'restore', avg
        return 'hold', avg

    def apply(self, action, value, preds):
        ts = datetime.now().isoformat()

        if action == 'prioritize_urgent':
            self.arp    = 1
            self.gbr_dl = min(self.MAX_GBR_DL, int(self.gbr_dl * 1.50))
            self.gbr_ul = min(self.MAX_GBR_UL, int(self.gbr_ul * 1.50))
            print(f"  [URLLC] !! URGENT    → RTT={value:.2f}ms > {self.SLA}ms  | "
                  f"ARP=1, GBR={self.gbr_dl} Mbps")

        elif action == 'prioritize':
            self.arp    = 1
            self.gbr_dl = min(self.MAX_GBR_DL, int(self.gbr_dl * 1.25))
            self.gbr_ul = min(self.MAX_GBR_UL, int(self.gbr_ul * 1.25))
            print(f"  [URLLC] ^^ PROACTIVE → RTT={value:.2f}ms > {self.PROACTIVE}ms | "
                  f"ARP=1, GBR={self.gbr_dl} Mbps")

        elif action == 'restore':
            self.arp    = 2
            self.gbr_dl = max(self.INIT_GBR_DL,
                              max(self.MIN_GBR_DL, int(self.gbr_dl * 0.80)))
            self.gbr_ul = max(self.INIT_GBR_UL,
                              max(self.MIN_GBR_UL, int(self.gbr_ul * 0.80)))
            print(f"  [URLLC] vv RESTORE   → RTT={value:.2f}ms < {self.RESTORE}ms  | "
                  f"ARP=2, GBR={self.gbr_dl} Mbps")
        else:
            pass  # hold — silent

        # Publish URLLC status so eMBB can react
        self.shared.urllc_status  = action
        self.shared.urllc_gbr_dl  = self.gbr_dl

        self.decisions.append({
            'slice':            'URLLC',
            'timestamp':        ts,
            'action':           action,
            'trigger_value_ms': round(value, 3),
            'predicted_ms':     [round(float(p), 3) for p in preds],
            'arp_priority':     self.arp,
            'gbr_dl_mbps':      self.gbr_dl,
            'gbr_ul_mbps':      self.gbr_ul,
        })

    def run_simulation(self, data):
        print("\n[URLLC] Running simulation ...")
        for i in range(self.WINDOW, len(data)):
            preds  = self.predict(data[i - self.WINDOW:i])
            action, value = self.decide(preds)
            self.apply(action, value, preds)

        actions = [d['action'] for d in self.decisions]
        print(f"  [URLLC] Done — urgent:{actions.count('prioritize_urgent')}  "
              f"proactive:{actions.count('prioritize')}  "
              f"restore:{actions.count('restore')}  "
              f"hold:{actions.count('hold')}")
        return self.decisions


# ═══════════════════════════════════════════════════════════════
# UNIFIED ORCHESTRATOR  — ties everything together
# ═══════════════════════════════════════════════════════════════

class UnifiedOrchestrator:

    def __init__(self):
        print("=" * 65)
        print("  Unified 5G Slice Orchestrator — Zero-Touch Network Mgmt")
        print("=" * 65)
        print("  Slices managed:")
        print("    eMBB   SST=1, SD=000001  →  4K video, max bandwidth")
        print("    mMTC   SST=1, SD=000002  →  IoT sensors, massive devices")
        print("    URLLC  SST=2, SD=000002  →  mission-critical, <10ms RTT")
        print()
        print("  Cross-slice rule:")
        print("    URLLC URGENT/PROACTIVE  →  eMBB expansion BLOCKED")
        print()

        self.shared = SharedState()

        print("Loading LSTM models ...")
        self.embb  = EmbbController(self.shared)
        print("  [eMBB ] model loaded (hidden=64, layers=2, predicts throughput)")
        self.mmtc  = MmtcController(self.shared)
        print("  [mMTC ] model loaded (hidden=128, layers=3, predicts packet_rate)")
        self.urllc = UrllcController(self.shared)
        print("  [URLLC] model loaded (hidden=128, layers=2, predicts latency ms)")
        print("\nAll 3 models ready.\n")

    # ── data loading helpers ──────────────────────────────────
    def _load_urllc_features(self):
        df = pd.read_csv(
            os.path.join(BASE_DIR, '5g-slicing-urllc', 'data', 'Training_data.csv'))
        df.columns = df.columns.str.strip()
        df['latency_ms'] = pd.to_numeric(df['latency_ms'], errors='coerce')
        df.dropna(subset=['latency_ms'], inplace=True)
        agg = (df.groupby('seq')
                 .agg(max_lat=('latency_ms','max'), mean_lat=('latency_ms','mean'),
                      std_lat=('latency_ms','std'),
                      total_recv=('recv','sum'), total_lost=('lost','sum'))
                 .fillna(0).sort_index().reset_index())
        agg['loss_rate'] = agg['total_lost'] / (agg['total_recv'] + 1e-6)
        return agg[['max_lat','mean_lat','std_lat','loss_rate']].values.astype(np.float32)

    # ── main simulation loop ──────────────────────────────────
    def run_simulation(self):
        print("Loading datasets ...")
        embb_data  = pd.read_csv(
            os.path.join(BASE_DIR, '5g-slicing-embb', 'data',
                         'embb_traffic_timeseries.csv'))['throughput_mbps'].values
        mmtc_data  = pd.read_csv(
            os.path.join(BASE_DIR, '5g-slicing-mmtc', 'data',
                         'mmtc_traffic_timeseries.csv'))['packet_rate'].values
        urllc_data = self._load_urllc_features()

        print(f"  eMBB:  {len(embb_data)} timesteps")
        print(f"  mMTC:  {len(mmtc_data)} timesteps")
        print(f"  URLLC: {len(urllc_data)} timesteps")

        # ── Execution order matters ──────────────────────────
        # URLLC runs first → sets shared.urllc_status
        # eMBB runs second → reads shared.urllc_status to decide whether to expand
        # mMTC runs third  → independent (IoT sensors don't compete with latency)
        print("\n--- Phase 1: URLLC (sets cross-slice priority) ---")
        urllc_dec = self.urllc.run_simulation(urllc_data)

        print("\n--- Phase 2: eMBB (cross-slice aware) ---")
        embb_dec  = self.embb.run_simulation(embb_data)

        print("\n--- Phase 3: mMTC (independent) ---")
        mmtc_dec  = self.mmtc.run_simulation(mmtc_data)

        # ── Summary ──────────────────────────────────────────
        all_decisions = urllc_dec + embb_dec + mmtc_dec
        blocked = sum(1 for d in embb_dec if d['action'] == 'hold_urllc_priority')
        urgent  = sum(1 for d in urllc_dec if d['action'] == 'prioritize_urgent')

        print("\n" + "=" * 65)
        print("  FINAL SUMMARY")
        print("=" * 65)
        print(f"  Total control decisions logged : {len(all_decisions)}")
        print(f"  URLLC urgent latency events    : {urgent}")
        print(f"  eMBB expansions blocked by URLLC: {blocked}")
        print()
        print(f"  Final eMBB  bandwidth  : DL={self.embb.bw_dl} Mbps, UL={self.embb.bw_ul} Mbps")
        print(f"  Final mMTC  capacity   : {self.mmtc.cap} pps, DL={self.mmtc.bw_dl} Mbps")
        print(f"  Final URLLC GBR        : DL={self.urllc.gbr_dl} Mbps, ARP={self.urllc.arp}")

        # Save unified decision log
        out = os.path.join(BASE_DIR, 'unified_decisions.json')
        with open(out, 'w') as f:
            json.dump(all_decisions, f, indent=2)
        print(f"\n  All decisions saved → {out}")
        return all_decisions


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified 5G Slice Orchestrator')
    parser.add_argument('--live', action='store_true',
                        help='Live mode (reads from running 5G network)')
    args = parser.parse_args()

    orchestrator = UnifiedOrchestrator()

    if args.live:
        print("Live mode not yet implemented — run simulation mode first.")
    else:
        orchestrator.run_simulation()
