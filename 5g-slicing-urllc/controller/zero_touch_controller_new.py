"""
Zero-Touch URLLC Slice QoS Controller.
- Loads trained LSTM model from models/lstm_predictor.py
- Predicts future latency demand over a rolling window
- Proactively adjusts QoS (5QI + ARP priority) via Open5GS WebUI API
- Logs all decisions for evaluation

Usage:
    python3 zero_touch_controller_new.py                     # Simulation mode (default)
    python3 zero_touch_controller_new.py --mode live         # Live mode (polls CSV)
    python3 zero_touch_controller_new.py --mode sim          # Explicit simulation mode
"""

import numpy as np
import torch
import requests
import time
import json
import csv
import re
import os
import argparse
import pandas as pd
from datetime import datetime

# ── Model imports ──────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))
from lstm_predictor import (
    TrafficLSTM,
    build_features,
    WINDOW_SIZE,
    HORIZON,
    HIDDEN_SIZE,
    NUM_LAYERS,
    N_FEATURES,
)

# ── Config ─────────────────────────────────────────────────────────────────────
WEBUI_URL   = "http://127.0.0.1:9999"
WEBUI_USER  = "admin"
WEBUI_PASS  = "1423"
OPEN5GS_API = f"{WEBUI_URL}/api"

CHECK_INTERVAL = 1          # seconds between live-mode polls

SST_TARGET            = 2   # URLLC slice SST
HIGH_LATENCY_THRESHOLD = 2.0  # ms — elevate QoS when predicted latency exceeds this
LOW_LATENCY_THRESHOLD  = 1.0  # ms — relax QoS when predicted latency drops below this

# QoS profiles applied to every URLLC subscriber (SST=2)
QOS_NORMAL = {
    "5qi":          85,
    "arp_priority": 5,
    "pre_emp_cap":  2,
    "pre_emp_vuln": 1,
    "state_name":   "NORMAL",
}

QOS_ELEVATED = {
    "5qi":          82,
    "arp_priority": 1,
    "pre_emp_cap":  1,
    "pre_emp_vuln": 2,
    "state_name":   "ELEVATED (CRITICAL)",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Controller ─────────────────────────────────────────────────────────────────
class URLLCController:

    def __init__(self, model_path=None):
        # Default model path
        if model_path is None:
            model_path = os.path.join(BASE_DIR, '..', 'saved', 'lstm_urllc.pth')

        # ── Load LSTM model ────────────────────────────────────────────────────
        self.model = TrafficLSTM(N_FEATURES, HIDDEN_SIZE, NUM_LAYERS, HORIZON)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

        # ── Load scalers saved by lstm_predictor.py ────────────────────────────
        saved_dir = os.path.join(BASE_DIR, '..', 'saved')
        self.feat_scaler_min   = np.load(os.path.join(saved_dir, 'feat_scaler_min.npy'))
        self.feat_scaler_scale = np.load(os.path.join(saved_dir, 'feat_scaler_scale.npy'))
        self.tgt_scaler_min    = np.load(os.path.join(saved_dir, 'tgt_scaler_min.npy'))
        self.tgt_scaler_scale  = np.load(os.path.join(saved_dir, 'tgt_scaler_scale.npy'))

        # ── State ──────────────────────────────────────────────────────────────
        self.latency_buffer = []      # rolling buffer of raw latency values (live mode)
        self.current_state  = QOS_NORMAL['state_name']
        self.decisions      = []

        # ── Open5GS authentication ─────────────────────────────────────────────
        self.auth_token  = None
        self.csrf_token  = None
        self.session     = requests.Session()
        self.authenticate()

    # ── Scaler helpers ─────────────────────────────────────────────────────────

    def _scale_features(self, arr: np.ndarray) -> np.ndarray:
        """MinMax-scale a (W, N_FEATURES) window using the saved feature scaler."""
        return (arr - self.feat_scaler_min) * self.feat_scaler_scale

    def _inverse_scale_target(self, arr: np.ndarray) -> np.ndarray:
        """Invert MinMax scaling on a predicted target array."""
        return arr / self.tgt_scaler_scale[0] + self.tgt_scaler_min[0]

    # ── Prediction ─────────────────────────────────────────────────────────────

    def predict(self, raw_latency_window: np.ndarray) -> np.ndarray:
        """
        Predict next HORIZON latency steps from a window of raw latency values.

        Parameters
        ----------
        raw_latency_window : (WINDOW_SIZE,) float32 array of raw latency (ms)

        Returns
        -------
        predictions : (HORIZON,) array of predicted latency values (ms)
        """
        # Build the 3-feature matrix [raw, rolling_mean, rolling_std]
        features = build_features(raw_latency_window)          # (W, 3)

        # Scale features
        features_scaled = self._scale_features(features)       # (W, 3)

        # Run LSTM
        x = torch.FloatTensor(features_scaled).unsqueeze(0)    # (1, W, 3)
        with torch.no_grad():
            pred_scaled = self.model(x).numpy()[0]             # (HORIZON,)

        # Inverse-scale predictions back to ms
        return self._inverse_scale_target(pred_scaled)

    # ── Decision logic ─────────────────────────────────────────────────────────

    def decide_action(self, predicted_latencies: np.ndarray):
        """
        Choose an action based on mean predicted latency and current QoS state.

        Returns
        -------
        action : str  — 'elevate', 'relax', 'hold', 'hold_max', or 'hold_min'
        value  : float — mean predicted latency that triggered the decision
        """
        if len(predicted_latencies) == 0:
            return 'hold', 0.0

        mean_latency = float(np.mean(predicted_latencies))

        if mean_latency > HIGH_LATENCY_THRESHOLD:
            if self.current_state != QOS_ELEVATED['state_name']:
                return 'elevate', mean_latency
            return 'hold_max', mean_latency

        if mean_latency < LOW_LATENCY_THRESHOLD:
            if self.current_state != QOS_NORMAL['state_name']:
                return 'relax', mean_latency
            return 'hold_min', mean_latency

        return 'hold', mean_latency

    # ── QoS application ────────────────────────────────────────────────────────

    def apply_qos(self, action: str, trigger_value: float):
        """
        Print decision, update internal state, log it, and push to Open5GS.
        'hold' actions are logged but do NOT call the Open5GS API.
        """
        timestamp = datetime.now().isoformat()

        if action == 'elevate':
            target_qos = QOS_ELEVATED
            print(
                f"[{timestamp}] ⚠️  ALERT: Latency {trigger_value:.2f} ms > HIGH ({HIGH_LATENCY_THRESHOLD} ms)\n"
                f"    ↳ ELEVATING QoS → 5QI: {target_qos['5qi']}, ARP: {target_qos['arp_priority']}"
            )

        elif action == 'relax':
            target_qos = QOS_NORMAL
            print(
                f"[{timestamp}] ✅ STABLE: Latency {trigger_value:.2f} ms < LOW ({LOW_LATENCY_THRESHOLD} ms)\n"
                f"    ↳ RELAXING QoS → 5QI: {target_qos['5qi']}, ARP: {target_qos['arp_priority']}"
            )

        elif action == 'hold_max':
            print(
                f"[{timestamp}] ⏸️  HOLD: Latency {trigger_value:.2f} ms > HIGH "
                f"({HIGH_LATENCY_THRESHOLD} ms) — already at max QoS ({self.current_state})"
            )
            self.decisions.append({
                'timestamp':       timestamp,
                'action':          'hold',
                'trigger_latency': round(trigger_value, 2),
                'state':           self.current_state,
                'applied_5qi':     None,
                'applied_arp':     None,
            })
            return
        elif action == 'hold_min':
            print(
                f"[{timestamp}] ⏸️  HOLD: Latency {trigger_value:.2f} ms < LOW "
                f"({LOW_LATENCY_THRESHOLD} ms) — already at min QoS ({self.current_state})"
            )
            self.decisions.append({
                'timestamp':       timestamp,
                'action':          'hold',
                'trigger_latency': round(trigger_value, 2),
                'state':           self.current_state,
                'applied_5qi':     None,
                'applied_arp':     None,
            })
            return
        else:  # hold — truly within band
            print(
                f"[{timestamp}] ⏸️  HOLD: Latency {trigger_value:.2f} ms within band "
                f"({LOW_LATENCY_THRESHOLD} ms – {HIGH_LATENCY_THRESHOLD} ms)"
            )
            self.decisions.append({
                'timestamp':       timestamp,
                'action':          'hold',
                'trigger_latency': round(trigger_value, 2),
                'state':           self.current_state,
                'applied_5qi':     None,
                'applied_arp':     None,
            })
            return

        # Update state and log
        self.current_state = target_qos['state_name']
        self.decisions.append({
            'timestamp':       timestamp,
            'action':          action,
            'trigger_latency': round(trigger_value, 2),
            'state':           self.current_state,
            'applied_5qi':     target_qos['5qi'],
            'applied_arp':     target_qos['arp_priority'],
        })

        # Push to Open5GS
        self.update_open5gs_subscribers(target_qos)

    # ── Open5GS API ────────────────────────────────────────────────────────────

    def authenticate(self):
        """Log in to Open5GS WebUI using CSRF-token + JWT flow."""
        print("🔐 Authenticating with Open5GS WebUI...")
        try:
            # Step 1: fetch homepage → get CSRF token and connect.sid cookie
            home = self.session.get(WEBUI_URL)
            match = re.search(r'__NEXT_DATA__\s*=\s*({.*?})\s*\n', home.text, re.DOTALL)
            if not match:
                raise Exception("Could not find __NEXT_DATA__ in homepage HTML.")

            next_data       = json.loads(match.group(1))
            self.csrf_token = next_data["props"]["initialProps"]["session"]["csrfToken"]
            print(f"    [+] CSRF token: {self.csrf_token[:30]}...")

            # Step 2: login (same session carries the connect.sid cookie)
            self.session.post(
                f"{WEBUI_URL}/api/auth/login",
                json={"username": WEBUI_USER, "password": WEBUI_PASS},
                headers={
                    "Content-Type": "application/json",
                    "X-CSRF-Token": self.csrf_token,
                },
                allow_redirects=False,
            )

            # Step 3: get fresh CSRF token + JWT from /session endpoint
            sess_resp = self.session.get(
                f"{WEBUI_URL}/api/auth/session",
                headers={"X-CSRF-Token": self.csrf_token},
            )
            sess_data       = sess_resp.json()
            self.csrf_token = sess_data.get("csrfToken", self.csrf_token)
            self.auth_token = sess_data.get("authToken")

            if not self.auth_token:
                raise Exception("No authToken returned — check credentials.")

            print(f"    [+] JWT authToken: {self.auth_token[:40]}...")
            print("    [+] Authentication Successful!\n")

        except Exception as e:
            print(f"    [!] Authentication Failed: {e}")
            self.auth_token = None

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type":  "application/json",
            "X-CSRF-Token":  self.csrf_token,
        }

    def update_open5gs_subscribers(self, target_qos: dict):
        """Update all URLLC subscribers (SST=2) via the Open5GS REST API."""
        if not self.auth_token:
            print("  [!] Cannot update — not authenticated.")
            return
        try:
            resp = self.session.get(
                f"{OPEN5GS_API}/db/Subscriber",
                headers=self._headers(),
            )
            resp.raise_for_status()
            subscribers = resp.json()

            updated_count = 0
            for sub in subscribers:
                imsi     = sub.get('imsi')
                modified = False

                for s_nssai in sub.get('slice', []):
                    if s_nssai.get('sst') == SST_TARGET:
                        for sess in s_nssai.get('session', []):
                            sess['qos']['index']                           = target_qos['5qi']
                            sess['qos']['arp']['priority_level']           = target_qos['arp_priority']
                            sess['qos']['arp']['pre_emption_capability']   = target_qos['pre_emp_cap']
                            sess['qos']['arp']['pre_emption_vulnerability'] = target_qos['pre_emp_vuln']
                            modified = True

                if modified:
                    update_resp = self.session.put(
                        f"{OPEN5GS_API}/db/Subscriber/{imsi}",
                        json=sub,
                        headers=self._headers(),
                    )
                    update_resp.raise_for_status()
                    updated_count += 1
                    print(f"    [OK] Updated IMSI: {imsi}")

            if updated_count > 0:
                print(f"    [OK] Pushed QoS update to {updated_count} URLLC subscriber(s).")
            else:
                print(f"    [!] No subscribers found with SST={SST_TARGET}.")

        except Exception as e:
            print(f"  [!] Open5GS API Error: {e}")

    # ── Live data helper ───────────────────────────────────────────────────────

    def get_current_latency(self, csv_path: str) -> float:
        """Read the most recent latency_ms value from the rolling CSV."""
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows   = list(reader)
                if rows and 'latency_ms' in rows[-1]:
                    return float(rows[-1]['latency_ms'])
        except Exception:
            pass
        return 0.0

    # ── Run modes ──────────────────────────────────────────────────────────────

    def run_simulation(self, simulation_data: np.ndarray):
        """
        Offline simulation mode: replay a 1-D latency array.

        Parameters
        ----------
        simulation_data : (T,) float32 array of latency values (ms)
        """
        print("=" * 60)
        print("  Zero-Touch URLLC Controller  —  SIMULATION MODE")
        print("=" * 60)
        print(f"  LSTM: window={WINDOW_SIZE} steps | horizon={HORIZON} steps")
        print(f"  Features per step: {N_FEATURES} (raw, rolling_mean, rolling_std)")
        print(f"  Thresholds — HIGH: {HIGH_LATENCY_THRESHOLD} ms | LOW: {LOW_LATENCY_THRESHOLD} ms")
        print(f"  Processing {len(simulation_data)} data points...\n")

        T = len(simulation_data)
        for i in range(WINDOW_SIZE, T):
            window      = simulation_data[i - WINDOW_SIZE : i]  # (WINDOW_SIZE,)
            predictions = self.predict(window)                    # (HORIZON,)
            action, mean_val = self.decide_action(predictions)
            self.apply_qos(action, mean_val)

        # Summary
        actions = [d['action'] for d in self.decisions]
        print(f"\n{'=' * 60}")
        print(
            f"  Summary: {actions.count('elevate')} elevations, "
            f"{actions.count('relax')} relaxations, "
            f"{actions.count('hold')} holds"
        )
        print(f"  Total logged decisions: {len(self.decisions)}")
        return self.decisions

    def run_live(self, csv_path: str):
        """
        Live mode: poll the latest latency from a CSV on disk every CHECK_INTERVAL seconds.

        Parameters
        ----------
        csv_path : path to the rolling latency CSV (must have a 'latency_ms' column)
        """
        print("=" * 60)
        print("  Zero-Touch URLLC Controller  —  LIVE MODE")
        print("=" * 60)
        print(f"  Polling '{csv_path}' every {CHECK_INTERVAL}s.")
        print(f"  Warming up: need {WINDOW_SIZE} samples before first prediction.")
        print(f"  Press Ctrl+C to stop.\n")

        try:
            while True:
                latency = self.get_current_latency(csv_path)
                self.latency_buffer.append(latency)

                # Keep only the last WINDOW_SIZE readings
                if len(self.latency_buffer) > WINDOW_SIZE:
                    self.latency_buffer.pop(0)

                if len(self.latency_buffer) == WINDOW_SIZE:
                    window      = np.array(self.latency_buffer, dtype=np.float32)
                    predictions = self.predict(window)
                    print(f"[LIVE] Predictions (next {HORIZON} steps): {np.round(predictions, 3)}")
                    action, mean_val = self.decide_action(predictions)
                    self.apply_qos(action, mean_val)
                else:
                    remaining = WINDOW_SIZE - len(self.latency_buffer)
                    print(f"[LIVE] Warming up… {remaining} more sample(s) needed.")

                time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            print("\n\n[LIVE] Stopped by user.")
            print(f"Total logged decisions: {len(self.decisions)}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Zero-Touch URLLC QoS Controller — simulation or live mode",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["sim", "live"],
        default="sim",
        help="'sim'  → replay a CSV in simulation mode (default)\n"
             "'live' → poll get_current_latency() in real time",
    )
    parser.add_argument(
        "--model",
        default=os.path.join(BASE_DIR, '..', 'saved', 'lstm_urllc.pth'),
        help="Path to the trained LSTM model (default: ../saved/lstm_urllc.pth)",
    )
    parser.add_argument(
        "--data",
        default=os.path.join(BASE_DIR, '..', 'data', 'urllc_timeseries.csv'),
        help="Path to the latency CSV (default: ../data/urllc_timeseries.csv)",
    )
    args = parser.parse_args()

    controller = URLLCController(model_path=args.model)

    if args.mode == "live":
        controller.run_live(args.data)
    else:
        # Simulation mode — load CSV and replay
        df = pd.read_csv(args.data)
        df.columns = df.columns.str.strip()
        df['latency_ms'] = pd.to_numeric(df['latency_ms'], errors='coerce')
        df.dropna(subset=['latency_ms'], inplace=True)

        latency_data = df['latency_ms'].values.astype(np.float32)

        decisions = controller.run_simulation(simulation_data=latency_data)

        # Save decision log
        output_path = os.path.join(BASE_DIR, '..', 'data', 'controller_decisions_urllc.json')
        with open(output_path, 'w') as f:
            json.dump(decisions, f, indent=2)
        print(f"\nDecisions saved → {output_path}")
