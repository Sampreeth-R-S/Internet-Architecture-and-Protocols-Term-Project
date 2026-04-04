
import numpy as np
import time
import requests
import json
import re
import os
from datetime import datetime
import torch.nn as nn
import torch
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import argparse


# --- Configuration ---
WEBUI_URL = "http://127.0.0.1:9999"
OPEN5GS_API = f"{WEBUI_URL}/api"
CHECK_INTERVAL = 1

WEBUI_USER = "admin"
WEBUI_PASS = "1423"

SST_TARGET = 2
WINDOW_SIZE = 10
LATENCY_THRESHOLD = 4.0

QOS_NORMAL = {
    "5qi": 85,
    "arp_priority": 5,
    "pre_emp_cap": 2,
    "pre_emp_vuln": 1,
    "state_name": "NORMAL"
}

QOS_ELEVATED = {
    "5qi": 82,
    "arp_priority": 1,
    "pre_emp_cap": 1,
    "pre_emp_vuln": 2,
    "state_name": "ELEVATED (CRITICAL)"
}

WINDOW_SIZE  = 10
HORIZON      = 3
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
N_FEATURES   = 4    # [max_lat, mean_lat, std_lat, loss_rate]
SMOOTH_WINDOW = 3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..')

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


class URLLCController:
  
    def __init__(self,model_path):
        self.model = TrafficLSTM(N_FEATURES, HIDDEN_SIZE, NUM_LAYERS, HORIZON)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()
        self.feat_scaler, self.tgt_scaler = self._build_scalers()

        self.latency_buffer = []
        self.current_state = QOS_NORMAL['state_name']
        self.decisions = []
        self.auth_token = None
        self.csrf_token = None
        self.session = requests.Session()  # ONE session used everywhere
        self.authenticate()

    def _build_scalers(self):
        """Refit MinMaxScaler from training data - identical to lstm_model.py."""
        data_path = "./data/training_data.csv"
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

    def authenticate(self):
        """Logs into Open5GS WebUI using CSRF token + JWT flow."""
        print("🔐 Authenticating with Open5GS WebUI...")
        try:
            # Step 1: Fetch homepage — get CSRF token AND set connect.sid cookie
            home = self.session.get(WEBUI_URL)
            match = re.search(r'__NEXT_DATA__\s*=\s*({.*?})\s*\n', home.text, re.DOTALL)
            if not match:
                raise Exception("Could not find __NEXT_DATA__ in homepage HTML.")

            next_data = json.loads(match.group(1))
            self.csrf_token = next_data["props"]["initialProps"]["session"]["csrfToken"]
            print(f"    [+] CSRF token: {self.csrf_token[:30]}...")

            # Step 2: Login — same session (carries the connect.sid cookie)
            self.session.post(
                f"{WEBUI_URL}/api/auth/login",
                json={"username": WEBUI_USER, "password": WEBUI_PASS},
                headers={
                    "Content-Type": "application/json",
                    "X-CSRF-Token": self.csrf_token
                },
                allow_redirects=False
            )

            # Step 3: Get fresh CSRF token + JWT from session endpoint
            sess_resp = self.session.get(
                f"{WEBUI_URL}/api/auth/session",
                headers={"X-CSRF-Token": self.csrf_token}
            )
            sess_data = sess_resp.json()

            # Use the fresh CSRF token returned by /session for all future requests
            self.csrf_token = sess_data.get("csrfToken", self.csrf_token)
            self.auth_token = sess_data.get("authToken")

            if not self.auth_token:
                raise Exception("No authToken returned. Check credentials.")

            print(f"    [+] JWT authToken: {self.auth_token[:40]}...")
            print("    [+] Authentication Successful!\n")

        except Exception as e:
            print(f"    [!] Authentication Failed: {e}")
            self.auth_token = None

    def _headers(self):
        """Headers for JWT-authenticated + CSRF-safe requests."""
        return {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
            "X-CSRF-Token": self.csrf_token   # required for PUT/POST/DELETE
        }

    def evaluate_predictions(self, predicted_latencies):
        if len(predicted_latencies) == 0:
            return 'hold', 0.0
        mean_predicted_latency = np.mean(predicted_latencies)
        if mean_predicted_latency > LATENCY_THRESHOLD and self.current_state != QOS_ELEVATED['state_name']:
            return 'elevate', mean_predicted_latency
        elif mean_predicted_latency <= LATENCY_THRESHOLD and self.current_state != QOS_NORMAL['state_name']:
            return 'relax', mean_predicted_latency
        else:
            return 'hold', mean_predicted_latency

    def apply_qos(self, action, trigger_value):
        timestamp = datetime.now().isoformat()
        if action == 'elevate':
            target_qos = QOS_ELEVATED
            print(f"[{timestamp}] ⚠️  ALERT: Predicted latency {trigger_value:.2f}ms > {LATENCY_THRESHOLD}ms.")
            print(f"    ↳ ELEVATING Priority -> 5QI: {target_qos['5qi']}, ARP: {target_qos['arp_priority']}")
        elif action == 'relax':
            target_qos = QOS_NORMAL
            print(f"[{timestamp}] ✅ STABLE: Predicted latency {trigger_value:.2f}ms <= {LATENCY_THRESHOLD}ms.")
            print(f"    ↳ RELAXING Priority -> 5QI: {target_qos['5qi']}, ARP: {target_qos['arp_priority']}")
        else:
            return

        self.current_state = target_qos['state_name']
        self.decisions.append({
            'timestamp': timestamp,
            'action': action,
            'predicted_mean_latency': round(float(trigger_value), 2),
            'applied_5qi': target_qos['5qi'],
            'applied_arp': target_qos['arp_priority']
        })
        self.update_open5gs_subscribers(target_qos)

    def update_open5gs_subscribers(self, target_qos):
        """Updates only URLLC subscribers (SST=2) using the same authenticated session."""
        if not self.auth_token:
            print("  [!] Cannot update — not authenticated.")
            return

        try:
            # Use self.session for ALL requests (carries connect.sid cookie)
            resp = self.session.get(
                f"{OPEN5GS_API}/db/Subscriber",
                headers=self._headers()
            )
            resp.raise_for_status()
            subscribers = resp.json()

            updated_count = 0
            for sub in subscribers:
                imsi = sub.get('imsi')
                modified = False

                for s_nssai in sub.get('slice', []):
                    if s_nssai.get('sst') == SST_TARGET:
                        for sess in s_nssai.get('session', []):
                            sess['qos']['index'] = target_qos['5qi']
                            sess['qos']['arp']['priority_level'] = target_qos['arp_priority']
                            sess['qos']['arp']['pre_emption_capability'] = target_qos['pre_emp_cap']
                            sess['qos']['arp']['pre_emption_vulnerability'] = target_qos['pre_emp_vuln']
                            modified = True

                if modified:
                    update_resp = self.session.put(       # self.session, not requests
                        f"{OPEN5GS_API}/db/Subscriber/{imsi}",
                        json=sub,
                        headers=self._headers()
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

    def get_current_features(self,path):
        try:
            csv_path = path
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            df['latency_ms'] = pd.to_numeric(df['latency_ms'], errors='coerce')
            df.dropna(subset=['latency_ms'], inplace=True)
            df = df.tail(50)
            lats = df['latency_ms'].values
            recv = df['recv'].sum() if 'recv' in df.columns else 1
            lost = df['lost'].sum() if 'lost' in df.columns else 0
            return np.array([lats.max(), lats.mean(),
                             lats.std() if len(lats) > 1 else 0.0,
                             lost / (recv + 1e-6)], dtype=np.float32)
        except Exception:
            return np.zeros(N_FEATURES, dtype=np.float32)

    def run_simulation(self, simulation_data):
        print(f"Starting URLLC Simulation | Threshold: {LATENCY_THRESHOLD}ms\n")
        T = len(simulation_data)
        print(T)
        for i in range(WINDOW_SIZE, T):
                window = simulation_data[i - WINDOW_SIZE:i]   # (W, F)
                predictions = self.predict(window)
                print(predictions)
                action, mean_val = self.evaluate_predictions(predictions)
                self.apply_qos(action, mean_val)
                if action != 'hold':
                    time.sleep(1)

        print("\nSimulation Complete.")
        print(f"Total Interventions: {len(self.decisions)}")
    
    def run_live(self,csv_path):
        print(f"Starting URLLC Live Controller | Threshold: {LATENCY_THRESHOLD}ms")
        print(f"Polling every {CHECK_INTERVAL}s. Press Ctrl+C to stop.\n")

        feature_window = []  # rolling buffer of (N_FEATURES,) vectors

        try:
            while True:
                features = self.get_current_features(csv_path)   # (N_FEATURES,)
                feature_window.append(features)

                # Keep only the last WINDOW_SIZE entries
                if len(feature_window) > WINDOW_SIZE:
                    feature_window.pop(0)

                if len(feature_window) == WINDOW_SIZE:
                    window_arr = np.stack(feature_window)        # (W, F)
                    predictions = self.predict(window_arr)
                    print(f"[LIVE] Predictions (next {HORIZON} steps): {predictions}")
                    action, mean_val = self.evaluate_predictions(predictions)
                    self.apply_qos(action, mean_val)
                else:
                    remaining = WINDOW_SIZE - len(feature_window)
                    print(f"[LIVE] Warming up... {remaining} more sample(s) needed.")

                time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            print("\n\n[LIVE] Stopped by user.")
            print(f"Total Interventions: {len(self.decisions)}")


if __name__ == '__main__':
    # ── CLI argument parsing ──────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="URLLC QoS Controller — simulation or live mode"
    )
    parser.add_argument(
        "--mode",
        choices=["sim", "live"],
        default="sim",
        help="'sim'  → replay training_data.csv (default)\n"
             "'live' → poll get_current_features() in real time"
    )
    parser.add_argument(
        "--model",
        default="./saved/lstm_urllc.pth",
        help="Path to the trained LSTM model (default: ./saved/lstm_urllc.pth)"
    )
    parser.add_argument(
        "--data",
        default="./data/training_data.csv",
        help="Path to training CSV used in sim mode (default: ./data/training_data.csv)"
    )
    args = parser.parse_args()

    # ── Initialise controller ─────────────────────────────────────────────────
    controller = URLLCController(args.model)

    # ── Branch on mode ────────────────────────────────────────────────────────
    if args.mode == "live":
        controller.run_live(args.data)

    else:  # sim (default)
        df = pd.read_csv(args.data)
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

        controller.run_simulation(simulation_data=sim_data)