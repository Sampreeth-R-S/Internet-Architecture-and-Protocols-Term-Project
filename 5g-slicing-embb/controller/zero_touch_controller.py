"""
Zero-Touch eMBB Slice Resource Controller.
- Loads trained LSTM model
- Continuously monitors traffic (or simulates from dataset)
- Predicts future demand
- Proactively resizes slice bandwidth via Open5GS API
- Logs all decisions for evaluation

Need to update with the lstm for other two qos slices as well.

Usage:
    python3 zero_touch_controller.py            # Simulation mode (uses dataset)
    python3 zero_touch_controller.py --live      # Live mode (reads from ogstun)
"""
import numpy as np
import torch
import requests
import time
import json
import csv
import os
import argparse
from datetime import datetime

# Add models directory to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))
from lstm_predictor import TrafficLSTM, WINDOW_SIZE, HORIZON, HIDDEN_SIZE, NUM_LAYERS

# --- Config ---
OPEN5GS_API = "http://127.0.0.1:9999/api"  # Open5GS WebUI API
CHECK_INTERVAL = 5       # seconds
NUM_SUBSCRIBERS = 5      # eMBB slice subscribers
# Slice-level bandwidth allocation (total across all subscribers in the slice)
INITIAL_SLICE_BW_DL = 500    # Mbps (total for slice)
INITIAL_SLICE_BW_UL = 250    # Mbps (total for slice)
MIN_SLICE_BW_DL = 250        # Mbps (minimum total for slice)
MIN_SLICE_BW_UL = 125        # Mbps (minimum total for slice)
MAX_SLICE_BW_DL = 2500       # Mbps (maximum total for slice)
MAX_SLICE_BW_UL = 1250       # Mbps (maximum total for slice)
EXPAND_STEP_RATIO = 0.25
CONTRACT_STEP_RATIO = 0.20
HIGH_THRESHOLD_RATIO = 0.80   # Expand when total throughput > 80% of slice BW
LOW_THRESHOLD_RATIO = 0.40    # Contract when total throughput < 40% of slice BW
SUBSCRIBER_IMSIS = [
    "999700000000001",
    "999700000000002",
    "999700000000003",
    "999700000000004",
    "999700000000005",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class SliceController:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(BASE_DIR, '..', 'models', 'saved', 'lstm_embb.pth')

        # Load model
        self.model = TrafficLSTM(1, HIDDEN_SIZE, NUM_LAYERS, HORIZON)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

        # Load scaler
        saved_dir = os.path.join(BASE_DIR, '..', 'models', 'saved')
        self.scaler_min = np.load(os.path.join(saved_dir, 'scaler_min.npy'))
        self.scaler_scale = np.load(os.path.join(saved_dir, 'scaler_scale.npy'))

        # Traffic history buffer
        self.history = []
        self.current_bw_config = 'normal'  # normal, expanded, contracted
        self.current_slice_bw_dl = INITIAL_SLICE_BW_DL  # Total slice bandwidth (all subscribers)
        self.current_slice_bw_ul = INITIAL_SLICE_BW_UL

        # Decision log
        self.decisions = []

    def scale(self, value):
        return (value - self.scaler_min[0]) * self.scaler_scale[0]

    def inverse_scale(self, value):
        return value / self.scaler_scale[0] + self.scaler_min[0]

    def predict(self, window):
        """Predict next HORIZON steps from window of WINDOW_SIZE values."""
        scaled = np.array([self.scale(v) for v in window])
        x = torch.FloatTensor(scaled).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            pred_scaled = self.model(x).numpy()[0]
        return np.array([self.inverse_scale(v) for v in pred_scaled])

    def decide_action(self, predictions):
        """Proactive decision: resize slice bandwidth based on predicted aggregate throughput.
        Predictions are total throughput across all subscribers in the slice.
        """
        peak_predicted = max(predictions)  # Total peak throughput (all subscribers)
        avg_predicted = np.mean(predictions)  # Total average throughput (all subscribers)
        high_threshold = self.current_slice_bw_dl * HIGH_THRESHOLD_RATIO
        low_threshold = self.current_slice_bw_dl * LOW_THRESHOLD_RATIO

        if peak_predicted > high_threshold and self.current_slice_bw_dl < MAX_SLICE_BW_DL:
            return 'expand', peak_predicted
        elif avg_predicted < low_threshold and self.current_slice_bw_dl > MIN_SLICE_BW_DL:
            return 'contract', avg_predicted
        else:
            return 'hold', avg_predicted

    def apply_action(self, action, trigger_value):
        """Apply per-slice bandwidth reconfiguration via Open5GS API.
        Updates the slice-level QoS settings (not individual subscriber bandwidth).
        """
        timestamp = datetime.now().isoformat()
        prev_dl = self.current_slice_bw_dl
        prev_ul = self.current_slice_bw_ul

        if action == 'expand':
            step_dl = max(1, int(round(prev_dl * EXPAND_STEP_RATIO)))
            step_ul = max(1, int(round(prev_ul * EXPAND_STEP_RATIO)))
            new_slice_bw_dl = min(MAX_SLICE_BW_DL, prev_dl + step_dl)
            new_slice_bw_ul = min(MAX_SLICE_BW_UL, prev_ul + step_ul)
            print(f"[{timestamp}] ⬆ EXPANDING slice BW: Total DL={new_slice_bw_dl}M, UL={new_slice_bw_ul}M "
                f"(from DL={prev_dl}M, UL={prev_ul}M | predicted peak: {trigger_value:.1f} Mbps) "
                f"for {NUM_SUBSCRIBERS} subscribers")
            self.current_bw_config = 'expanded'
            self.current_slice_bw_dl = new_slice_bw_dl
            self.current_slice_bw_ul = new_slice_bw_ul

        elif action == 'contract':
            step_dl = max(1, int(round(prev_dl * CONTRACT_STEP_RATIO)))
            step_ul = max(1, int(round(prev_ul * CONTRACT_STEP_RATIO)))
            new_slice_bw_dl = max(MIN_SLICE_BW_DL, prev_dl - step_dl)
            new_slice_bw_ul = max(MIN_SLICE_BW_UL, prev_ul - step_ul)
            print(f"[{timestamp}] ⬇ CONTRACTING slice BW: Total DL={new_slice_bw_dl}M, UL={new_slice_bw_ul}M "
                f"(from DL={prev_dl}M, UL={prev_ul}M | predicted avg: {trigger_value:.1f} Mbps) "
                f"for {NUM_SUBSCRIBERS} subscribers")
            self.current_bw_config = 'contracted'
            self.current_slice_bw_dl = new_slice_bw_dl
            self.current_slice_bw_ul = new_slice_bw_ul

        else:
            new_slice_bw_dl = prev_dl
            new_slice_bw_ul = prev_ul
            print(f"[{timestamp}] ● HOLD: no change (Total DL={new_slice_bw_dl}M, UL={new_slice_bw_ul}M | "
                  f"predicted: {trigger_value:.1f} Mbps)")
            self.current_bw_config = self.current_bw_config or 'normal'

        # Log decision
        self.decisions.append({
            'timestamp': timestamp,
            'action': action,
            'trigger_value': round(float(trigger_value), 2),
            'config': self.current_bw_config,
            'num_subscribers': NUM_SUBSCRIBERS,
            'slice_bw_dl_mbps': self.current_slice_bw_dl,
            'slice_bw_ul_mbps': self.current_slice_bw_ul,
            'per_subscriber_dl_mbps': round(self.current_slice_bw_dl / NUM_SUBSCRIBERS, 2),
            'per_subscriber_ul_mbps': round(self.current_slice_bw_ul / NUM_SUBSCRIBERS, 2),
            'high_threshold_mbps': round(self.current_slice_bw_dl * HIGH_THRESHOLD_RATIO, 2),
            'low_threshold_mbps': round(self.current_slice_bw_dl * LOW_THRESHOLD_RATIO, 2)
        })

        # In a real deployment, this would call the Open5GS REST API to update slice-level QoS:
        # try:
        #     # Update slice configuration (e.g., via /api/nssai endpoint or similar)
        #     # This sets per-slice AMBR that applies to all {NUM_SUBSCRIBERS} subscribers
        #     resp = requests.put(
        #         f"{OPEN5GS_API}/nssai/slice-config",
        #         json={
        #             "sst": 1,
        #             "sd": "000001",
        #             "ambr": {"downlink": new_slice_bw_dl, "uplink": new_slice_bw_ul}
        #         },
        #         headers={"Content-Type": "application/json"}
        #     )
        #     resp.raise_for_status()
        #     print(f"  API Response: Slice BW updated to DL={new_slice_bw_dl}M, UL={new_slice_bw_ul}M")
        # except Exception as e:
        #     print(f"  API Error: {e}")

    def get_current_throughput(self):
        """Read latest throughput from data collection pipeline."""
        try:
            csv_path = os.path.join(BASE_DIR, '..', 'data', 'embb_traffic_timeseries.csv')
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    return float(rows[-1]['throughput_mbps'])
        except Exception:
            pass
        return 0.0

    def run(self, simulation_data=None):
        """Main control loop. Use simulation_data for offline testing."""
        print("=" * 60)
        print("  Zero-Touch eMBB Slice Controller Started")
        print("=" * 60)
        print(f"  LSTM Prediction: Total aggregate throughput for {NUM_SUBSCRIBERS} subscribers")
        print(f"  Slice Management: Per-slice bandwidth allocation (not per-user)")
        print(f"  Window: {WINDOW_SIZE} steps | Horizon: {HORIZON} steps")
        print(
            f"  Dynamic thresholds (from current slice DL BW): "
            f"HIGH={HIGH_THRESHOLD_RATIO:.0%}, LOW={LOW_THRESHOLD_RATIO:.0%}"
        )
        print(
            f"  Initial Slice BW: DL={self.current_slice_bw_dl} Mbps (per-subscriber: {self.current_slice_bw_dl//NUM_SUBSCRIBERS} Mbps)"
        )
        print(
            f"                    UL={self.current_slice_bw_ul} Mbps (per-subscriber: {self.current_slice_bw_ul//NUM_SUBSCRIBERS} Mbps)"
        )
        print(f"  Subscribers in slice: {NUM_SUBSCRIBERS} ({', '.join(SUBSCRIBER_IMSIS)})")
        print()

        if simulation_data is not None:
            # Offline simulation mode
            print(f"[SIMULATION MODE] Processing {len(simulation_data)} data points...\n")
            for i in range(WINDOW_SIZE, len(simulation_data)):
                window = simulation_data[i - WINDOW_SIZE:i]
                predictions = self.predict(window)
                action, value = self.decide_action(predictions)
                if action != 'hold':  # Only print non-hold actions to reduce noise
                    self.apply_action(action, value)
                else:
                    self.decisions.append({
                        'timestamp': datetime.now().isoformat(),
                        'action': 'hold',
                        'trigger_value': round(float(value), 2),
                        'config': self.current_bw_config,
                        'num_subscribers': NUM_SUBSCRIBERS,
                        'slice_bw_dl_mbps': self.current_slice_bw_dl,
                        'slice_bw_ul_mbps': self.current_slice_bw_ul,
                        'per_subscriber_dl_mbps': round(self.current_slice_bw_dl / NUM_SUBSCRIBERS, 2),
                        'per_subscriber_ul_mbps': round(self.current_slice_bw_ul / NUM_SUBSCRIBERS, 2),
                        'high_threshold_mbps': round(self.current_slice_bw_dl * HIGH_THRESHOLD_RATIO, 2),
                        'low_threshold_mbps': round(self.current_slice_bw_dl * LOW_THRESHOLD_RATIO, 2)
                    })

            # Summary
            actions = [d['action'] for d in self.decisions]
            print(f"\n{'=' * 60}")
            print(f"  Summary: {actions.count('expand')} expansions, "
                  f"{actions.count('contract')} contractions, "
                  f"{actions.count('hold')} holds")
            return self.decisions

        # Live mode
        print("[LIVE MODE] Monitoring traffic...\n")
        while True:
            throughput = self.get_current_throughput()
            self.history.append(throughput)

            if len(self.history) >= WINDOW_SIZE:
                window = self.history[-WINDOW_SIZE:]
                predictions = self.predict(window)
                action, value = self.decide_action(predictions)
                self.apply_action(action, value)

            time.sleep(CHECK_INTERVAL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-Touch eMBB Slice Controller')
    parser.add_argument('--live', action='store_true', help='Run in live mode')
    args = parser.parse_args()

    controller = SliceController()

    if args.live:
        controller.run()
    else:
        # Simulation mode
        import pandas as pd
        data_path = os.path.join(BASE_DIR, '..', 'data', 'embb_traffic_timeseries.csv')
        df = pd.read_csv(data_path)
        data = df['throughput_mbps'].values

        decisions = controller.run(simulation_data=data)

        # Save decisions log
        output_path = os.path.join(BASE_DIR, '..', 'data', 'controller_decisions.json')
        with open(output_path, 'w') as f:
            json.dump(decisions, f, indent=2)
        print(f"\nDecisions saved → {output_path}")
