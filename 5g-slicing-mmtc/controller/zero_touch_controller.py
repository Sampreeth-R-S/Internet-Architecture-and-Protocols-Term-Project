<<<<<<< HEAD
"""
Zero-Touch mMTC Slice Resource Controller.
- Loads trained LSTM model (predicts packet_rate / connection density)
- Monitors mMTC slice traffic (or simulates from dataset)
- Predicts future congestion from device connection surges
- Proactively adjusts slice connection capacity via Open5GS API
- Logs all decisions for evaluation

mMTC vs eMBB controller differences:
  - Predicts packet_rate (connection density) instead of throughput
  - Manages connection capacity and scheduling priority, not raw bandwidth
  - Lower bandwidth thresholds (mMTC is low-throughput, high-device-count)
  - 5QI 79 (delay-tolerant IoT) instead of 5QI 8/9 (high-throughput)

Usage:
    python3 zero_touch_controller.py            # Simulation mode (uses dataset)
    python3 zero_touch_controller.py --live      # Live mode (reads from interface)
"""
import numpy as np
import torch
import time
import json
import csv
import os
import argparse
from datetime import datetime

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))
from lstm_predictor import TrafficLSTM, WINDOW_SIZE, HORIZON, HIDDEN_SIZE, NUM_LAYERS

# --- mMTC Slice Config ---
OPEN5GS_API = "http://127.0.0.1:9999/api"
CHECK_INTERVAL = 5       # seconds

# mMTC slice manages connection capacity (max concurrent devices)
# and low-bandwidth allocation across thousands of IoT devices
NUM_SUBSCRIBERS = 1000   # mMTC slice IoT devices (representative batch)

# Connection capacity thresholds (packet rate = proxy for device density)
INITIAL_CAPACITY_PPS = 500       # packets/sec baseline capacity
MIN_CAPACITY_PPS = 200           # minimum
MAX_CAPACITY_PPS = 2000          # maximum

# Bandwidth is much lower for mMTC (small packets, many devices)
INITIAL_SLICE_BW_DL = 10    # Mbps total for mMTC slice
INITIAL_SLICE_BW_UL = 5     # Mbps total for mMTC slice
MIN_SLICE_BW_DL = 5         # Mbps
MIN_SLICE_BW_UL = 2         # Mbps
MAX_SLICE_BW_DL = 50        # Mbps
MAX_SLICE_BW_UL = 25        # Mbps

EXPAND_STEP_RATIO = 0.30
CONTRACT_STEP_RATIO = 0.20
HIGH_THRESHOLD_RATIO = 0.80   # Expand when packet rate > 80% of capacity
LOW_THRESHOLD_RATIO = 0.30    # Contract when packet rate < 30% of capacity

# mMTC subscriber IMSIs (representative set)
SUBSCRIBER_IMSIS = [f"99970000000{i:04d}" for i in range(11, 21)]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class MmtcSliceController:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(BASE_DIR, '..', 'models', 'saved', 'lstm_mmtc.pth')

        # Load model
        self.model = TrafficLSTM(1, HIDDEN_SIZE, NUM_LAYERS, HORIZON)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

        # Load scaler
        saved_dir = os.path.join(BASE_DIR, '..', 'models', 'saved')
        self.scaler_min = np.load(os.path.join(saved_dir, 'scaler_min.npy'))
        self.scaler_scale = np.load(os.path.join(saved_dir, 'scaler_scale.npy'))

        # State
        self.history = []
        self.current_config = 'normal'
        self.current_capacity_pps = INITIAL_CAPACITY_PPS
        self.current_slice_bw_dl = INITIAL_SLICE_BW_DL
        self.current_slice_bw_ul = INITIAL_SLICE_BW_UL

        # Decision log
        self.decisions = []

    def scale(self, value):
        return (value - self.scaler_min[0]) * self.scaler_scale[0]

    def inverse_scale(self, value):
        return value / self.scaler_scale[0] + self.scaler_min[0]

    def predict(self, window):
        """Predict next HORIZON steps of packet_rate from window."""
        scaled = np.array([self.scale(v) for v in window])
        x = torch.FloatTensor(scaled).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            pred_scaled = self.model(x).numpy()[0]
        return np.array([self.inverse_scale(v) for v in pred_scaled])

    def decide_action(self, predictions):
        """Proactive decision based on predicted packet rate (connection density).
        When too many devices transmit simultaneously, the slice gets congested.
        """
        peak_predicted = max(predictions)
        avg_predicted = np.mean(predictions)
        high_threshold = self.current_capacity_pps * HIGH_THRESHOLD_RATIO
        low_threshold = self.current_capacity_pps * LOW_THRESHOLD_RATIO

        if peak_predicted > high_threshold and self.current_capacity_pps < MAX_CAPACITY_PPS:
            return 'expand', peak_predicted
        elif avg_predicted < low_threshold and self.current_capacity_pps > MIN_CAPACITY_PPS:
            return 'contract', avg_predicted
        else:
            return 'hold', avg_predicted

    def apply_action(self, action, trigger_value):
        """Apply mMTC slice reconfiguration — adjust connection capacity and bandwidth."""
        timestamp = datetime.now().isoformat()
        prev_capacity = self.current_capacity_pps
        prev_dl = self.current_slice_bw_dl
        prev_ul = self.current_slice_bw_ul

        if action == 'expand':
            cap_step = max(1, int(round(prev_capacity * EXPAND_STEP_RATIO)))
            dl_step = max(1, int(round(prev_dl * EXPAND_STEP_RATIO)))
            ul_step = max(1, int(round(prev_ul * EXPAND_STEP_RATIO)))

            new_capacity = min(MAX_CAPACITY_PPS, prev_capacity + cap_step)
            new_dl = min(MAX_SLICE_BW_DL, prev_dl + dl_step)
            new_ul = min(MAX_SLICE_BW_UL, prev_ul + ul_step)

            print(f"[{timestamp}] ⬆ EXPANDING mMTC slice: "
                  f"Capacity={new_capacity} pps (from {prev_capacity}), "
                  f"BW DL={new_dl}M, UL={new_ul}M | "
                  f"predicted peak: {trigger_value:.1f} pps")
            self.current_config = 'expanded'
            self.current_capacity_pps = new_capacity
            self.current_slice_bw_dl = new_dl
            self.current_slice_bw_ul = new_ul

        elif action == 'contract':
            cap_step = max(1, int(round(prev_capacity * CONTRACT_STEP_RATIO)))
            dl_step = max(1, int(round(prev_dl * CONTRACT_STEP_RATIO)))
            ul_step = max(1, int(round(prev_ul * CONTRACT_STEP_RATIO)))

            new_capacity = max(MIN_CAPACITY_PPS, prev_capacity - cap_step)
            new_dl = max(MIN_SLICE_BW_DL, prev_dl - dl_step)
            new_ul = max(MIN_SLICE_BW_UL, prev_ul - ul_step)

            print(f"[{timestamp}] ⬇ CONTRACTING mMTC slice: "
                  f"Capacity={new_capacity} pps (from {prev_capacity}), "
                  f"BW DL={new_dl}M, UL={new_ul}M | "
                  f"predicted avg: {trigger_value:.1f} pps")
            self.current_config = 'contracted'
            self.current_capacity_pps = new_capacity
            self.current_slice_bw_dl = new_dl
            self.current_slice_bw_ul = new_ul

        else:
            new_capacity = prev_capacity
            new_dl = prev_dl
            new_ul = prev_ul
            print(f"[{timestamp}] ● HOLD: no change (Capacity={new_capacity} pps, "
                  f"BW DL={new_dl}M | predicted: {trigger_value:.1f} pps)")
            self.current_config = self.current_config or 'normal'

        # Log decision
        self.decisions.append({
            'timestamp': timestamp,
            'action': action,
            'trigger_value': round(float(trigger_value), 2),
            'config': self.current_config,
            'capacity_pps': self.current_capacity_pps,
            'slice_bw_dl_mbps': self.current_slice_bw_dl,
            'slice_bw_ul_mbps': self.current_slice_bw_ul,
            'high_threshold_pps': round(self.current_capacity_pps * HIGH_THRESHOLD_RATIO, 2),
            'low_threshold_pps': round(self.current_capacity_pps * LOW_THRESHOLD_RATIO, 2),
        })

        # In a real deployment, update Open5GS slice config:
        # try:
        #     resp = requests.put(
        #         f"{OPEN5GS_API}/nssai/slice-config",
        #         json={
        #             "sst": 1,
        #             "sd": "000002",
        #             "ambr": {"downlink": new_dl, "uplink": new_ul},
        #             "max_devices": new_capacity
        #         },
        #         headers={"Content-Type": "application/json"}
        #     )
        #     resp.raise_for_status()
        # except Exception as e:
        #     print(f"  API Error: {e}")

    def get_current_packet_rate(self):
        """Read latest packet rate from data collection pipeline."""
        try:
            csv_path = os.path.join(BASE_DIR, '..', 'data', 'mmtc_traffic_timeseries.csv')
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    return float(rows[-1]['packet_rate'])
        except Exception:
            pass
        return 0.0

    def run(self, simulation_data=None):
        """Main control loop."""
        print("=" * 60)
        print("  Zero-Touch mMTC Slice Controller Started")
        print("=" * 60)
        print(f"  LSTM Prediction: Packet rate (connection density proxy)")
        print(f"  Slice Management: Connection capacity + bandwidth allocation")
        print(f"  Window: {WINDOW_SIZE} steps | Horizon: {HORIZON} steps")
        print(
            f"  Dynamic thresholds: "
            f"HIGH={HIGH_THRESHOLD_RATIO:.0%} of capacity, LOW={LOW_THRESHOLD_RATIO:.0%}"
        )
        print(f"  Initial capacity: {self.current_capacity_pps} pps")
        print(f"  Initial BW: DL={self.current_slice_bw_dl} Mbps, UL={self.current_slice_bw_ul} Mbps")
        print(f"  5QI: 79 (non-GBR, delay-tolerant IoT)")
        print()

        if simulation_data is not None:
            print(f"[SIMULATION MODE] Processing {len(simulation_data)} data points...\n")
            for i in range(WINDOW_SIZE, len(simulation_data)):
                window = simulation_data[i - WINDOW_SIZE:i]
                predictions = self.predict(window)
                action, value = self.decide_action(predictions)
                if action != 'hold':
                    self.apply_action(action, value)
                else:
                    self.decisions.append({
                        'timestamp': datetime.now().isoformat(),
                        'action': 'hold',
                        'trigger_value': round(float(value), 2),
                        'config': self.current_config,
                        'capacity_pps': self.current_capacity_pps,
                        'slice_bw_dl_mbps': self.current_slice_bw_dl,
                        'slice_bw_ul_mbps': self.current_slice_bw_ul,
                        'high_threshold_pps': round(self.current_capacity_pps * HIGH_THRESHOLD_RATIO, 2),
                        'low_threshold_pps': round(self.current_capacity_pps * LOW_THRESHOLD_RATIO, 2),
                    })

            # Summary
            actions = [d['action'] for d in self.decisions]
            print(f"\n{'=' * 60}")
            print(f"  Summary: {actions.count('expand')} expansions, "
                  f"{actions.count('contract')} contractions, "
                  f"{actions.count('hold')} holds")
            return self.decisions

        # Live mode
        print("[LIVE MODE] Monitoring mMTC traffic...\n")
        while True:
            pkt_rate = self.get_current_packet_rate()
            self.history.append(pkt_rate)

            if len(self.history) >= WINDOW_SIZE:
                window = self.history[-WINDOW_SIZE:]
                predictions = self.predict(window)
                action, value = self.decide_action(predictions)
                self.apply_action(action, value)

            time.sleep(CHECK_INTERVAL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-Touch mMTC Slice Controller')
    parser.add_argument('--live', action='store_true', help='Run in live mode')
    args = parser.parse_args()

    controller = MmtcSliceController()

    if args.live:
        controller.run()
    else:
        import pandas as pd
        data_path = os.path.join(BASE_DIR, '..', 'data', 'mmtc_traffic_timeseries.csv')
        df = pd.read_csv(data_path)
        data = df['packet_rate'].values

        decisions = controller.run(simulation_data=data)

        output_path = os.path.join(BASE_DIR, '..', 'data', 'controller_decisions.json')
        with open(output_path, 'w') as f:
            json.dump(decisions, f, indent=2)
        print(f"\nDecisions saved → {output_path}")
=======
"""
Zero-Touch mMTC Slice Resource Controller.
- Loads trained LSTM model (predicts packet_rate / connection density)
- Monitors mMTC slice traffic (or simulates from dataset)
- Predicts future congestion from device connection surges
- Proactively adjusts slice connection capacity via Open5GS API
- Logs all decisions for evaluation

mMTC vs eMBB controller differences:
  - Predicts packet_rate (connection density) instead of throughput
  - Manages connection capacity and scheduling priority, not raw bandwidth
  - Lower bandwidth thresholds (mMTC is low-throughput, high-device-count)
  - 5QI 79 (delay-tolerant IoT) instead of 5QI 8/9 (high-throughput)

Usage:
    python3 zero_touch_controller.py            # Simulation mode (uses dataset)
    python3 zero_touch_controller.py --live      # Live mode (reads from interface)
"""
import numpy as np
import torch
import time
import json
import csv
import os
import argparse
from datetime import datetime

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))
from lstm_predictor import TrafficLSTM, WINDOW_SIZE, HORIZON, HIDDEN_SIZE, NUM_LAYERS

# --- mMTC Slice Config ---
OPEN5GS_API = "http://127.0.0.1:9999/api"
CHECK_INTERVAL = 5       # seconds

# mMTC slice manages connection capacity (max concurrent devices)
# and low-bandwidth allocation across thousands of IoT devices
NUM_SUBSCRIBERS = 1000   # mMTC slice IoT devices (representative batch)

# Connection capacity thresholds (packet rate = proxy for device density)
INITIAL_CAPACITY_PPS = 500       # packets/sec baseline capacity
MIN_CAPACITY_PPS = 200           # minimum
MAX_CAPACITY_PPS = 2000          # maximum

# Bandwidth is much lower for mMTC (small packets, many devices)
INITIAL_SLICE_BW_DL = 10    # Mbps total for mMTC slice
INITIAL_SLICE_BW_UL = 5     # Mbps total for mMTC slice
MIN_SLICE_BW_DL = 5         # Mbps
MIN_SLICE_BW_UL = 2         # Mbps
MAX_SLICE_BW_DL = 50        # Mbps
MAX_SLICE_BW_UL = 25        # Mbps

EXPAND_STEP_RATIO = 0.30
CONTRACT_STEP_RATIO = 0.20
HIGH_THRESHOLD_RATIO = 0.80   # Expand when packet rate > 80% of capacity
LOW_THRESHOLD_RATIO = 0.30    # Contract when packet rate < 30% of capacity

# mMTC subscriber IMSIs (representative set)
SUBSCRIBER_IMSIS = [f"99970000000{i:04d}" for i in range(11, 21)]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class MmtcSliceController:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(BASE_DIR, '..', 'models', 'saved', 'lstm_mmtc.pth')

        # Load model
        self.model = TrafficLSTM(1, HIDDEN_SIZE, NUM_LAYERS, HORIZON)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

        # Load scaler
        saved_dir = os.path.join(BASE_DIR, '..', 'models', 'saved')
        self.scaler_min = np.load(os.path.join(saved_dir, 'scaler_min.npy'))
        self.scaler_scale = np.load(os.path.join(saved_dir, 'scaler_scale.npy'))

        # State
        self.history = []
        self.current_config = 'normal'
        self.current_capacity_pps = INITIAL_CAPACITY_PPS
        self.current_slice_bw_dl = INITIAL_SLICE_BW_DL
        self.current_slice_bw_ul = INITIAL_SLICE_BW_UL

        # Decision log
        self.decisions = []

    def scale(self, value):
        return (value - self.scaler_min[0]) * self.scaler_scale[0]

    def inverse_scale(self, value):
        return value / self.scaler_scale[0] + self.scaler_min[0]

    def predict(self, window):
        """Predict next HORIZON steps of packet_rate from window."""
        scaled = np.array([self.scale(v) for v in window])
        x = torch.FloatTensor(scaled).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            pred_scaled = self.model(x).numpy()[0]
        return np.array([self.inverse_scale(v) for v in pred_scaled])

    def decide_action(self, predictions):
        """Proactive decision based on predicted packet rate (connection density).
        When too many devices transmit simultaneously, the slice gets congested.
        """
        peak_predicted = max(predictions)
        avg_predicted = np.mean(predictions)
        high_threshold = self.current_capacity_pps * HIGH_THRESHOLD_RATIO
        low_threshold = self.current_capacity_pps * LOW_THRESHOLD_RATIO

        if peak_predicted > high_threshold and self.current_capacity_pps < MAX_CAPACITY_PPS:
            return 'expand', peak_predicted
        elif avg_predicted < low_threshold and self.current_capacity_pps > MIN_CAPACITY_PPS:
            return 'contract', avg_predicted
        else:
            return 'hold', avg_predicted

    def apply_action(self, action, trigger_value):
        """Apply mMTC slice reconfiguration — adjust connection capacity and bandwidth."""
        timestamp = datetime.now().isoformat()
        prev_capacity = self.current_capacity_pps
        prev_dl = self.current_slice_bw_dl
        prev_ul = self.current_slice_bw_ul

        if action == 'expand':
            cap_step = max(1, int(round(prev_capacity * EXPAND_STEP_RATIO)))
            dl_step = max(1, int(round(prev_dl * EXPAND_STEP_RATIO)))
            ul_step = max(1, int(round(prev_ul * EXPAND_STEP_RATIO)))

            new_capacity = min(MAX_CAPACITY_PPS, prev_capacity + cap_step)
            new_dl = min(MAX_SLICE_BW_DL, prev_dl + dl_step)
            new_ul = min(MAX_SLICE_BW_UL, prev_ul + ul_step)

            print(f"[{timestamp}] ⬆ EXPANDING mMTC slice: "
                  f"Capacity={new_capacity} pps (from {prev_capacity}), "
                  f"BW DL={new_dl}M, UL={new_ul}M | "
                  f"predicted peak: {trigger_value:.1f} pps")
            self.current_config = 'expanded'
            self.current_capacity_pps = new_capacity
            self.current_slice_bw_dl = new_dl
            self.current_slice_bw_ul = new_ul

        elif action == 'contract':
            cap_step = max(1, int(round(prev_capacity * CONTRACT_STEP_RATIO)))
            dl_step = max(1, int(round(prev_dl * CONTRACT_STEP_RATIO)))
            ul_step = max(1, int(round(prev_ul * CONTRACT_STEP_RATIO)))

            new_capacity = max(MIN_CAPACITY_PPS, prev_capacity - cap_step)
            new_dl = max(MIN_SLICE_BW_DL, prev_dl - dl_step)
            new_ul = max(MIN_SLICE_BW_UL, prev_ul - ul_step)

            print(f"[{timestamp}] ⬇ CONTRACTING mMTC slice: "
                  f"Capacity={new_capacity} pps (from {prev_capacity}), "
                  f"BW DL={new_dl}M, UL={new_ul}M | "
                  f"predicted avg: {trigger_value:.1f} pps")
            self.current_config = 'contracted'
            self.current_capacity_pps = new_capacity
            self.current_slice_bw_dl = new_dl
            self.current_slice_bw_ul = new_ul

        else:
            new_capacity = prev_capacity
            new_dl = prev_dl
            new_ul = prev_ul
            print(f"[{timestamp}] ● HOLD: no change (Capacity={new_capacity} pps, "
                  f"BW DL={new_dl}M | predicted: {trigger_value:.1f} pps)")
            self.current_config = self.current_config or 'normal'

        # Log decision
        self.decisions.append({
            'timestamp': timestamp,
            'action': action,
            'trigger_value': round(float(trigger_value), 2),
            'config': self.current_config,
            'capacity_pps': self.current_capacity_pps,
            'slice_bw_dl_mbps': self.current_slice_bw_dl,
            'slice_bw_ul_mbps': self.current_slice_bw_ul,
            'high_threshold_pps': round(self.current_capacity_pps * HIGH_THRESHOLD_RATIO, 2),
            'low_threshold_pps': round(self.current_capacity_pps * LOW_THRESHOLD_RATIO, 2),
        })

        # In a real deployment, update Open5GS slice config:
        # try:
        #     resp = requests.put(
        #         f"{OPEN5GS_API}/nssai/slice-config",
        #         json={
        #             "sst": 1,
        #             "sd": "000002",
        #             "ambr": {"downlink": new_dl, "uplink": new_ul},
        #             "max_devices": new_capacity
        #         },
        #         headers={"Content-Type": "application/json"}
        #     )
        #     resp.raise_for_status()
        # except Exception as e:
        #     print(f"  API Error: {e}")

    def get_current_packet_rate(self):
        """Read latest packet rate from data collection pipeline."""
        try:
            csv_path = os.path.join(BASE_DIR, '..', 'data', 'mmtc_traffic_timeseries.csv')
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    return float(rows[-1]['packet_rate'])
        except Exception:
            pass
        return 0.0

    def run(self, simulation_data=None):
        """Main control loop."""
        print("=" * 60)
        print("  Zero-Touch mMTC Slice Controller Started")
        print("=" * 60)
        print(f"  LSTM Prediction: Packet rate (connection density proxy)")
        print(f"  Slice Management: Connection capacity + bandwidth allocation")
        print(f"  Window: {WINDOW_SIZE} steps | Horizon: {HORIZON} steps")
        print(
            f"  Dynamic thresholds: "
            f"HIGH={HIGH_THRESHOLD_RATIO:.0%} of capacity, LOW={LOW_THRESHOLD_RATIO:.0%}"
        )
        print(f"  Initial capacity: {self.current_capacity_pps} pps")
        print(f"  Initial BW: DL={self.current_slice_bw_dl} Mbps, UL={self.current_slice_bw_ul} Mbps")
        print(f"  5QI: 79 (non-GBR, delay-tolerant IoT)")
        print()

        if simulation_data is not None:
            print(f"[SIMULATION MODE] Processing {len(simulation_data)} data points...\n")
            for i in range(WINDOW_SIZE, len(simulation_data)):
                window = simulation_data[i - WINDOW_SIZE:i]
                predictions = self.predict(window)
                action, value = self.decide_action(predictions)
                if action != 'hold':
                    self.apply_action(action, value)
                else:
                    self.decisions.append({
                        'timestamp': datetime.now().isoformat(),
                        'action': 'hold',
                        'trigger_value': round(float(value), 2),
                        'config': self.current_config,
                        'capacity_pps': self.current_capacity_pps,
                        'slice_bw_dl_mbps': self.current_slice_bw_dl,
                        'slice_bw_ul_mbps': self.current_slice_bw_ul,
                        'high_threshold_pps': round(self.current_capacity_pps * HIGH_THRESHOLD_RATIO, 2),
                        'low_threshold_pps': round(self.current_capacity_pps * LOW_THRESHOLD_RATIO, 2),
                    })

            # Summary
            actions = [d['action'] for d in self.decisions]
            print(f"\n{'=' * 60}")
            print(f"  Summary: {actions.count('expand')} expansions, "
                  f"{actions.count('contract')} contractions, "
                  f"{actions.count('hold')} holds")
            return self.decisions

        # Live mode
        print("[LIVE MODE] Monitoring mMTC traffic...\n")
        while True:
            pkt_rate = self.get_current_packet_rate()
            self.history.append(pkt_rate)

            if len(self.history) >= WINDOW_SIZE:
                window = self.history[-WINDOW_SIZE:]
                predictions = self.predict(window)
                action, value = self.decide_action(predictions)
                self.apply_action(action, value)

            time.sleep(CHECK_INTERVAL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-Touch mMTC Slice Controller')
    parser.add_argument('--live', action='store_true', help='Run in live mode')
    args = parser.parse_args()

    controller = MmtcSliceController()

    if args.live:
        controller.run()
    else:
        import pandas as pd
        data_path = os.path.join(BASE_DIR, '..', 'data', 'mmtc_traffic_timeseries.csv')
        df = pd.read_csv(data_path)
        data = df['packet_rate'].values

        decisions = controller.run(simulation_data=data)

        output_path = os.path.join(BASE_DIR, '..', 'data', 'controller_decisions.json')
        with open(output_path, 'w') as f:
            json.dump(decisions, f, indent=2)
        print(f"\nDecisions saved → {output_path}")
>>>>>>> d7e5ce9f407c4039a9955cd4e8aa4162b84a16e1
