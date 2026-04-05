"""
Unified Zero-Touch 5G Slice Controller
=======================================
Manages three 5G network slices concurrently using separate OS processes:

  • eMBB  (SST=1) — bandwidth expand/contract + AMBR update via Open5GS API
  • mMTC  (SST=3) — connection-capacity expand/contract + AMBR update via Open5GS API
  • URLLC (SST=2) — latency-aware 5QI/ARP elevation via Open5GS API

All three controllers:
  - Authenticate to Open5GS WebUI using CSRF-token + JWT flow
  - Fetch ALL subscribers, filter by SST value (no hardcoded IMSI lists)
  - Push configuration updates via REST PUT per subscriber

Each slice runs in its own multiprocessing.Process.
All three lstm_predictor.py files share the same file name, so they are loaded
via importlib under uniquely aliased module names:
    lstm_predictor_urllc  (3-feature: raw+rolling_mean+rolling_std)
    lstm_predictor_mmtc   (1-feature: packet_rate)
    lstm_predictor_embb   (1-feature: throughput_mbps)

Usage
-----
    python3 unified_controller.py               # Simulation mode (default)
    python3 unified_controller.py --mode live   # Live polling mode

    python3 unified_controller.py \\
        --urllc-data /path/to/urllc.csv \\
        --mmtc-data  /path/to/mmtc.csv  \\
        --embb-data  /path/to/embb.csv  \\
        --urllc-model /path/to/lstm_urllc.pth \\
        --mmtc-model  /path/to/lstm_mmtc.pth  \\
        --embb-model  /path/to/lstm_embb.pth
"""

from __future__ import annotations

import importlib.util
import sys
import os
import re
import csv
import json
import time
import argparse
import multiprocessing
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import requests
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR  = os.path.dirname(os.path.abspath(__file__))
URLLC_DIR = os.path.join(ROOT_DIR, '5g-slicing-urllc')
MMTC_DIR  = os.path.join(ROOT_DIR, '5g-slicing-mmtc')
EMBB_DIR  = os.path.join(ROOT_DIR, '5g-slicing-embb')


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic LSTM-module loader  (avoids naming collision between same-named files)
# ─────────────────────────────────────────────────────────────────────────────
def _load_lstm_module(abs_path: str, alias: str):
    """
    Load a Python source file from *abs_path* into sys.modules under *alias*.
    Returns the loaded module object.
    """
    spec = importlib.util.spec_from_file_location(alias, abs_path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# Open5GS WebUI Config  (shared by all three controllers)
# ─────────────────────────────────────────────────────────────────────────────
WEBUI_URL   = "http://127.0.0.1:9999"
WEBUI_USER  = "admin"
WEBUI_PASS  = "1423"
OPEN5GS_API = f"{WEBUI_URL}/api"

CHECK_INTERVAL = 1   # seconds between live-mode polls


# ─────────────────────────────────────────────────────────────────────────────
# Shared Open5GS auth mixin
# ─────────────────────────────────────────────────────────────────────────────
class Open5GSAuthMixin:
    """
    Provides CSRF-token + JWT authentication and a common subscriber-update
    helper.  Sub-classes must define self.SLICE_TAG (string used in log lines)
    and call self._init_session() from their __init__.
    """

    SLICE_TAG = "BASE"   # overridden in each subclass

    def _init_session(self):
        """Initialise the requests session and authenticate."""
        self.auth_token = None
        self.csrf_token = None
        self.session    = requests.Session()
        self.authenticate()

    def authenticate(self):
        """Log in to Open5GS WebUI using CSRF-token + JWT flow."""
        print(f"[{self.SLICE_TAG}] 🔐 Authenticating with Open5GS WebUI...")
        try:
            # Step 1: fetch homepage → CSRF token + connect.sid cookie
            home  = self.session.get(WEBUI_URL)
            match = re.search(r'__NEXT_DATA__\s*=\s*({.*?})\s*\n', home.text, re.DOTALL)
            if not match:
                raise Exception("Could not find __NEXT_DATA__ in homepage HTML.")

            next_data       = json.loads(match.group(1))
            self.csrf_token = next_data["props"]["initialProps"]["session"]["csrfToken"]
            print(f"    [+] CSRF token: {self.csrf_token[:30]}...")

            # Step 2: POST login — session carries the connect.sid cookie
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
            sess_resp       = self.session.get(
                f"{WEBUI_URL}/api/auth/session",
                headers={"X-CSRF-Token": self.csrf_token},
            )
            sess_data       = sess_resp.json()
            self.csrf_token = sess_data.get("csrfToken", self.csrf_token)
            self.auth_token = sess_data.get("authToken")

            if not self.auth_token:
                raise Exception("No authToken returned — check credentials.")

            print(f"    [+] JWT authToken: {self.auth_token[:40]}...")
            print(f"    [+] [{self.SLICE_TAG}] Authentication Successful!\n")

        except Exception as e:
            print(f"    [!] [{self.SLICE_TAG}] Authentication Failed: {e}")
            self.auth_token = None

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type":  "application/json",
            "X-CSRF-Token":  self.csrf_token,
        }

    def _fetch_all_subscribers(self) -> list:
        """Fetch all subscribers from Open5GS.  Returns list or []."""
        if not self.auth_token:
            print(f"  [{self.SLICE_TAG}][!] Not authenticated — skipping API call.")
            return []
        try:
            resp = self.session.get(
                f"{OPEN5GS_API}/db/Subscriber",
                headers=self._headers(),
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"  [{self.SLICE_TAG}][!] Failed to fetch subscribers: {e}")
            return []

    def _put_subscriber(self, imsi: str, sub: dict) -> bool:
        """PUT an updated subscriber document.  Returns True on success."""
        try:
            r = self.session.put(
                f"{OPEN5GS_API}/db/Subscriber/{imsi}",
                json=sub,
                headers=self._headers(),
            )
            r.raise_for_status()
            return True
        except Exception as e:
            print(f"  [{self.SLICE_TAG}][!] PUT {imsi} failed: {e}")
            return False


# ═════════════════════════════════════════════════════════════════════════════
# URLLC Controller  (SST=2)
# ═════════════════════════════════════════════════════════════════════════════
URLLC_SST_TARGET             = 2
URLLC_HIGH_LATENCY_THRESHOLD = 2.0   # ms
URLLC_LOW_LATENCY_THRESHOLD  = 1.0   # ms

URLLC_QOS_NORMAL = {
    "5qi":          85,
    "arp_priority": 5,
    "pre_emp_cap":  2,
    "pre_emp_vuln": 1,
    "state_name":   "NORMAL",
}
URLLC_QOS_ELEVATED = {
    "5qi":          82,
    "arp_priority": 1,
    "pre_emp_cap":  1,
    "pre_emp_vuln": 2,
    "state_name":   "ELEVATED (CRITICAL)",
}


class URLLCController(Open5GSAuthMixin):
    """
    Zero-touch controller for the URLLC slice (SST=2).

    Predicts latency via a 3-feature LSTM, then elevates or relaxes
    5QI + ARP priority for every subscriber whose slice SST == 2.
    """

    SLICE_TAG = "URLLC"

    def __init__(self, model_path: Optional[str] = None):
        # ── Load the URLLC LSTM module under a unique alias ────────────────
        lstm_path   = os.path.join(URLLC_DIR, 'models', 'lstm_predictor.py')
        self._lstm  = _load_lstm_module(lstm_path, 'lstm_predictor_urllc')

        self._build_features = self._lstm.build_features
        self.WINDOW_SIZE     = self._lstm.WINDOW_SIZE
        self.HORIZON         = self._lstm.HORIZON
        self.HIDDEN_SIZE     = self._lstm.HIDDEN_SIZE
        self.NUM_LAYERS      = self._lstm.NUM_LAYERS
        self.N_FEATURES      = self._lstm.N_FEATURES

        if model_path is None:
            model_path = os.path.join(URLLC_DIR, 'saved', 'lstm_urllc.pth')

        self.model = self._lstm.TrafficLSTM(
            self.N_FEATURES, self.HIDDEN_SIZE, self.NUM_LAYERS, self.HORIZON
        )
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

        saved_dir = os.path.join(URLLC_DIR, 'saved')
        self.feat_scaler_min   = np.load(os.path.join(saved_dir, 'feat_scaler_min.npy'))
        self.feat_scaler_scale = np.load(os.path.join(saved_dir, 'feat_scaler_scale.npy'))
        self.tgt_scaler_min    = np.load(os.path.join(saved_dir, 'tgt_scaler_min.npy'))
        self.tgt_scaler_scale  = np.load(os.path.join(saved_dir, 'tgt_scaler_scale.npy'))

        self.latency_buffer = []
        self.current_state  = URLLC_QOS_NORMAL['state_name']
        self.decisions: list = []

        self._init_session()

    # ── Scalers ────────────────────────────────────────────────────────────

    def _scale_features(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.feat_scaler_min) * self.feat_scaler_scale

    def _inverse_scale_target(self, arr: np.ndarray) -> np.ndarray:
        return arr / self.tgt_scaler_scale[0] + self.tgt_scaler_min[0]

    # ── Prediction ─────────────────────────────────────────────────────────

    def predict(self, raw_latency_window: np.ndarray) -> np.ndarray:
        features        = self._build_features(raw_latency_window)
        features_scaled = self._scale_features(features)
        x = torch.FloatTensor(features_scaled).unsqueeze(0)
        with torch.no_grad():
            pred_scaled = self.model(x).numpy()[0]
        return self._inverse_scale_target(pred_scaled)

    # ── Decision ───────────────────────────────────────────────────────────

    def decide_action(self, predicted_latencies: np.ndarray):
        if len(predicted_latencies) == 0:
            return 'hold', 0.0
        mean_latency = float(np.mean(predicted_latencies))
        if (mean_latency > URLLC_HIGH_LATENCY_THRESHOLD
                and self.current_state != URLLC_QOS_ELEVATED['state_name']):
            return 'elevate', mean_latency
        if (mean_latency < URLLC_LOW_LATENCY_THRESHOLD
                and self.current_state != URLLC_QOS_NORMAL['state_name']):
            return 'relax', mean_latency
        return 'hold', mean_latency

    # ── QoS application ────────────────────────────────────────────────────

    def apply_qos(self, action: str, trigger_value: float):
        timestamp = datetime.now().isoformat()

        if action == 'elevate':
            target_qos = URLLC_QOS_ELEVATED
            print(
                f"[URLLC][{timestamp}] ⚠️  ALERT: Latency {trigger_value:.2f} ms > HIGH "
                f"({URLLC_HIGH_LATENCY_THRESHOLD} ms)\n"
                f"    ↳ ELEVATING QoS → 5QI: {target_qos['5qi']}, "
                f"ARP: {target_qos['arp_priority']}"
            )
        elif action == 'relax':
            target_qos = URLLC_QOS_NORMAL
            print(
                f"[URLLC][{timestamp}] ✅ STABLE: Latency {trigger_value:.2f} ms < LOW "
                f"({URLLC_LOW_LATENCY_THRESHOLD} ms)\n"
                f"    ↳ RELAXING QoS → 5QI: {target_qos['5qi']}, "
                f"ARP: {target_qos['arp_priority']}"
            )
        else:
            print(
                f"[URLLC][{timestamp}] ⏸️  HOLD: Latency {trigger_value:.2f} ms within band "
                f"({URLLC_LOW_LATENCY_THRESHOLD} – {URLLC_HIGH_LATENCY_THRESHOLD} ms)"
            )
            self.decisions.append({
                'slice':           'URLLC',
                'timestamp':       timestamp,
                'action':          'hold',
                'trigger_latency': round(trigger_value, 2),
                'state':           self.current_state,
                'applied_5qi':     None,
                'applied_arp':     None,
            })
            return

        self.current_state = target_qos['state_name']
        self.decisions.append({
            'slice':           'URLLC',
            'timestamp':       timestamp,
            'action':          action,
            'trigger_latency': round(trigger_value, 2),
            'state':           self.current_state,
            'applied_5qi':     target_qos['5qi'],
            'applied_arp':     target_qos['arp_priority'],
        })
        self.update_open5gs_subscribers(target_qos)

    # ── Open5GS subscriber update (SST-based) ─────────────────────────────

    def update_open5gs_subscribers(self, target_qos: dict):
        """
        Fetch all subscribers, filter by SST=URLLC_SST_TARGET,
        then push 5QI + ARP updates via REST PUT.
        """
        subscribers = self._fetch_all_subscribers()
        updated = 0
        for sub in subscribers:
            imsi     = sub.get('imsi')
            modified = False
            for s_nssai in sub.get('slice', []):
                if s_nssai.get('sst') == URLLC_SST_TARGET:
                    for sess in s_nssai.get('session', []):
                        sess['qos']['index']                            = target_qos['5qi']
                        sess['qos']['arp']['priority_level']            = target_qos['arp_priority']
                        sess['qos']['arp']['pre_emption_capability']    = target_qos['pre_emp_cap']
                        sess['qos']['arp']['pre_emption_vulnerability'] = target_qos['pre_emp_vuln']
                        modified = True
            if modified and self._put_subscriber(imsi, sub):
                updated += 1
                print(f"    [URLLC][OK] Updated IMSI {imsi} (SST={URLLC_SST_TARGET})")

        if updated:
            print(f"    [URLLC] Pushed QoS to {updated} subscriber(s) with SST={URLLC_SST_TARGET}.")
        else:
            print(f"    [URLLC][!] No subscribers found with SST={URLLC_SST_TARGET}.")

    # ── Live helper ────────────────────────────────────────────────────────

    def get_current_latency(self, csv_path: str) -> float:
        try:
            with open(csv_path, 'r') as f:
                rows = list(csv.DictReader(f))
                if rows and 'latency_ms' in rows[-1]:
                    return float(rows[-1]['latency_ms'])
        except Exception:
            pass
        return 0.0

    # ── Run modes ──────────────────────────────────────────────────────────

    def run_simulation(self, simulation_data: np.ndarray) -> list:
        print("=" * 60)
        print("  URLLC Controller  —  SIMULATION MODE")
        print("=" * 60)
        print(f"  LSTM window={self.WINDOW_SIZE} | horizon={self.HORIZON} | "
              f"features={self.N_FEATURES}")
        print(f"  HIGH: {URLLC_HIGH_LATENCY_THRESHOLD} ms | LOW: {URLLC_LOW_LATENCY_THRESHOLD} ms")
        print(f"  SST filter: {URLLC_SST_TARGET}")
        T = len(simulation_data)
        print(f"  Data points: {T}\n")

        LOG_EVERY = 5000
        for i in range(self.WINDOW_SIZE, T):
            window      = simulation_data[i - self.WINDOW_SIZE: i]
            predictions = self.predict(window)
            action, val = self.decide_action(predictions)
            self.apply_qos(action, val)
            if (i - self.WINDOW_SIZE + 1) % LOG_EVERY == 0:
                pct = 100 * (i - self.WINDOW_SIZE + 1) / (T - self.WINDOW_SIZE)
                print(f"[URLLC] Progress: {i - self.WINDOW_SIZE + 1}/{T - self.WINDOW_SIZE} "
                      f"steps ({pct:.1f}%) | decisions={len(self.decisions)}")

        actions = [d['action'] for d in self.decisions]
        print(f"\n{'=' * 60}")
        print(f"  [URLLC] {actions.count('elevate')} elevate | "
              f"{actions.count('relax')} relax | {actions.count('hold')} hold")
        return self.decisions

    def run_live(self, csv_path: str):
        print("=" * 60)
        print("  URLLC Controller  —  LIVE MODE")
        print("=" * 60)
        print(f"  Polling '{csv_path}' every {CHECK_INTERVAL}s.")
        print(f"  Need {self.WINDOW_SIZE} warm-up samples. Press Ctrl+C to stop.\n")
        try:
            while True:
                latency = self.get_current_latency(csv_path)
                self.latency_buffer.append(latency)
                if len(self.latency_buffer) > self.WINDOW_SIZE:
                    self.latency_buffer.pop(0)
                if len(self.latency_buffer) == self.WINDOW_SIZE:
                    window      = np.array(self.latency_buffer, dtype=np.float32)
                    predictions = self.predict(window)
                    print(f"[URLLC][LIVE] Preds (next {self.HORIZON}): {np.round(predictions, 3)}")
                    action, val = self.decide_action(predictions)
                    self.apply_qos(action, val)
                else:
                    print(f"[URLLC][LIVE] Warming up… "
                          f"{self.WINDOW_SIZE - len(self.latency_buffer)} more.")
                time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            print(f"\n[URLLC] Stopped. Total decisions: {len(self.decisions)}")


# ═════════════════════════════════════════════════════════════════════════════
# mMTC Controller  (SST=3)
# ═════════════════════════════════════════════════════════════════════════════
MMTC_SST_TARGET           = 3
MMTC_NUM_SUBSCRIBERS      = 1000   # informational only
MMTC_INITIAL_CAPACITY     = 500    # pps baseline
MMTC_MIN_CAPACITY         = 200
MMTC_MAX_CAPACITY         = 2000
MMTC_INITIAL_BW_DL        = 10    # Mbps
MMTC_INITIAL_BW_UL        = 5
MMTC_MIN_BW_DL            = 5
MMTC_MIN_BW_UL            = 2
MMTC_MAX_BW_DL            = 50
MMTC_MAX_BW_UL            = 25
MMTC_EXPAND_RATIO         = 0.30
MMTC_CONTRACT_RATIO       = 0.20
MMTC_HIGH_THRESHOLD_RATIO = 0.8
MMTC_LOW_THRESHOLD_RATIO  = 0.4


class MMTCController(Open5GSAuthMixin):
    """
    Zero-touch controller for the mMTC slice (SST=3).

    Predicts packet_rate via a 1-feature LSTM, then expands or contracts
    connection capacity + slice AMBR for every subscriber with SST == 3.
    """

    SLICE_TAG = "mMTC"

    def __init__(self, model_path: Optional[str] = None):
        # ── Load mMTC LSTM under unique alias ──────────────────────────────
        lstm_path  = os.path.join(MMTC_DIR, 'models', 'lstm_predictor.py')
        self._lstm = _load_lstm_module(lstm_path, 'lstm_predictor_mmtc')

        self.WINDOW_SIZE = self._lstm.WINDOW_SIZE
        self.HORIZON     = self._lstm.HORIZON
        self.HIDDEN_SIZE = self._lstm.HIDDEN_SIZE
        self.NUM_LAYERS  = self._lstm.NUM_LAYERS

        if model_path is None:
            model_path = os.path.join(MMTC_DIR, 'models', 'saved', 'lstm_mmtc.pth')

        self.model = self._lstm.TrafficLSTM(
            1, self.HIDDEN_SIZE, self.NUM_LAYERS, self.HORIZON
        )
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

        saved_dir = os.path.join(MMTC_DIR, 'models', 'saved')
        self.scaler_min   = np.load(os.path.join(saved_dir, 'scaler_min.npy'))
        self.scaler_scale = np.load(os.path.join(saved_dir, 'scaler_scale.npy'))

        self.history              = []
        self.current_config       = 'normal'
        self.current_capacity_pps = MMTC_INITIAL_CAPACITY
        self.current_slice_bw_dl  = MMTC_INITIAL_BW_DL
        self.current_slice_bw_ul  = MMTC_INITIAL_BW_UL
        self.decisions: list      = []

        self._init_session()

    # ── Scalers ────────────────────────────────────────────────────────────

    def _scale(self, v: float) -> float:
        return (v - self.scaler_min[0]) * self.scaler_scale[0]

    def _inverse_scale(self, v: float) -> float:
        return v / self.scaler_scale[0] + self.scaler_min[0]

    # ── Prediction ─────────────────────────────────────────────────────────

    def predict(self, window) -> np.ndarray:
        scaled = np.array([self._scale(v) for v in window])
        x = torch.FloatTensor(scaled).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            pred_scaled = self.model(x).numpy()[0]
        return np.array([self._inverse_scale(v) for v in pred_scaled])

    # ── Decision ───────────────────────────────────────────────────────────

    def decide_action(self, predictions):
        peak  = max(predictions)
        avg   = float(np.mean(predictions))
        hi_th = self.current_capacity_pps * MMTC_HIGH_THRESHOLD_RATIO
        lo_th = self.current_capacity_pps * MMTC_LOW_THRESHOLD_RATIO
        if peak > hi_th and self.current_capacity_pps < MMTC_MAX_CAPACITY:
            return 'expand', peak
        elif avg < lo_th and self.current_capacity_pps > MMTC_MIN_CAPACITY:
            return 'contract', avg
        return 'hold', avg

    # ── Action application ─────────────────────────────────────────────────

    def apply_action(self, action: str, trigger_value: float):
        timestamp = datetime.now().isoformat()
        prev_cap  = self.current_capacity_pps
        prev_dl   = self.current_slice_bw_dl
        prev_ul   = self.current_slice_bw_ul

        if action == 'expand':
            new_cap = min(MMTC_MAX_CAPACITY,
                          prev_cap + max(1, int(round(prev_cap * MMTC_EXPAND_RATIO))))
            new_dl  = min(MMTC_MAX_BW_DL,
                          prev_dl + max(1, int(round(prev_dl * MMTC_EXPAND_RATIO))))
            new_ul  = min(MMTC_MAX_BW_UL,
                          prev_ul + max(1, int(round(prev_ul * MMTC_EXPAND_RATIO))))
            print(f"[mMTC][{timestamp}] ⬆ EXPANDING: Capacity={new_cap} pps "
                  f"(from {prev_cap}), BW DL={new_dl}M UL={new_ul}M | "
                  f"peak pred={trigger_value:.1f} pps")
            self.current_config       = 'expanded'
            self.current_capacity_pps = new_cap
            self.current_slice_bw_dl  = new_dl
            self.current_slice_bw_ul  = new_ul

        elif action == 'contract':
            new_cap = max(MMTC_MIN_CAPACITY,
                          prev_cap - max(1, int(round(prev_cap * MMTC_CONTRACT_RATIO))))
            new_dl  = max(MMTC_MIN_BW_DL,
                          prev_dl - max(1, int(round(prev_dl * MMTC_CONTRACT_RATIO))))
            new_ul  = max(MMTC_MIN_BW_UL,
                          prev_ul - max(1, int(round(prev_ul * MMTC_CONTRACT_RATIO))))
            print(f"[mMTC][{timestamp}] ⬇ CONTRACTING: Capacity={new_cap} pps "
                  f"(from {prev_cap}), BW DL={new_dl}M UL={new_ul}M | "
                  f"avg pred={trigger_value:.1f} pps")
            self.current_config       = 'contracted'
            self.current_capacity_pps = new_cap
            self.current_slice_bw_dl  = new_dl
            self.current_slice_bw_ul  = new_ul

        else:
            new_cap = prev_cap
            new_dl  = prev_dl
            new_ul  = prev_ul
            print(f"[mMTC][{timestamp}] ● HOLD: Capacity={new_cap} pps, "
                  f"BW DL={new_dl}M | pred={trigger_value:.1f} pps")
            self.current_config = self.current_config or 'normal'

        self.decisions.append({
            'slice':              'mMTC',
            'timestamp':          timestamp,
            'action':             action,
            'trigger_value':      round(float(trigger_value), 2),
            'config':             self.current_config,
            'capacity_pps':       self.current_capacity_pps,
            'slice_bw_dl_mbps':   self.current_slice_bw_dl,
            'slice_bw_ul_mbps':   self.current_slice_bw_ul,
            'high_threshold_pps': round(self.current_capacity_pps * MMTC_HIGH_THRESHOLD_RATIO, 2),
            'low_threshold_pps':  round(self.current_capacity_pps * MMTC_LOW_THRESHOLD_RATIO, 2),
        })

        # Push AMBR update to all SST=3 subscribers (non-hold actions only)
        if action != 'hold':
            self.update_open5gs_subscribers(new_dl, new_ul)

    # ── Open5GS subscriber update (SST-based) ─────────────────────────────

    def update_open5gs_subscribers(self, dl_mbps: int, ul_mbps: int):
        """
        Fetch all subscribers, filter by SST=MMTC_SST_TARGET,
        then update the session AMBR to reflect the new bandwidth allocation.
        AMBR unit: 2 = Mbps (Open5GS encoding).
        """
        subscribers = self._fetch_all_subscribers()
        updated = 0
        for sub in subscribers:
            imsi     = sub.get('imsi')
            modified = False
            for s_nssai in sub.get('slice', []):
                if s_nssai.get('sst') == MMTC_SST_TARGET:
                    for sess in s_nssai.get('session', []):
                        # Update session-level AMBR
                        sess.setdefault('ambr', {})
                        sess['ambr']['downlink'] = {'value': dl_mbps, 'unit': 2}
                        sess['ambr']['uplink']   = {'value': ul_mbps, 'unit': 2}
                        modified = True
            if modified and self._put_subscriber(imsi, sub):
                updated += 1
                print(f"    [mMTC][OK] Updated IMSI {imsi} "
                      f"(SST={MMTC_SST_TARGET}) → DL={dl_mbps}M UL={ul_mbps}M")

        if updated:
            print(f"    [mMTC] Pushed AMBR to {updated} subscriber(s) "
                  f"with SST={MMTC_SST_TARGET}.")
        else:
            print(f"    [mMTC][!] No subscribers found with SST={MMTC_SST_TARGET}.")

    # ── Live helper ────────────────────────────────────────────────────────

    def get_current_packet_rate(self, csv_path: str) -> float:
        try:
            with open(csv_path, 'r') as f:
                rows = list(csv.DictReader(f))
                if rows and 'packet_rate' in rows[-1]:
                    return float(rows[-1]['packet_rate'])
        except Exception:
            pass
        return 0.0

    # ── Run modes ──────────────────────────────────────────────────────────

    def run_simulation(self, simulation_data: np.ndarray) -> list:
        print("=" * 60)
        print("  mMTC Controller  —  SIMULATION MODE")
        print("=" * 60)
        print(f"  LSTM window={self.WINDOW_SIZE} | horizon={self.HORIZON}")
        print(f"  Initial: {self.current_capacity_pps} pps, "
              f"DL={self.current_slice_bw_dl}M UL={self.current_slice_bw_ul}M")
        print(f"  SST filter: {MMTC_SST_TARGET}")
        T = len(simulation_data)
        print(f"  Data points: {T}\n")

        LOG_EVERY = 5000
        for i in range(self.WINDOW_SIZE, T):
            window      = simulation_data[i - self.WINDOW_SIZE: i]
            predictions = self.predict(window)
            action, val = self.decide_action(predictions)
            self.apply_action(action, val)
            if (i - self.WINDOW_SIZE + 1) % LOG_EVERY == 0:
                pct = 100 * (i - self.WINDOW_SIZE + 1) / (T - self.WINDOW_SIZE)
                print(f"[mMTC] Progress: {i - self.WINDOW_SIZE + 1}/{T - self.WINDOW_SIZE} "
                      f"steps ({pct:.1f}%) | decisions={len(self.decisions)}")

        actions = [d['action'] for d in self.decisions]
        print(f"\n{'=' * 60}")
        print(f"  [mMTC] {actions.count('expand')} expand | "
              f"{actions.count('contract')} contract | {actions.count('hold')} hold")
        return self.decisions

    def run_live(self, csv_path: str):
        print("=" * 60)
        print("  mMTC Controller  —  LIVE MODE")
        print("=" * 60)
        print(f"  Polling '{csv_path}' every {CHECK_INTERVAL}s. Press Ctrl+C to stop.\n")
        try:
            while True:
                pkt_rate = self.get_current_packet_rate(csv_path)
                self.history.append(pkt_rate)
                if len(self.history) >= self.WINDOW_SIZE:
                    window      = self.history[-self.WINDOW_SIZE:]
                    predictions = self.predict(window)
                    action, val = self.decide_action(predictions)
                    self.apply_action(action, val)
                else:
                    print(f"[mMTC][LIVE] Warming up… "
                          f"{self.WINDOW_SIZE - len(self.history)} more.")
                time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            print(f"\n[mMTC] Stopped. Total decisions: {len(self.decisions)}")


# ═════════════════════════════════════════════════════════════════════════════
# eMBB Controller  (SST=1)
# ═════════════════════════════════════════════════════════════════════════════
EMBB_SST_TARGET           = 1
EMBB_NUM_SUBSCRIBERS      = 5   # informational only
EMBB_INITIAL_BW_DL        = 500    # Mbps
EMBB_INITIAL_BW_UL        = 250
EMBB_MIN_BW_DL            = 250
EMBB_MIN_BW_UL            = 125
EMBB_MAX_BW_DL            = 2500
EMBB_MAX_BW_UL            = 1250
EMBB_EXPAND_RATIO         = 0.25
EMBB_CONTRACT_RATIO       = 0.20
EMBB_HIGH_THRESHOLD_RATIO = 0.8
EMBB_LOW_THRESHOLD_RATIO  = 0.4


class EMBBController(Open5GSAuthMixin):
    """
    Zero-touch controller for the eMBB slice (SST=1).

    Predicts aggregate throughput via a 1-feature LSTM, then expands or contracts
    slice AMBR for every subscriber with SST == 1.
    """

    SLICE_TAG = "eMBB"

    def __init__(self, model_path: Optional[str] = None):
        # ── Load eMBB LSTM under unique alias ──────────────────────────────
        lstm_path  = os.path.join(EMBB_DIR, 'models', 'lstm_predictor.py')
        self._lstm = _load_lstm_module(lstm_path, 'lstm_predictor_embb')

        self.WINDOW_SIZE = self._lstm.WINDOW_SIZE
        self.HORIZON     = self._lstm.HORIZON
        self.HIDDEN_SIZE = self._lstm.HIDDEN_SIZE
        self.NUM_LAYERS  = self._lstm.NUM_LAYERS

        if model_path is None:
            model_path = os.path.join(EMBB_DIR, 'models', 'saved', 'lstm_embb.pth')

        self.model = self._lstm.TrafficLSTM(
            1, self.HIDDEN_SIZE, self.NUM_LAYERS, self.HORIZON
        )
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

        saved_dir = os.path.join(EMBB_DIR, 'models', 'saved')
        self.scaler_min   = np.load(os.path.join(saved_dir, 'scaler_min.npy'))
        self.scaler_scale = np.load(os.path.join(saved_dir, 'scaler_scale.npy'))

        self.history             = []
        self.current_bw_config   = 'normal'
        self.current_slice_bw_dl = EMBB_INITIAL_BW_DL
        self.current_slice_bw_ul = EMBB_INITIAL_BW_UL
        self.decisions: list     = []

        self._init_session()

    # ── Scalers ────────────────────────────────────────────────────────────

    def _scale(self, v: float) -> float:
        return (v - self.scaler_min[0]) * self.scaler_scale[0]

    def _inverse_scale(self, v: float) -> float:
        return v / self.scaler_scale[0] + self.scaler_min[0]

    # ── Prediction ─────────────────────────────────────────────────────────

    def predict(self, window) -> np.ndarray:
        scaled = np.array([self._scale(v) for v in window])
        x = torch.FloatTensor(scaled).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            pred_scaled = self.model(x).numpy()[0]
        return np.array([self._inverse_scale(v) for v in pred_scaled])

    # ── Decision ───────────────────────────────────────────────────────────

    def decide_action(self, predictions):
        peak  = max(predictions)
        avg   = float(np.mean(predictions))
        hi_th = self.current_slice_bw_dl * EMBB_HIGH_THRESHOLD_RATIO
        lo_th = self.current_slice_bw_dl * EMBB_LOW_THRESHOLD_RATIO
        if peak > hi_th and self.current_slice_bw_dl < EMBB_MAX_BW_DL:
            return 'expand', peak
        elif avg < lo_th and self.current_slice_bw_dl > EMBB_MIN_BW_DL:
            return 'contract', avg
        return 'hold', avg

    # ── Action application ─────────────────────────────────────────────────

    def apply_action(self, action: str, trigger_value: float):
        timestamp = datetime.now().isoformat()
        prev_dl   = self.current_slice_bw_dl
        prev_ul   = self.current_slice_bw_ul

        if action == 'expand':
            new_dl = min(EMBB_MAX_BW_DL,
                         prev_dl + max(1, int(round(prev_dl * EMBB_EXPAND_RATIO))))
            new_ul = min(EMBB_MAX_BW_UL,
                         prev_ul + max(1, int(round(prev_ul * EMBB_EXPAND_RATIO))))
            print(f"[eMBB][{timestamp}] ⬆ EXPANDING: Total DL={new_dl}M, UL={new_ul}M "
                  f"(from DL={prev_dl}M, UL={prev_ul}M | peak pred={trigger_value:.1f} Mbps)")
            self.current_bw_config   = 'expanded'
            self.current_slice_bw_dl = new_dl
            self.current_slice_bw_ul = new_ul

        elif action == 'contract':
            new_dl = max(EMBB_MIN_BW_DL,
                         prev_dl - max(1, int(round(prev_dl * EMBB_CONTRACT_RATIO))))
            new_ul = max(EMBB_MIN_BW_UL,
                         prev_ul - max(1, int(round(prev_ul * EMBB_CONTRACT_RATIO))))
            print(f"[eMBB][{timestamp}] ⬇ CONTRACTING: Total DL={new_dl}M, UL={new_ul}M "
                  f"(from DL={prev_dl}M, UL={prev_ul}M | avg pred={trigger_value:.1f} Mbps)")
            self.current_bw_config   = 'contracted'
            self.current_slice_bw_dl = new_dl
            self.current_slice_bw_ul = new_ul

        else:
            new_dl = prev_dl
            new_ul = prev_ul
            print(f"[eMBB][{timestamp}] ● HOLD: Total DL={new_dl}M, UL={new_ul}M | "
                  f"pred={trigger_value:.1f} Mbps")
            self.current_bw_config = self.current_bw_config or 'normal'

        self.decisions.append({
            'slice':                  'eMBB',
            'timestamp':              timestamp,
            'action':                 action,
            'trigger_value':          round(float(trigger_value), 2),
            'config':                 self.current_bw_config,
            'num_subscribers':        EMBB_NUM_SUBSCRIBERS,
            'slice_bw_dl_mbps':       self.current_slice_bw_dl,
            'slice_bw_ul_mbps':       self.current_slice_bw_ul,
            'per_subscriber_dl_mbps': round(self.current_slice_bw_dl / EMBB_NUM_SUBSCRIBERS, 2),
            'per_subscriber_ul_mbps': round(self.current_slice_bw_ul / EMBB_NUM_SUBSCRIBERS, 2),
            'high_threshold_mbps':    round(self.current_slice_bw_dl * EMBB_HIGH_THRESHOLD_RATIO, 2),
            'low_threshold_mbps':     round(self.current_slice_bw_dl * EMBB_LOW_THRESHOLD_RATIO, 2),
        })

        # Push AMBR update to all SST=1 subscribers (non-hold only)
        if action != 'hold':
            self.update_open5gs_subscribers(new_dl, new_ul)

    # ── Open5GS subscriber update (SST-based) ─────────────────────────────

    def update_open5gs_subscribers(self, dl_mbps: int, ul_mbps: int):
        """
        Fetch all subscribers, filter by SST=EMBB_SST_TARGET,
        then update session AMBR to reflect the new bandwidth allocation.
        AMBR unit: 2 = Mbps (Open5GS encoding).
        """
        subscribers = self._fetch_all_subscribers()
        updated = 0
        for sub in subscribers:
            imsi     = sub.get('imsi')
            modified = False
            for s_nssai in sub.get('slice', []):
                if s_nssai.get('sst') == EMBB_SST_TARGET:
                    for sess in s_nssai.get('session', []):
                        sess.setdefault('ambr', {})
                        sess['ambr']['downlink'] = {'value': dl_mbps, 'unit': 2}
                        sess['ambr']['uplink']   = {'value': ul_mbps, 'unit': 2}
                        modified = True
            if modified and self._put_subscriber(imsi, sub):
                updated += 1
                print(f"    [eMBB][OK] Updated IMSI {imsi} "
                      f"(SST={EMBB_SST_TARGET}) → DL={dl_mbps}M UL={ul_mbps}M")

        if updated:
            print(f"    [eMBB] Pushed AMBR to {updated} subscriber(s) "
                  f"with SST={EMBB_SST_TARGET}.")
        else:
            print(f"    [eMBB][!] No subscribers found with SST={EMBB_SST_TARGET}.")

    # ── Live helper ────────────────────────────────────────────────────────

    def get_current_throughput(self, csv_path: str) -> float:
        try:
            with open(csv_path, 'r') as f:
                rows = list(csv.DictReader(f))
                if rows and 'throughput_mbps' in rows[-1]:
                    return float(rows[-1]['throughput_mbps'])
        except Exception:
            pass
        return 0.0

    # ── Run modes ──────────────────────────────────────────────────────────

    def run_simulation(self, simulation_data: np.ndarray) -> list:
        print("=" * 60)
        print("  eMBB Controller  —  SIMULATION MODE")
        print("=" * 60)
        print(f"  LSTM window={self.WINDOW_SIZE} | horizon={self.HORIZON}")
        print(f"  Initial Slice BW: DL={self.current_slice_bw_dl}M "
              f"(per-sub={self.current_slice_bw_dl // EMBB_NUM_SUBSCRIBERS}M)")
        print(f"  SST filter: {EMBB_SST_TARGET}")
        T = len(simulation_data)
        print(f"  Data points: {T}\n")

        LOG_EVERY = 5000
        for i in range(self.WINDOW_SIZE, T):
            window      = simulation_data[i - self.WINDOW_SIZE: i]
            predictions = self.predict(window)
            action, val = self.decide_action(predictions)
            self.apply_action(action, val)
            if (i - self.WINDOW_SIZE + 1) % LOG_EVERY == 0:
                pct = 100 * (i - self.WINDOW_SIZE + 1) / (T - self.WINDOW_SIZE)
                print(f"[eMBB] Progress: {i - self.WINDOW_SIZE + 1}/{T - self.WINDOW_SIZE} "
                      f"steps ({pct:.1f}%) | decisions={len(self.decisions)}")

        actions = [d['action'] for d in self.decisions]
        print(f"\n{'=' * 60}")
        print(f"  [eMBB] {actions.count('expand')} expand | "
              f"{actions.count('contract')} contract | {actions.count('hold')} hold")
        return self.decisions

    def run_live(self, csv_path: str):
        print("=" * 60)
        print("  eMBB Controller  —  LIVE MODE")
        print("=" * 60)
        print(f"  Polling '{csv_path}' every {CHECK_INTERVAL}s. Press Ctrl+C to stop.\n")
        try:
            while True:
                tp = self.get_current_throughput(csv_path)
                self.history.append(tp)
                if len(self.history) >= self.WINDOW_SIZE:
                    window      = self.history[-self.WINDOW_SIZE:]
                    predictions = self.predict(window)
                    action, val = self.decide_action(predictions)
                    self.apply_action(action, val)
                else:
                    print(f"[eMBB][LIVE] Warming up… "
                          f"{self.WINDOW_SIZE - len(self.history)} more.")
                time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            print(f"\n[eMBB] Stopped. Total decisions: {len(self.decisions)}")


# ═════════════════════════════════════════════════════════════════════════════
# Process targets
# ═════════════════════════════════════════════════════════════════════════════

def run_urllc_process(mode: str, data_path: str, model_path: str):
    """Entry point for the URLLC subprocess."""
    output_path = os.path.join(ROOT_DIR, 'unified_decisions_urllc.json')

    def _save(ctrl):
        with open(output_path, 'w') as f:
            json.dump(ctrl.decisions, f, indent=2)
        print(f"[URLLC] Decisions saved → {output_path} "
              f"({len(ctrl.decisions)} entries)")

    try:
        ctrl = URLLCController(model_path=model_path)
    except Exception as e:
        print(f"[URLLC] Failed to initialise controller: {e}")
        return

    try:
        if mode == 'live':
            ctrl.run_live(data_path)
        else:
            try:
                df = pd.read_csv(data_path)
                df.columns = df.columns.str.strip()
                df['latency_ms'] = pd.to_numeric(df['latency_ms'], errors='coerce')
                df.dropna(subset=['latency_ms'], inplace=True)
                sim_data = df['latency_ms'].values.astype(np.float32)
            except Exception as e:
                print(f"[URLLC] Failed to load data: {e}")
                return
            ctrl.run_simulation(simulation_data=sim_data)
    except (KeyboardInterrupt, SystemExit):
        print(f"\n[URLLC] Interrupted — saving {len(ctrl.decisions)} partial decisions…")
    finally:
        _save(ctrl)


def run_mmtc_process(mode: str, data_path: str, model_path: str):
    """Entry point for the mMTC subprocess."""
    output_path = os.path.join(ROOT_DIR, 'unified_decisions_mmtc.json')

    def _save(ctrl):
        with open(output_path, 'w') as f:
            json.dump(ctrl.decisions, f, indent=2)
        print(f"[mMTC] Decisions saved → {output_path} "
              f"({len(ctrl.decisions)} entries)")

    try:
        ctrl = MMTCController(model_path=model_path)
    except Exception as e:
        print(f"[mMTC] Failed to initialise controller: {e}")
        return

    try:
        if mode == 'live':
            ctrl.run_live(data_path)
        else:
            try:
                df   = pd.read_csv(data_path)
                data = df['packet_rate'].values.astype(np.float32)
            except Exception as e:
                print(f"[mMTC] Failed to load data: {e}")
                return
            ctrl.run_simulation(simulation_data=data)
    except (KeyboardInterrupt, SystemExit):
        print(f"\n[mMTC] Interrupted — saving {len(ctrl.decisions)} partial decisions…")
    finally:
        _save(ctrl)


def run_embb_process(mode: str, data_path: str, model_path: str):
    """Entry point for the eMBB subprocess."""
    output_path = os.path.join(ROOT_DIR, 'unified_decisions_embb.json')

    def _save(ctrl):
        with open(output_path, 'w') as f:
            json.dump(ctrl.decisions, f, indent=2)
        print(f"[eMBB] Decisions saved → {output_path} "
              f"({len(ctrl.decisions)} entries)")

    try:
        ctrl = EMBBController(model_path=model_path)
    except Exception as e:
        print(f"[eMBB] Failed to initialise controller: {e}")
        return

    try:
        if mode == 'live':
            ctrl.run_live(data_path)
        else:
            try:
                df   = pd.read_csv(data_path)
                data = df['throughput_mbps'].values.astype(np.float32)
            except Exception as e:
                print(f"[eMBB] Failed to load data: {e}")
                return
            ctrl.run_simulation(simulation_data=data)
    except (KeyboardInterrupt, SystemExit):
        print(f"\n[eMBB] Interrupted — saving {len(ctrl.decisions)} partial decisions…")
    finally:
        _save(ctrl)


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Unified Zero-Touch 5G Slice Controller (eMBB + mMTC + URLLC)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["sim", "live"], default="sim",
        help="'sim'  → offline simulation (default)\n'live' → live polling",
    )
    parser.add_argument("--urllc-data",
        default=os.path.join(URLLC_DIR, 'data', 'urllc_timeseries.csv'),
        help="URLLC latency CSV ('latency_ms' column required)")
    parser.add_argument("--mmtc-data",
        default=os.path.join(MMTC_DIR, 'data', 'mmtc_traffic_timeseries.csv'),
        help="mMTC traffic CSV ('packet_rate' column required)")
    parser.add_argument("--embb-data",
        default=os.path.join(EMBB_DIR, 'data', 'embb_traffic_timeseries.csv'),
        help="eMBB traffic CSV ('throughput_mbps' column required)")
    parser.add_argument("--urllc-model",
        default=os.path.join(URLLC_DIR, 'saved', 'lstm_urllc.pth'))
    parser.add_argument("--mmtc-model",
        default=os.path.join(MMTC_DIR, 'models', 'saved', 'lstm_mmtc.pth'))
    parser.add_argument("--embb-model",
        default=os.path.join(EMBB_DIR, 'models', 'saved', 'lstm_embb.pth'))

    args = parser.parse_args()

    print("=" * 70)
    print("  Unified Zero-Touch 5G Slice Controller")
    print("=" * 70)
    print(f"  Mode : {args.mode.upper()}")
    print(f"  SST assignments: eMBB={EMBB_SST_TARGET}, mMTC={MMTC_SST_TARGET}, "
          f"URLLC={URLLC_SST_TARGET}")
    print(f"  URLLC data  : {args.urllc_data}")
    print(f"  mMTC  data  : {args.mmtc_data}")
    print(f"  eMBB  data  : {args.embb_data}")
    print("=" * 70)
    print("  Spawning 3 slice controller processes…\n")

    processes = [
        multiprocessing.Process(
            target=run_urllc_process,
            args=(args.mode, args.urllc_data, args.urllc_model),
            name="URLLC-Controller",
        ),
        multiprocessing.Process(
            target=run_mmtc_process,
            args=(args.mode, args.mmtc_data, args.mmtc_model),
            name="mMTC-Controller",
        ),
        multiprocessing.Process(
            target=run_embb_process,
            args=(args.mode, args.embb_data, args.embb_model),
            name="eMBB-Controller",
        ),
    ]

    for p in processes:
        p.start()
        print(f"  [PID {p.pid}] Started {p.name}")

    print()
    try:
        for p in processes:
            p.join()
            status = "OK" if p.exitcode == 0 else f"EXITED {p.exitcode}"
            print(f"  [{p.name}] {status}")
    except KeyboardInterrupt:
        print("\n  Stopping all controllers…")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
        print("  All processes terminated.")

    print("\n  Output files:")
    for fname in ['unified_decisions_urllc.json',
                  'unified_decisions_mmtc.json',
                  'unified_decisions_embb.json']:
        fpath = os.path.join(ROOT_DIR, fname)
        size  = os.path.getsize(fpath) if os.path.exists(fpath) else None
        note  = f"{size // 1024} KB" if size else "not created"
        print(f"    {fname}  ({note})")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
