"""
Static Reactive 5G Slice Controller
====================================
Manages three 5G network slices concurrently using separate OS processes:

  • eMBB  (SST=1) — bandwidth expand/contract + AMBR update via Open5GS API
  • mMTC  (SST=3) — connection-capacity expand/contract + AMBR update via Open5GS API
  • URLLC (SST=2) — latency-aware 5QI/ARP elevation via Open5GS API

Unlike the LSTM-based unified controller, this controller makes decisions
based on the **current observed metric value** (static reactive / threshold-
based approach) — no predictive model is involved.

All three controllers:
  - Authenticate to Open5GS WebUI using CSRF-token + JWT flow
  - Fetch ALL subscribers, filter by SST value (no hardcoded IMSI lists)
  - Push configuration updates via REST PUT per subscriber

Each slice runs in its own multiprocessing.Process.

Usage
-----
    python3 reactive_controller.py               # Simulation mode (default)
    python3 reactive_controller.py --mode live   # Live polling mode

    python3 reactive_controller.py \\
        --urllc-data /path/to/urllc.csv \\
        --mmtc-data  /path/to/mmtc.csv  \\
        --embb-data  /path/to/embb.csv
"""

from __future__ import annotations

import os
import re
import csv
import json
import signal
import time
import argparse
import multiprocessing
from datetime import datetime
from typing import Optional

import numpy as np
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
# URLLC Controller  (SST=2)  — Static Reactive
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
    Static reactive controller for the URLLC slice (SST=2).

    Compares the current observed latency against static thresholds and
    elevates or relaxes 5QI + ARP priority for every subscriber with SST == 2.
    No predictive model is used.
    """

    SLICE_TAG = "URLLC"

    def __init__(self):
        self.current_state = URLLC_QOS_NORMAL['state_name']
        self.decisions: list = []
        self._init_session()

    # ── Decision ───────────────────────────────────────────────────────────

    def decide_action(self, current_latency: float):
        if (current_latency > URLLC_HIGH_LATENCY_THRESHOLD
                and self.current_state != URLLC_QOS_ELEVATED['state_name']):
            return 'elevate', current_latency
        if (current_latency < URLLC_LOW_LATENCY_THRESHOLD
                and self.current_state != URLLC_QOS_NORMAL['state_name']):
            return 'relax', current_latency
        return 'hold', current_latency

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
        print("  URLLC Controller  —  SIMULATION MODE (Reactive)")
        print("=" * 60)
        print(f"  HIGH: {URLLC_HIGH_LATENCY_THRESHOLD} ms | LOW: {URLLC_LOW_LATENCY_THRESHOLD} ms")
        print(f"  SST filter: {URLLC_SST_TARGET}")
        T = len(simulation_data)
        print(f"  Data points: {T}\n")

        LOG_EVERY = 5000
        DECISION_EVERY = 100
        for i in range(0, T, DECISION_EVERY):
            chunk = simulation_data[i:i + DECISION_EVERY]
            current_val = float(np.mean(chunk))
            action, val = self.decide_action(current_val)
            self.apply_qos(action, val)
            if (i + DECISION_EVERY) % LOG_EVERY < DECISION_EVERY:
                pct = 100 * min(i + DECISION_EVERY, T) / T
                print(f"[URLLC] Progress: {min(i + DECISION_EVERY, T)}/{T} "
                      f"steps ({pct:.1f}%) | decisions={len(self.decisions)}")

        actions = [d['action'] for d in self.decisions]
        print(f"\n{'=' * 60}")
        print(f"  [URLLC] {actions.count('elevate')} elevate | "
              f"{actions.count('relax')} relax | {actions.count('hold')} hold")
        return self.decisions

    def run_live(self, csv_path: str):
        print("=" * 60)
        print("  URLLC Controller  —  LIVE MODE (Reactive)")
        print("=" * 60)
        print(f"  Polling '{csv_path}' every {CHECK_INTERVAL}s. Press Ctrl+C to stop.\n")
        try:
            while True:
                latency = self.get_current_latency(csv_path)
                action, val = self.decide_action(latency)
                self.apply_qos(action, val)
                time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            print(f"\n[URLLC] Stopped. Total decisions: {len(self.decisions)}")


# ═════════════════════════════════════════════════════════════════════════════
# mMTC Controller  (SST=3)  — Static Reactive
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
    Static reactive controller for the mMTC slice (SST=3).

    Compares the current observed packet_rate against capacity thresholds and
    expands or contracts connection capacity + slice AMBR for every subscriber
    with SST == 3.  No predictive model is used.
    """

    SLICE_TAG = "mMTC"

    def __init__(self):
        self.current_config       = 'normal'
        self.current_capacity_pps = MMTC_INITIAL_CAPACITY
        self.current_slice_bw_dl  = MMTC_INITIAL_BW_DL
        self.current_slice_bw_ul  = MMTC_INITIAL_BW_UL
        self.decisions: list      = []
        self._init_session()

    # ── Decision ───────────────────────────────────────────────────────────

    def decide_action(self, current_packet_rate: float):
        hi_th = self.current_capacity_pps * MMTC_HIGH_THRESHOLD_RATIO
        lo_th = self.current_capacity_pps * MMTC_LOW_THRESHOLD_RATIO
        if current_packet_rate > hi_th and self.current_capacity_pps < MMTC_MAX_CAPACITY:
            return 'expand', current_packet_rate
        elif current_packet_rate < lo_th and self.current_capacity_pps > MMTC_MIN_CAPACITY:
            return 'contract', current_packet_rate
        return 'hold', current_packet_rate

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
                  f"current={trigger_value:.1f} pps")
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
                  f"current={trigger_value:.1f} pps")
            self.current_config       = 'contracted'
            self.current_capacity_pps = new_cap
            self.current_slice_bw_dl  = new_dl
            self.current_slice_bw_ul  = new_ul

        else:
            new_cap = prev_cap
            new_dl  = prev_dl
            new_ul  = prev_ul
            print(f"[mMTC][{timestamp}] ● HOLD: Capacity={new_cap} pps, "
                  f"BW DL={new_dl}M | current={trigger_value:.1f} pps")
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

        if action != 'hold':
            self.update_open5gs_subscribers(new_dl, new_ul)

    # ── Open5GS subscriber update (SST-based) ─────────────────────────────

    def update_open5gs_subscribers(self, dl_mbps: int, ul_mbps: int):
        subscribers = self._fetch_all_subscribers()
        updated = 0
        for sub in subscribers:
            imsi     = sub.get('imsi')
            modified = False
            for s_nssai in sub.get('slice', []):
                if s_nssai.get('sst') == MMTC_SST_TARGET:
                    for sess in s_nssai.get('session', []):
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
        print("  mMTC Controller  —  SIMULATION MODE (Reactive)")
        print("=" * 60)
        print(f"  Initial: {self.current_capacity_pps} pps, "
              f"DL={self.current_slice_bw_dl}M UL={self.current_slice_bw_ul}M")
        print(f"  SST filter: {MMTC_SST_TARGET}")
        T = len(simulation_data)
        print(f"  Data points: {T}\n")

        LOG_EVERY = 5000
        for i in range(T):
            current_val = float(simulation_data[i])
            action, val = self.decide_action(current_val)
            self.apply_action(action, val)
            if (i + 1) % LOG_EVERY == 0:
                pct = 100 * (i + 1) / T
                print(f"[mMTC] Progress: {i + 1}/{T} "
                      f"steps ({pct:.1f}%) | decisions={len(self.decisions)}")

        actions = [d['action'] for d in self.decisions]
        print(f"\n{'=' * 60}")
        print(f"  [mMTC] {actions.count('expand')} expand | "
              f"{actions.count('contract')} contract | {actions.count('hold')} hold")
        return self.decisions

    def run_live(self, csv_path: str):
        print("=" * 60)
        print("  mMTC Controller  —  LIVE MODE (Reactive)")
        print("=" * 60)
        print(f"  Polling '{csv_path}' every {CHECK_INTERVAL}s. Press Ctrl+C to stop.\n")
        try:
            while True:
                pkt_rate = self.get_current_packet_rate(csv_path)
                action, val = self.decide_action(pkt_rate)
                self.apply_action(action, val)
                time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            print(f"\n[mMTC] Stopped. Total decisions: {len(self.decisions)}")


# ═════════════════════════════════════════════════════════════════════════════
# eMBB Controller  (SST=1)  — Static Reactive
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
    Static reactive controller for the eMBB slice (SST=1).

    Compares the current observed throughput against capacity thresholds and
    expands or contracts slice AMBR for every subscriber with SST == 1.
    No predictive model is used.
    """

    SLICE_TAG = "eMBB"

    def __init__(self):
        self.current_bw_config   = 'normal'
        self.current_slice_bw_dl = EMBB_INITIAL_BW_DL
        self.current_slice_bw_ul = EMBB_INITIAL_BW_UL
        self.decisions: list     = []
        self._init_session()

    # ── Decision ───────────────────────────────────────────────────────────

    def decide_action(self, current_throughput: float):
        hi_th = self.current_slice_bw_dl * EMBB_HIGH_THRESHOLD_RATIO
        lo_th = self.current_slice_bw_dl * EMBB_LOW_THRESHOLD_RATIO
        if current_throughput > hi_th and self.current_slice_bw_dl < EMBB_MAX_BW_DL:
            return 'expand', current_throughput
        elif current_throughput < lo_th and self.current_slice_bw_dl > EMBB_MIN_BW_DL:
            return 'contract', current_throughput
        return 'hold', current_throughput

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
                  f"(from DL={prev_dl}M, UL={prev_ul}M | current={trigger_value:.1f} Mbps)")
            self.current_bw_config   = 'expanded'
            self.current_slice_bw_dl = new_dl
            self.current_slice_bw_ul = new_ul

        elif action == 'contract':
            new_dl = max(EMBB_MIN_BW_DL,
                         prev_dl - max(1, int(round(prev_dl * EMBB_CONTRACT_RATIO))))
            new_ul = max(EMBB_MIN_BW_UL,
                         prev_ul - max(1, int(round(prev_ul * EMBB_CONTRACT_RATIO))))
            print(f"[eMBB][{timestamp}] ⬇ CONTRACTING: Total DL={new_dl}M, UL={new_ul}M "
                  f"(from DL={prev_dl}M, UL={prev_ul}M | current={trigger_value:.1f} Mbps)")
            self.current_bw_config   = 'contracted'
            self.current_slice_bw_dl = new_dl
            self.current_slice_bw_ul = new_ul

        else:
            new_dl = prev_dl
            new_ul = prev_ul
            print(f"[eMBB][{timestamp}] ● HOLD: Total DL={new_dl}M, UL={new_ul}M | "
                  f"current={trigger_value:.1f} Mbps")
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

        if action != 'hold':
            self.update_open5gs_subscribers(new_dl, new_ul)

    # ── Open5GS subscriber update (SST-based) ─────────────────────────────

    def update_open5gs_subscribers(self, dl_mbps: int, ul_mbps: int):
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
        print("  eMBB Controller  —  SIMULATION MODE (Reactive)")
        print("=" * 60)
        print(f"  Initial Slice BW: DL={self.current_slice_bw_dl}M "
              f"(per-sub={self.current_slice_bw_dl // EMBB_NUM_SUBSCRIBERS}M)")
        print(f"  SST filter: {EMBB_SST_TARGET}")
        T = len(simulation_data)
        print(f"  Data points: {T}\n")

        LOG_EVERY = 5000
        for i in range(T):
            current_val = float(simulation_data[i])
            action, val = self.decide_action(current_val)
            self.apply_action(action, val)
            if (i + 1) % LOG_EVERY == 0:
                pct = 100 * (i + 1) / T
                print(f"[eMBB] Progress: {i + 1}/{T} "
                      f"steps ({pct:.1f}%) | decisions={len(self.decisions)}")

        actions = [d['action'] for d in self.decisions]
        print(f"\n{'=' * 60}")
        print(f"  [eMBB] {actions.count('expand')} expand | "
              f"{actions.count('contract')} contract | {actions.count('hold')} hold")
        return self.decisions

    def run_live(self, csv_path: str):
        print("=" * 60)
        print("  eMBB Controller  —  LIVE MODE (Reactive)")
        print("=" * 60)
        print(f"  Polling '{csv_path}' every {CHECK_INTERVAL}s. Press Ctrl+C to stop.\n")
        try:
            while True:
                tp = self.get_current_throughput(csv_path)
                action, val = self.decide_action(tp)
                self.apply_action(action, val)
                time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            print(f"\n[eMBB] Stopped. Total decisions: {len(self.decisions)}")


# ═════════════════════════════════════════════════════════════════════════════
# Process targets
# ═════════════════════════════════════════════════════════════════════════════

def run_urllc_process(mode: str, data_path: str):
    """Entry point for the URLLC subprocess."""
    output_path = os.path.join(ROOT_DIR, 'reactive_decisions_urllc.json')

    def _save(ctrl):
        prev = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            with open(output_path, 'w') as f:
                json.dump(ctrl.decisions, f, indent=2)
            print(f"[URLLC] Decisions saved → {output_path} "
                  f"({len(ctrl.decisions)} entries)")
        finally:
            signal.signal(signal.SIGINT, prev)

    try:
        ctrl = URLLCController()
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


def run_mmtc_process(mode: str, data_path: str):
    """Entry point for the mMTC subprocess."""
    output_path = os.path.join(ROOT_DIR, 'reactive_decisions_mmtc.json')

    def _save(ctrl):
        prev = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            with open(output_path, 'w') as f:
                json.dump(ctrl.decisions, f, indent=2)
            print(f"[mMTC] Decisions saved → {output_path} "
                  f"({len(ctrl.decisions)} entries)")
        finally:
            signal.signal(signal.SIGINT, prev)

    try:
        ctrl = MMTCController()
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


def run_embb_process(mode: str, data_path: str):
    """Entry point for the eMBB subprocess."""
    output_path = os.path.join(ROOT_DIR, 'reactive_decisions_embb.json')

    def _save(ctrl):
        prev = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            with open(output_path, 'w') as f:
                json.dump(ctrl.decisions, f, indent=2)
            print(f"[eMBB] Decisions saved → {output_path} "
                  f"({len(ctrl.decisions)} entries)")
        finally:
            signal.signal(signal.SIGINT, prev)

    try:
        ctrl = EMBBController()
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
# Post-simulation visualisation  (one summary image per slice)
# ═════════════════════════════════════════════════════════════════════════════

def _style_ax(ax, title: str, ylabel: str):
    """Apply the dark-theme style to an Axes object."""
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='white', labelsize=7)
    ax.set_title(title, color='white', fontsize=10, pad=6)
    ax.set_ylabel(ylabel, color='#aaaaaa', fontsize=9)
    ax.spines[:].set_color('#444466')
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha('right')
        lbl.set_fontsize(6)


def _xtick_labels(timestamps: list, max_ticks: int = 20) -> list:
    """Subsample x-axis labels to at most max_ticks entries."""
    n   = len(timestamps)
    if n <= max_ticks:
        return [t[11:19] for t in timestamps]  # HH:MM:SS
    step  = max(1, n // max_ticks)
    shown = [t[11:19] if i % step == 0 else '' for i, t in enumerate(timestamps)]
    return shown


def _plot_urllc_summary(decisions: list, out_dir: str) -> str:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print('[URLLC] matplotlib not available — skipping visualization.')
        return ''

    if not decisions:
        print('[URLLC] No decisions to plot.')
        return ''

    ts      = [d['timestamp']       for d in decisions]
    lat     = [d.get('trigger_latency', 0) for d in decisions]
    fqi     = [d.get('applied_5qi') or (85 if d.get('state') == 'NORMAL' else 82)
               for d in decisions]
    arp     = [d.get('applied_arp')  or (5  if d.get('state') == 'NORMAL' else 1)
               for d in decisions]
    actions = [d['action'] for d in decisions]

    HIGH = URLLC_HIGH_LATENCY_THRESHOLD
    LOW  = URLLC_LOW_LATENCY_THRESHOLD

    elev_color = '#e74c3c'
    norm_color = '#2ecc71'
    hold_color = '#3498db'

    point_colors = [elev_color if a == 'elevate' else
                    (norm_color if a == 'relax'   else hold_color)
                    for a in actions]
    xlbls = _xtick_labels(ts)
    x     = list(range(len(ts)))

    fig = plt.figure(figsize=(16, 10), facecolor='#1a1a2e')
    fig.suptitle(
        f'URLLC Slice — Reactive Summary  |  {len(decisions)} decisions'
        f'  |  {ts[0][0:10]} → {ts[-1][0:10]}',
        fontsize=13, color='white', fontweight='bold', y=0.98
    )
    gs = gridspec.GridSpec(2, 2, left=0.07, right=0.96, hspace=0.40, wspace=0.28)

    # Panel 1 — Latency trigger value
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, lat, color='#00d4ff', linewidth=1.0, alpha=0.7)
    ax1.scatter(x, lat, c=point_colors, s=12, zorder=4)
    ax1.axhline(HIGH, color=elev_color, linestyle='--', linewidth=1,
                label=f'HIGH {HIGH} ms')
    ax1.axhline(LOW,  color=norm_color, linestyle='--', linewidth=1,
                label=f'LOW {LOW} ms')
    ax1.fill_between(x, lat, HIGH,
                     where=[l > HIGH for l in lat],
                     color=elev_color, alpha=0.18)
    ax1.legend(fontsize=7, labelcolor='white',
               facecolor='#16213e', edgecolor='none')
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlbls)
    _style_ax(ax1, '📡 Current Latency (ms)', 'ms')

    # Panel 2 — 5QI over time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.step(x, fqi, color='#f39c12', linewidth=1.5, where='post')
    ax2.scatter(x, fqi, c=point_colors, s=12, zorder=4)
    ax2.set_ylim(80, 87)
    ax2.set_yticks([82, 85])
    ax2.set_yticklabels(['82 (ELEVATED)', '85 (NORMAL)'], fontsize=8, color='white')
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlbls)
    _style_ax(ax2, '⚡ 5QI Over Time', '5QI')

    # Panel 3 — ARP priority
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.step(x, arp, color='#9b59b6', linewidth=1.5, where='post')
    ax3.scatter(x, arp, c=point_colors, s=12, zorder=4)
    ax3.set_ylim(0, 7)
    ax3.set_yticks([1, 5])
    ax3.set_yticklabels(['1 (ELEVATED)', '5 (NORMAL)'], fontsize=8, color='white')
    ax3.invert_yaxis()
    ax3.set_xticks(x)
    ax3.set_xticklabels(xlbls)
    _style_ax(ax3, '🎯 ARP Priority (lower = higher priority)', 'Priority')

    # Panel 4 — Action distribution bar
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#16213e')
    cats   = ['elevate', 'relax', 'hold']
    counts = [actions.count(c) for c in cats]
    bars   = ax4.bar(cats, counts,
                     color=[elev_color, norm_color, hold_color],
                     edgecolor='#444466', width=0.5)
    for bar, cnt in zip(bars, counts):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(counts) * 0.02,
                 str(cnt), ha='center', va='bottom',
                 color='white', fontsize=10, fontweight='bold')
    ax4.tick_params(colors='white')
    ax4.set_title('📊 Decision Distribution', color='white', fontsize=10, pad=6)
    ax4.set_ylabel('Count', color='#aaaaaa', fontsize=9)
    ax4.spines[:].set_color('#444466')
    ax4.set_ylim(0, max(counts) * 1.15 + 1)

    # State badge
    last_state = decisions[-1].get('state', 'NORMAL')
    badge_col  = elev_color if 'ELEVATED' in last_state else norm_color
    fig.text(0.5, 0.01, f'Final State: {last_state}',
             ha='center', fontsize=12, fontweight='bold', color='white',
             bbox=dict(facecolor=badge_col, edgecolor='none',
                       pad=6, boxstyle='round'))

    out = os.path.join(out_dir, 'reactive_summary_urllc.png')
    plt.savefig(out, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    return out


def _plot_mmtc_summary(decisions: list, out_dir: str) -> str:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print('[mMTC] matplotlib not available — skipping visualization.')
        return ''

    if not decisions:
        print('[mMTC] No decisions to plot.')
        return ''

    ts       = [d['timestamp']                   for d in decisions]
    trig     = [d.get('trigger_value', 0)        for d in decisions]
    cap      = [d.get('capacity_pps', 0)         for d in decisions]
    bw_dl    = [d.get('slice_bw_dl_mbps', 0)     for d in decisions]
    bw_ul    = [d.get('slice_bw_ul_mbps', 0)     for d in decisions]
    actions  = [d['action']                      for d in decisions]

    exp_color  = '#e67e22'
    con_color  = '#3498db'
    hold_color = '#95a5a6'

    point_colors = [exp_color  if a == 'expand'   else
                    (con_color if a == 'contract' else hold_color)
                    for a in actions]
    xlbls = _xtick_labels(ts)
    x     = list(range(len(ts)))

    fig = plt.figure(figsize=(16, 10), facecolor='#1a1a2e')
    fig.suptitle(
        f'mMTC Slice — Reactive Summary  |  {len(decisions)} decisions'
        f'  |  {ts[0][0:10]} → {ts[-1][0:10]}',
        fontsize=13, color='white', fontweight='bold', y=0.98
    )
    gs = gridspec.GridSpec(2, 2, left=0.07, right=0.96, hspace=0.40, wspace=0.28)

    # Panel 1 — Packet-rate trigger
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, trig, color='#00d4ff', linewidth=1.0, alpha=0.7)
    ax1.scatter(x, trig, c=point_colors, s=12, zorder=4)
    ax1.set_xticks(x); ax1.set_xticklabels(xlbls)
    _style_ax(ax1, '📡 Current Packet Rate (pps)', 'pps')

    # Panel 2 — Connection capacity
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.step(x, cap, color='#f39c12', linewidth=1.5, where='post')
    ax2.scatter(x, cap, c=point_colors, s=12, zorder=4)
    ax2.set_xticks(x); ax2.set_xticklabels(xlbls)
    _style_ax(ax2, '🔗 Connection Capacity (pps)', 'pps')

    # Panel 3 — Slice bandwidth DL/UL
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.step(x, bw_dl, color='#2ecc71', linewidth=1.5, where='post', label='DL')
    ax3.step(x, bw_ul, color='#e74c3c',  linewidth=1.5, where='post', label='UL', linestyle='--')
    ax3.legend(fontsize=7, labelcolor='white', facecolor='#16213e', edgecolor='none')
    ax3.set_xticks(x); ax3.set_xticklabels(xlbls)
    _style_ax(ax3, '📶 Slice Bandwidth (Mbps)', 'Mbps')

    # Panel 4 — Action distribution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#16213e')
    cats   = ['expand', 'contract', 'hold']
    counts = [actions.count(c) for c in cats]
    bars   = ax4.bar(cats, counts,
                     color=[exp_color, con_color, hold_color],
                     edgecolor='#444466', width=0.5)
    for bar, cnt in zip(bars, counts):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(counts) * 0.02,
                 str(cnt), ha='center', va='bottom',
                 color='white', fontsize=10, fontweight='bold')
    ax4.tick_params(colors='white')
    ax4.set_title('📊 Decision Distribution', color='white', fontsize=10, pad=6)
    ax4.set_ylabel('Count', color='#aaaaaa', fontsize=9)
    ax4.spines[:].set_color('#444466')
    ax4.set_ylim(0, max(counts) * 1.15 + 1)

    # Config badge
    last_cfg  = decisions[-1].get('config', 'normal')
    badge_col = exp_color if last_cfg == 'expanded' else (
                con_color if last_cfg == 'contracted' else '#7f8c8d')
    fig.text(0.5, 0.01, f'Final Config: {last_cfg.upper()}',
             ha='center', fontsize=12, fontweight='bold', color='white',
             bbox=dict(facecolor=badge_col, edgecolor='none',
                       pad=6, boxstyle='round'))

    out = os.path.join(out_dir, 'reactive_summary_mmtc.png')
    plt.savefig(out, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    return out


def _plot_embb_summary(decisions: list, out_dir: str) -> str:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print('[eMBB] matplotlib not available — skipping visualization.')
        return ''

    if not decisions:
        print('[eMBB] No decisions to plot.')
        return ''

    ts       = [d['timestamp']                          for d in decisions]
    trig     = [d.get('trigger_value', 0)               for d in decisions]
    bw_dl    = [d.get('slice_bw_dl_mbps', 0)            for d in decisions]
    bw_ul    = [d.get('slice_bw_ul_mbps', 0)            for d in decisions]
    sub_dl   = [d.get('per_subscriber_dl_mbps', 0)      for d in decisions]
    sub_ul   = [d.get('per_subscriber_ul_mbps', 0)      for d in decisions]
    actions  = [d['action']                             for d in decisions]

    exp_color  = '#e67e22'
    con_color  = '#3498db'
    hold_color = '#95a5a6'

    point_colors = [exp_color  if a == 'expand'   else
                    (con_color if a == 'contract' else hold_color)
                    for a in actions]
    xlbls = _xtick_labels(ts)
    x     = list(range(len(ts)))

    fig = plt.figure(figsize=(16, 10), facecolor='#1a1a2e')
    fig.suptitle(
        f'eMBB Slice — Reactive Summary  |  {len(decisions)} decisions'
        f'  |  {ts[0][0:10]} → {ts[-1][0:10]}',
        fontsize=13, color='white', fontweight='bold', y=0.98
    )
    gs = gridspec.GridSpec(2, 2, left=0.07, right=0.96, hspace=0.40, wspace=0.28)

    # Panel 1 — Trigger throughput
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, trig, color='#00d4ff', linewidth=1.0, alpha=0.7)
    ax1.scatter(x, trig, c=point_colors, s=12, zorder=4)
    ax1.set_xticks(x); ax1.set_xticklabels(xlbls)
    _style_ax(ax1, '📡 Current Throughput (Mbps)', 'Mbps')

    # Panel 2 — Slice BW DL + UL
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.step(x, bw_dl, color='#2ecc71', linewidth=1.5, where='post', label='DL')
    ax2.step(x, bw_ul, color='#e74c3c',  linewidth=1.5, where='post', label='UL', linestyle='--')
    ax2.legend(fontsize=7, labelcolor='white', facecolor='#16213e', edgecolor='none')
    ax2.set_xticks(x); ax2.set_xticklabels(xlbls)
    _style_ax(ax2, '📶 Slice Bandwidth (Mbps)', 'Mbps')

    # Panel 3 — Per-subscriber BW
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.step(x, sub_dl, color='#f39c12', linewidth=1.5, where='post', label='DL/sub')
    ax3.step(x, sub_ul, color='#9b59b6', linewidth=1.5, where='post', label='UL/sub', linestyle='--')
    ax3.legend(fontsize=7, labelcolor='white', facecolor='#16213e', edgecolor='none')
    ax3.set_xticks(x); ax3.set_xticklabels(xlbls)
    _style_ax(ax3, '👤 Per-Subscriber BW (Mbps)', 'Mbps')

    # Panel 4 — Action distribution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#16213e')
    cats   = ['expand', 'contract', 'hold']
    counts = [actions.count(c) for c in cats]
    bars   = ax4.bar(cats, counts,
                     color=[exp_color, con_color, hold_color],
                     edgecolor='#444466', width=0.5)
    for bar, cnt in zip(bars, counts):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(counts) * 0.02,
                 str(cnt), ha='center', va='bottom',
                 color='white', fontsize=10, fontweight='bold')
    ax4.tick_params(colors='white')
    ax4.set_title('📊 Decision Distribution', color='white', fontsize=10, pad=6)
    ax4.set_ylabel('Count', color='#aaaaaa', fontsize=9)
    ax4.spines[:].set_color('#444466')
    ax4.set_ylim(0, max(counts) * 1.15 + 1)

    # Config badge
    last_cfg  = decisions[-1].get('config', 'normal')
    badge_col = exp_color if last_cfg == 'expanded' else (
                con_color if last_cfg == 'contracted' else '#7f8c8d')
    fig.text(0.5, 0.01, f'Final Config: {last_cfg.upper()}',
             ha='center', fontsize=12, fontweight='bold', color='white',
             bbox=dict(facecolor=badge_col, edgecolor='none',
                       pad=6, boxstyle='round'))

    out = os.path.join(out_dir, 'reactive_summary_embb.png')
    plt.savefig(out, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    return out


def generate_slice_visualizations(out_dir: Optional[str] = None):
    if out_dir is None:
        out_dir = os.path.join(ROOT_DIR, 'reactive_visualizations')
    os.makedirs(out_dir, exist_ok=True)

    mapping = [
        ('URLLC', os.path.join(ROOT_DIR, 'reactive_decisions_urllc.json'),
         _plot_urllc_summary),
        ('mMTC',  os.path.join(ROOT_DIR, 'reactive_decisions_mmtc.json'),
         _plot_mmtc_summary),
        ('eMBB',  os.path.join(ROOT_DIR, 'reactive_decisions_embb.json'),
         _plot_embb_summary),
    ]

    print('\n  Generating reactive slice summary visualizations…')
    saved = []
    for label, json_path, plot_fn in mapping:
        if not os.path.exists(json_path):
            print(f'  [{label}] JSON not found ({json_path}) — skipping.')
            continue
        try:
            with open(json_path) as f:
                decisions = json.load(f)
            if not decisions:
                print(f'  [{label}] No decisions in JSON — skipping.')
                continue
            out = plot_fn(decisions, out_dir)
            if out:
                print(f'  [{label}] 📊 Summary saved → {out}')
                saved.append(out)
        except Exception as e:
            print(f'  [{label}] Visualization error: {e}')

    if saved:
        print(f'  {len(saved)}/3 visualizations written to {out_dir}/')
    return saved


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Static Reactive 5G Slice Controller (eMBB + mMTC + URLLC)",
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

    args = parser.parse_args()

    print("=" * 70)
    print("  Static Reactive 5G Slice Controller")
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
            args=(args.mode, args.urllc_data),
            name="URLLC-Controller",
        ),
        multiprocessing.Process(
            target=run_mmtc_process,
            args=(args.mode, args.mmtc_data),
            name="mMTC-Controller",
        ),
        multiprocessing.Process(
            target=run_embb_process,
            args=(args.mode, args.embb_data),
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
        print("\n  Ctrl+C received — gracefully stopping all controllers…")
        # Children already received SIGINT from the process group;
        # do NOT send a second one — it would interrupt their _save().
        # Just wait for them to finish saving (up to 15s each).
        for p in processes:
            p.join(timeout=15)
            if p.is_alive():
                print(f"  [{p.name}] did not exit in time, terminating…")
                p.terminate()
                p.join(timeout=5)
            status = "OK" if p.exitcode == 0 else f"INTERRUPTED (exit {p.exitcode})"
            print(f"  [{p.name}] {status}")

    print("\n  Output files:")
    for fname in ['reactive_decisions_urllc.json',
                  'reactive_decisions_mmtc.json',
                  'reactive_decisions_embb.json']:
        fpath = os.path.join(ROOT_DIR, fname)
        size  = os.path.getsize(fpath) if os.path.exists(fpath) else None
        note  = f"{size // 1024} KB" if size else "not created"
        print(f"    {fname}  ({note})")

    # ── Generate one summary PNG per slice once all processes are done ──────
    generate_slice_visualizations()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
