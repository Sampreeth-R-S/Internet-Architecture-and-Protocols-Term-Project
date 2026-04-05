#!/usr/bin/env python3
"""
Cross-Controller Performance Comparison
========================================
Compares three 5G network-slicing controllers across all three slices:

  Controllers:  Static  |  Reactive (threshold)  |  LSTM (proactive)
  Slices:       eMBB (SST=1) · mMTC (SST=3) · URLLC (SST=2)

Data sources
------------
  Stats CSVs:
    <slice>/data/lstm_controller_stats.csv      (or lstm_controller_stat.csv)
    <slice>/data/reactive_controller_stats.csv
    <slice>/data/static_stats.csv

  Decision JSONs:
    unified_decisions_{embb,mmtc,urllc}.json   (LSTM controller)
    reactive_decisions_{embb,mmtc,urllc}.json  (Reactive controller)

Missing files are handled gracefully — comparisons are produced for
whatever data is available.

Outputs are saved to  comparison_visualizations/

Usage:
    python3 compare_controllers.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── paths ────────────────────────────────────────────────────────────────────
ROOT  = os.path.dirname(os.path.abspath(__file__))
OUT   = os.path.join(ROOT, "comparison_visualizations")
os.makedirs(OUT, exist_ok=True)

SLICE_DIRS = {
    "embb":  os.path.join(ROOT, "5g-slicing-embb"),
    "mmtc":  os.path.join(ROOT, "5g-slicing-mmtc"),
    "urllc": os.path.join(ROOT, "5g-slicing-urllc"),
}

# ─── static allocation defaults (used to simulate a baseline) ─────────────────
# These match the initial / default allocation each controller starts with.
STATIC_DEFAULTS = {
    "embb": {
        "slice_bw_dl_mbps": 500,
        "slice_bw_ul_mbps": 250,
        "num_subscribers":  5,
    },
    "mmtc": {
        "capacity_pps":     500,
        "slice_bw_dl_mbps": 10,
        "slice_bw_ul_mbps": 5,
    },
    "urllc": {
        "5qi":          85,
        "arp_priority": 5,
    },
}


# ═════════════════════════════════════════════════════════════════════════════
#  Loading helpers
# ═════════════════════════════════════════════════════════════════════════════

def _load_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"  [warn] Cannot read {path}: {e}")
        return None


def _load_json(path: str) -> Optional[List[dict]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if not data:
            return None
        return data
    except Exception as e:
        print(f"  [warn] Cannot read {path}: {e}")
        return None


def _find_stats_csv(slice_key: str, controller: str) -> Optional[pd.DataFrame]:
    """Try multiple naming conventions for the stats CSV."""
    data_dir = os.path.join(SLICE_DIRS[slice_key], "data")
    candidates = []
    if controller == "lstm":
        candidates = [
            os.path.join(data_dir, "lstm_controller_stats.csv"),
            os.path.join(data_dir, "lstm_controller_stat.csv"),
        ]
    elif controller == "reactive":
        candidates = [
            os.path.join(data_dir, "reactive_controller_stats.csv"),
        ]
    elif controller == "static":
        candidates = [
            os.path.join(data_dir, "static_stats.csv"),
        ]
    for c in candidates:
        df = _load_csv(c)
        if df is not None:
            return df
    return None


def _load_decisions(controller: str, slice_key: str) -> Optional[List[dict]]:
    if controller == "lstm":
        fname = f"unified_decisions_{slice_key}.json"
    elif controller == "reactive":
        fname = f"reactive_decisions_{slice_key}.json"
    else:
        return None
    return _load_json(os.path.join(ROOT, fname))


# ═════════════════════════════════════════════════════════════════════════════
#  Style helpers
# ═════════════════════════════════════════════════════════════════════════════
COLORS = {
    "static":   "#e74c3c",
    "reactive": "#f39c12",
    "lstm":     "#2ecc71",
}
LABELS = {
    "static":   "Static Allocation",
    "reactive": "Reactive Controller",
    "lstm":     "LSTM (Proactive)",
}
DARK_BG    = "#0a1929"
PANEL_BG   = "#16213e"
GRID_COLOR = "#1a3a5c"
TEXT_COLOR = "white"


def _style_fig(fig):
    fig.patch.set_facecolor(DARK_BG)


def _style_ax(ax, title="", ylabel="", xlabel=""):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=TEXT_COLOR, fontsize=11, fontweight="bold", pad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=9)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.grid(True, alpha=0.15, color=GRID_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def _add_legend(ax, loc="upper right"):
    leg = ax.legend(loc=loc, fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COLOR)
    for t in leg.get_texts():
        t.set_color(TEXT_COLOR)


def _savefig(fig, name: str):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
#  Per-slice metric extraction from stats CSVs
# ═════════════════════════════════════════════════════════════════════════════

def _extract_embb_metrics(df: pd.DataFrame) -> dict:
    """Extract eMBB metrics from a controller stats CSV."""
    m: dict = {}
    if "throughput_mbps" in df.columns:
        tp = pd.to_numeric(df["throughput_mbps"], errors="coerce").dropna()
        m["throughput_mean"]   = tp.mean()
        m["throughput_median"] = tp.median()
        m["throughput_std"]    = tp.std()
        m["throughput_max"]    = tp.max()
        m["throughput_p95"]    = np.percentile(tp, 95) if len(tp) else 0
        m["throughput_p5"]     = np.percentile(tp, 5) if len(tp) else 0
        m["throughput_series"] = tp.values
    if "packet_rate" in df.columns:
        pr = pd.to_numeric(df["packet_rate"], errors="coerce").dropna()
        m["packet_rate_mean"] = pr.mean()
        m["packet_rate_series"] = pr.values
    if "tx_bytes" in df.columns:
        tb = pd.to_numeric(df["tx_bytes"], errors="coerce").dropna()
        m["total_tx_bytes"] = tb.sum()
    m["num_samples"] = len(df)
    return m


def _extract_mmtc_metrics(df: pd.DataFrame) -> dict:
    """Extract mMTC metrics from a controller stats CSV."""
    m: dict = {}
    if "packet_rate" in df.columns:
        pr = pd.to_numeric(df["packet_rate"], errors="coerce").dropna()
        m["packet_rate_mean"]   = pr.mean()
        m["packet_rate_median"] = pr.median()
        m["packet_rate_max"]    = pr.max()
        m["packet_rate_std"]    = pr.std()
        m["packet_rate_series"] = pr.values
    if "throughput_mbps" in df.columns:
        tp = pd.to_numeric(df["throughput_mbps"], errors="coerce").dropna()
        m["throughput_mean"]   = tp.mean()
        m["throughput_series"] = tp.values
    if "dropped_packets" in df.columns:
        dp = pd.to_numeric(df["dropped_packets"], errors="coerce").dropna()
        m["total_dropped"]    = dp.sum()
        m["drop_rate"]        = dp.sum() / max(1, pd.to_numeric(df["packets"], errors="coerce").sum()) * 100
        m["dropped_series"]   = dp.values
    if "retries" in df.columns:
        rt = pd.to_numeric(df["retries"], errors="coerce").dropna()
        m["total_retries"]   = rt.sum()
        m["retries_series"]  = rt.values
    if "active_devices" in df.columns:
        ad = pd.to_numeric(df["active_devices"], errors="coerce").dropna()
        m["active_devices_mean"] = ad.mean()
        m["active_devices_max"]  = ad.max()
    if "burst_active" in df.columns:
        ba = pd.to_numeric(df["burst_active"], errors="coerce").dropna()
        m["burst_ratio"] = ba.mean() * 100 if len(ba) else 0
    m["num_samples"] = len(df)
    return m


def _extract_urllc_metrics(df: pd.DataFrame) -> dict:
    """Extract URLLC metrics from a controller stats CSV (ping data)."""
    m: dict = {}
    if "latency_ms" in df.columns:
        lat = pd.to_numeric(df["latency_ms"], errors="coerce").dropna()
        m["latency_mean"]   = lat.mean()
        m["latency_median"] = lat.median()
        m["latency_std"]    = lat.std()
        m["latency_max"]    = lat.max()
        m["latency_min"]    = lat.min()
        m["latency_p99"]    = np.percentile(lat, 99) if len(lat) else 0
        m["latency_p95"]    = np.percentile(lat, 95) if len(lat) else 0
        m["latency_p50"]    = np.percentile(lat, 50) if len(lat) else 0
        m["latency_series"] = lat.values
        # SLA: URLLC target <1 ms; count violations
        for thresh in [1.0, 2.0, 5.0, 10.0]:
            k = f"sla_violation_{thresh}ms_pct"
            m[k] = (lat > thresh).mean() * 100
    if "lost" in df.columns:
        lost = pd.to_numeric(df["lost"], errors="coerce").dropna()
        m["total_lost"]    = lost.iloc[-1] if len(lost) else 0
        m["loss_series"]   = lost.values
    if "recv" in df.columns:
        recv = pd.to_numeric(df["recv"], errors="coerce").dropna()
        m["total_recv"]    = recv.iloc[-1] if len(recv) else 0
    if "total_lost" in m and "total_recv" in m and m["total_recv"] > 0:
        m["packet_loss_pct"] = m["total_lost"] / (m["total_recv"] + m["total_lost"]) * 100
    if "ip" in df.columns:
        m["num_ues"] = df["ip"].nunique()
    m["num_samples"] = len(df)
    return m


METRIC_EXTRACTORS = {
    "embb":  _extract_embb_metrics,
    "mmtc":  _extract_mmtc_metrics,
    "urllc": _extract_urllc_metrics,
}


# ═════════════════════════════════════════════════════════════════════════════
#  Decision-based metric extraction (from JSON)
# ═════════════════════════════════════════════════════════════════════════════

def _decisions_summary(decisions: List[dict], slice_key: str) -> dict:
    """Summarise a list of decision dicts."""
    if not decisions:
        return {}
    s: dict = {"total_decisions": len(decisions)}
    actions = [d.get("action", "hold") for d in decisions]
    for a in set(actions):
        s[f"action_{a}_count"] = actions.count(a)
        s[f"action_{a}_pct"]   = actions.count(a) / len(actions) * 100

    # Extract allocated bandwidth / capacity timeline
    if slice_key == "embb":
        bw = [d.get("slice_bw_dl_mbps", STATIC_DEFAULTS["embb"]["slice_bw_dl_mbps"]) for d in decisions]
        s["bw_dl_mean"]    = np.mean(bw)
        s["bw_dl_min"]     = np.min(bw)
        s["bw_dl_max"]     = np.max(bw)
        s["bw_dl_series"]  = bw
        trigger = [d.get("trigger_value", 0) for d in decisions]
        s["trigger_series"] = trigger
        s["trigger_mean"]   = np.mean(trigger)
    elif slice_key == "mmtc":
        cap = [d.get("capacity_pps", STATIC_DEFAULTS["mmtc"]["capacity_pps"]) for d in decisions]
        s["capacity_mean"]    = np.mean(cap)
        s["capacity_min"]     = np.min(cap)
        s["capacity_max"]     = np.max(cap)
        s["capacity_series"]  = cap
        bw = [d.get("slice_bw_dl_mbps", STATIC_DEFAULTS["mmtc"]["slice_bw_dl_mbps"]) for d in decisions]
        s["bw_dl_series"]     = bw
        trigger = [d.get("trigger_value", 0) for d in decisions]
        s["trigger_series"]   = trigger
        s["trigger_mean"]     = np.mean(trigger)
    elif slice_key == "urllc":
        lat = [d.get("trigger_latency", 0) for d in decisions]
        s["observed_latency_mean"]   = np.mean(lat)
        s["observed_latency_max"]    = np.max(lat)
        s["observed_latency_series"] = lat
        states = [d.get("state", "NORMAL") for d in decisions]
        s["elevated_pct"] = sum(1 for st in states if st == "ELEVATED") / len(states) * 100

    # timestamps for duration
    ts = []
    for d in decisions:
        raw = d.get("timestamp", "")
        try:
            ts.append(datetime.fromisoformat(raw))
        except Exception:
            pass
    if len(ts) >= 2:
        s["duration_s"] = (ts[-1] - ts[0]).total_seconds()

    return s


# ═════════════════════════════════════════════════════════════════════════════
#  Simulation of static allocation using LSTM controller data
# ═════════════════════════════════════════════════════════════════════════════

def _simulate_static_embb(lstm_df: pd.DataFrame) -> dict:
    """
    Simulate what would happen if eMBB kept the fixed default allocation.
    Uses the traffic from the LSTM stats CSV.
    """
    if "throughput_mbps" not in lstm_df.columns:
        return {}
    traffic = pd.to_numeric(lstm_df["throughput_mbps"], errors="coerce").fillna(0).values
    alloc   = STATIC_DEFAULTS["embb"]["slice_bw_dl_mbps"]
    delivered = np.minimum(traffic, alloc)
    excess    = np.clip(traffic - alloc, 0, None)
    wasted    = np.clip(alloc - traffic, 0, None)
    return {
        "throughput_mean":   delivered.mean(),
        "throughput_max":    delivered.max(),
        "throughput_series": delivered,
        "allocated_bw":     alloc,
        "utilization_pct":  (delivered.sum() / (alloc * len(traffic))) * 100 if len(traffic) else 0,
        "unmet_demand_pct": (excess.sum() / max(1, traffic.sum())) * 100,
        "waste_pct":        (wasted.sum() / (alloc * len(traffic))) * 100 if len(traffic) else 0,
        "sla_violation_pct": (excess > 0).mean() * 100,
        "traffic_series":   traffic,
    }


def _simulate_static_mmtc(lstm_df: pd.DataFrame) -> dict:
    if "packet_rate" not in lstm_df.columns:
        return {}
    pkt_rate = pd.to_numeric(lstm_df["packet_rate"], errors="coerce").fillna(0).values
    cap      = STATIC_DEFAULTS["mmtc"]["capacity_pps"]
    delivered = np.minimum(pkt_rate, cap)
    excess    = np.clip(pkt_rate - cap, 0, None)
    return {
        "packet_rate_mean":    delivered.mean(),
        "packet_rate_max":     delivered.max(),
        "packet_rate_series":  delivered,
        "allocated_capacity":  cap,
        "unmet_demand_pct":    (excess.sum() / max(1, pkt_rate.sum())) * 100,
        "sla_violation_pct":   (excess > 0).mean() * 100,
        "traffic_series":      pkt_rate,
    }


def _simulate_static_urllc() -> dict:
    """Static URLLC: always default QoS, no 5QI/ARP changes."""
    return {
        "5qi":          STATIC_DEFAULTS["urllc"]["5qi"],
        "arp_priority": STATIC_DEFAULTS["urllc"]["arp_priority"],
        "qos_changes":  0,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Simulation of reactive / LSTM allocation over the same traffic
# ═════════════════════════════════════════════════════════════════════════════

def _simulate_controller_embb(decisions: list, traffic: np.ndarray) -> dict:
    """Map controller decisions back over the traffic timeline."""
    n = len(traffic)
    nd = len(decisions)
    alloc = np.full(n, STATIC_DEFAULTS["embb"]["slice_bw_dl_mbps"], dtype=float)
    for i in range(n):
        idx = min(i, nd - 1) if nd else 0
        if nd:
            alloc[i] = decisions[idx].get("slice_bw_dl_mbps",
                                           STATIC_DEFAULTS["embb"]["slice_bw_dl_mbps"])
    delivered = np.minimum(traffic, alloc)
    excess    = np.clip(traffic - alloc, 0, None)
    wasted    = np.clip(alloc - traffic, 0, None)
    return {
        "throughput_mean":    delivered.mean(),
        "delivered_series":   delivered,
        "alloc_series":       alloc,
        "utilization_pct":    (delivered.sum() / max(1, alloc.sum())) * 100,
        "unmet_demand_pct":   (excess.sum() / max(1, traffic.sum())) * 100,
        "waste_pct":          (wasted.sum() / max(1, alloc.sum())) * 100,
        "sla_violation_pct":  (excess > 0).mean() * 100,
    }


def _simulate_controller_mmtc(decisions: list, traffic: np.ndarray) -> dict:
    n = len(traffic)
    nd = len(decisions)
    alloc = np.full(n, STATIC_DEFAULTS["mmtc"]["capacity_pps"], dtype=float)
    for i in range(n):
        idx = min(i, nd - 1) if nd else 0
        if nd:
            alloc[i] = decisions[idx].get("capacity_pps",
                                           STATIC_DEFAULTS["mmtc"]["capacity_pps"])
    delivered = np.minimum(traffic, alloc)
    excess    = np.clip(traffic - alloc, 0, None)
    wasted    = np.clip(alloc - traffic, 0, None)
    return {
        "packet_rate_mean":   delivered.mean(),
        "delivered_series":   delivered,
        "alloc_series":       alloc,
        "unmet_demand_pct":   (excess.sum() / max(1, traffic.sum())) * 100,
        "sla_violation_pct":  (excess > 0).mean() * 100,
        "waste_pct":          (wasted.sum() / max(1, alloc.sum())) * 100,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  VISUALIZATIONS
# ═════════════════════════════════════════════════════════════════════════════

# ── 1.  eMBB: Throughput comparison over time ─────────────────────────────

def plot_embb_throughput_timeseries(lstm_stats, reactive_stats, static_sim):
    """Throughput over time for each controller, overlaid."""
    fig, ax = plt.subplots(figsize=(14, 5))
    _style_fig(fig)
    _style_ax(ax, "eMBB — Throughput Over Time", "Throughput (Mbps)", "Time Step")

    plotted = False
    for label_key, series_key, data in [
        ("lstm",     "throughput_series", lstm_stats),
        ("reactive", "throughput_series", reactive_stats),
        ("static",   "throughput_series", static_sim),
    ]:
        if data and series_key in data:
            ax.plot(data[series_key], label=LABELS[label_key],
                    color=COLORS[label_key], linewidth=0.8, alpha=0.85)
            plotted = True

    if plotted:
        _add_legend(ax)
        _savefig(fig, "embb_throughput_timeseries.png")
    else:
        plt.close(fig)


# ── 2.  eMBB: Allocation vs demand ───────────────────────────────────────

def plot_embb_allocation_vs_demand(lstm_dec, reactive_dec, lstm_df):
    """Show how allocated bandwidth tracks (or misses) traffic demand."""
    if lstm_df is None or "throughput_mbps" not in lstm_df.columns:
        return
    traffic = pd.to_numeric(lstm_df["throughput_mbps"], errors="coerce").fillna(0).values

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    _style_fig(fig)
    fig.suptitle("eMBB — Allocated Bandwidth vs Traffic Demand", color=TEXT_COLOR,
                 fontsize=13, fontweight="bold", y=0.98)

    for i, (ctrl, dec, lbl) in enumerate([
        ("static",   None,         "Static"),
        ("reactive", reactive_dec, "Reactive"),
        ("lstm",     lstm_dec,     "LSTM"),
    ]):
        ax = axes[i]
        _style_ax(ax, f"{lbl} Controller", "Mbps")
        ax.plot(traffic, color="white", linewidth=0.6, alpha=0.7, label="Traffic demand")

        if ctrl == "static":
            alloc_val = STATIC_DEFAULTS["embb"]["slice_bw_dl_mbps"]
            ax.axhline(alloc_val, color=COLORS["static"], linewidth=1.5,
                       linestyle="--", label=f"Allocated ({alloc_val} Mbps)")
        elif dec:
            sim = _simulate_controller_embb(dec, traffic)
            ax.plot(sim["alloc_series"], color=COLORS[ctrl], linewidth=1.2,
                    label="Allocated BW", alpha=0.9)
            ax.fill_between(range(len(traffic)), traffic, sim["alloc_series"],
                            where=traffic > sim["alloc_series"],
                            color="#e74c3c", alpha=0.25, label="Unmet demand")

        _add_legend(ax)

    axes[-1].set_xlabel("Time Step", color=TEXT_COLOR, fontsize=9)
    _savefig(fig, "embb_allocation_vs_demand.png")


# ── 3.  eMBB: Bar chart summary metrics ──────────────────────────────────

def plot_embb_bar_summary(metrics_dict):
    """Bar chart comparing key eMBB metrics across controllers."""
    controllers = [c for c in ["static", "reactive", "lstm"] if c in metrics_dict and metrics_dict[c]]
    if len(controllers) < 2:
        return

    metric_names = []
    for key in ["utilization_pct", "unmet_demand_pct", "waste_pct", "sla_violation_pct"]:
        if all(key in metrics_dict[c] for c in controllers):
            metric_names.append(key)

    if not metric_names:
        return

    nice_names = {
        "utilization_pct":   "Resource Utilization (%)",
        "unmet_demand_pct":  "Unmet Demand (%)",
        "waste_pct":         "Resource Waste (%)",
        "sla_violation_pct": "SLA Violation Rate (%)",
    }

    n_metrics = len(metric_names)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    _style_fig(fig)
    fig.suptitle("eMBB — Controller Performance Summary", color=TEXT_COLOR,
                 fontsize=13, fontweight="bold")

    x = np.arange(len(controllers))
    for i, mk in enumerate(metric_names):
        ax = axes[i]
        _style_ax(ax, nice_names.get(mk, mk))
        vals = [metrics_dict[c].get(mk, 0) for c in controllers]
        bars = ax.bar(x, vals, color=[COLORS[c] for c in controllers], width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[c] for c in controllers], fontsize=8, color=TEXT_COLOR)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.1f}", ha="center", va="bottom", color=TEXT_COLOR, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, "embb_bar_summary.png")


# ── 4.  mMTC: Packet rate comparison ─────────────────────────────────────

def plot_mmtc_packet_rate(lstm_stats, reactive_stats, static_sim):
    fig, ax = plt.subplots(figsize=(14, 5))
    _style_fig(fig)
    _style_ax(ax, "mMTC — Packet Rate Over Time", "Packet Rate (pps)", "Time Step")

    plotted = False
    for lk, data in [("lstm", lstm_stats), ("reactive", reactive_stats), ("static", static_sim)]:
        series = None
        if data:
            series = data.get("packet_rate_series")
        if series is not None and len(series):
            ax.plot(series, label=LABELS[lk], color=COLORS[lk], linewidth=0.8, alpha=0.85)
            plotted = True

    if plotted:
        _add_legend(ax)
        _savefig(fig, "mmtc_packet_rate_timeseries.png")
    else:
        plt.close(fig)


# ── 5.  mMTC: Capacity allocation timeline ───────────────────────────────

def plot_mmtc_capacity_timeline(lstm_dec, reactive_dec, lstm_df):
    if lstm_df is None or "packet_rate" not in lstm_df.columns:
        return
    traffic = pd.to_numeric(lstm_df["packet_rate"], errors="coerce").fillna(0).values

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    _style_fig(fig)
    fig.suptitle("mMTC — Capacity Allocation vs Actual Packet Rate", color=TEXT_COLOR,
                 fontsize=13, fontweight="bold", y=0.98)

    for i, (ctrl, dec, lbl) in enumerate([
        ("static",   None,         "Static"),
        ("reactive", reactive_dec, "Reactive"),
        ("lstm",     lstm_dec,     "LSTM"),
    ]):
        ax = axes[i]
        _style_ax(ax, f"{lbl} Controller", "pps")
        ax.plot(traffic, color="white", linewidth=0.6, alpha=0.7, label="Actual packet rate")

        if ctrl == "static":
            cap = STATIC_DEFAULTS["mmtc"]["capacity_pps"]
            ax.axhline(cap, color=COLORS["static"], linewidth=1.5,
                       linestyle="--", label=f"Capacity ({cap} pps)")
        elif dec:
            sim = _simulate_controller_mmtc(dec, traffic)
            ax.plot(sim["alloc_series"], color=COLORS[ctrl], linewidth=1.2,
                    label="Allocated capacity", alpha=0.9)
            ax.fill_between(range(len(traffic)), traffic, sim["alloc_series"],
                            where=traffic > sim["alloc_series"],
                            color="#e74c3c", alpha=0.25, label="Overflow")

        _add_legend(ax)

    axes[-1].set_xlabel("Time Step", color=TEXT_COLOR, fontsize=9)
    _savefig(fig, "mmtc_capacity_timeline.png")


# ── 6.  mMTC: Bar summary ────────────────────────────────────────────────

def plot_mmtc_bar_summary(metrics_dict):
    controllers = [c for c in ["static", "reactive", "lstm"] if c in metrics_dict and metrics_dict[c]]
    if len(controllers) < 2:
        return

    metric_names = []
    for key in ["unmet_demand_pct", "sla_violation_pct", "waste_pct"]:
        if all(key in metrics_dict[c] for c in controllers):
            metric_names.append(key)

    if not metric_names:
        return

    nice_names = {
        "unmet_demand_pct":  "Unmet Demand (%)",
        "sla_violation_pct": "SLA Violation Rate (%)",
        "waste_pct":         "Resource Waste (%)",
    }

    n = len(metric_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    _style_fig(fig)
    fig.suptitle("mMTC — Controller Performance Summary", color=TEXT_COLOR,
                 fontsize=13, fontweight="bold")

    x = np.arange(len(controllers))
    for i, mk in enumerate(metric_names):
        ax = axes[i]
        _style_ax(ax, nice_names.get(mk, mk))
        vals = [metrics_dict[c].get(mk, 0) for c in controllers]
        bars = ax.bar(x, vals, color=[COLORS[c] for c in controllers], width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[c] for c in controllers], fontsize=8, color=TEXT_COLOR)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", color=TEXT_COLOR, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, "mmtc_bar_summary.png")


# ── 7.  URLLC: Latency comparison (CDF + box) ────────────────────────────

def plot_urllc_latency_cdf(lstm_stats, reactive_stats):
    """CDF of observed latencies under each controller."""
    fig, ax = plt.subplots(figsize=(10, 6))
    _style_fig(fig)
    _style_ax(ax, "URLLC — Latency CDF", "Cumulative Probability", "Latency (ms)")

    plotted = False
    for lk, data in [("lstm", lstm_stats), ("reactive", reactive_stats)]:
        if data and "latency_series" in data:
            sorted_lat = np.sort(data["latency_series"])
            cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
            ax.plot(sorted_lat, cdf, label=LABELS[lk], color=COLORS[lk], linewidth=1.5)
            plotted = True

    if plotted:
        for thresh in [1.0, 2.0, 5.0]:
            ax.axvline(thresh, color="white", linestyle=":", alpha=0.4, linewidth=0.8)
            ax.text(thresh + 0.1, 0.02, f"{thresh}ms", color="white", fontsize=7, alpha=0.6)
        _add_legend(ax)
        _savefig(fig, "urllc_latency_cdf.png")
    else:
        plt.close(fig)


def plot_urllc_latency_boxplot(lstm_stats, reactive_stats):
    """Box plots of latency distributions."""
    data_for_box = []
    labels_for_box = []
    colors_for_box = []
    for lk, data in [("static", None), ("lstm", lstm_stats), ("reactive", reactive_stats)]:
        src = data
        # For static, use LSTM data since QoS doesn't change — latency would be similar
        if lk == "static" and lstm_stats and "latency_series" in lstm_stats:
            src = lstm_stats
        if src and "latency_series" in src:
            data_for_box.append(src["latency_series"])
            labels_for_box.append(LABELS[lk])
            colors_for_box.append(COLORS[lk])

    if len(data_for_box) < 1:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    _style_fig(fig)
    _style_ax(ax, "URLLC — Latency Distribution", "Latency (ms)")

    bp = ax.boxplot(data_for_box, patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", linewidth=1.5))
    for patch, color in zip(bp["boxes"], colors_for_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ["whiskers", "caps"]:
        for item in bp[element]:
            item.set_color("white")
            item.set_alpha(0.6)

    ax.set_xticklabels(labels_for_box, color=TEXT_COLOR, fontsize=9)
    ax.axhline(1.0, color="#e74c3c", linestyle="--", alpha=0.5, linewidth=1, label="1ms target")
    _add_legend(ax)
    _savefig(fig, "urllc_latency_boxplot.png")


# ── 8.  URLLC: SLA violation rates ───────────────────────────────────────

def plot_urllc_sla_violations(lstm_stats, reactive_stats):
    """Bar chart of SLA violation percentages at different thresholds."""
    controllers = {}
    for lk, data in [("lstm", lstm_stats), ("reactive", reactive_stats)]:
        if data:
            violations = {}
            for thresh in [1.0, 2.0, 5.0, 10.0]:
                k = f"sla_violation_{thresh}ms_pct"
                if k in data:
                    violations[f">{thresh}ms"] = data[k]
            if violations:
                controllers[lk] = violations

    if not controllers:
        return

    thresholds = list(list(controllers.values())[0].keys())
    n_thresh = len(thresholds)
    n_ctrl   = len(controllers)

    fig, ax = plt.subplots(figsize=(10, 6))
    _style_fig(fig)
    _style_ax(ax, "URLLC — SLA Violation Rates by Threshold", "Violation Rate (%)")

    x = np.arange(n_thresh)
    width = 0.25
    ctrl_list = list(controllers.keys())
    for j, ctrl in enumerate(ctrl_list):
        vals = [controllers[ctrl].get(t, 0) for t in thresholds]
        offset = (j - (n_ctrl - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=LABELS[ctrl],
                      color=COLORS[ctrl], alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{v:.1f}%", ha="center", va="bottom", color=TEXT_COLOR, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(thresholds, color=TEXT_COLOR, fontsize=9)
    ax.set_xlabel("Latency Threshold", color=TEXT_COLOR, fontsize=9)
    _add_legend(ax)
    _savefig(fig, "urllc_sla_violations.png")


# ── 9.  URLLC: Per-UE latency heatmap ────────────────────────────────────

def plot_urllc_per_ue_latency(lstm_stats_df, title_suffix="LSTM"):
    """Heatmap of latency per UE over time (sequence number)."""
    if lstm_stats_df is None or "ip" not in lstm_stats_df.columns:
        return
    df = lstm_stats_df.copy()
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    df["seq"] = pd.to_numeric(df["seq"], errors="coerce")
    df.dropna(subset=["latency_ms", "seq"], inplace=True)

    ues = sorted(df["ip"].unique())
    if not ues:
        return

    fig, axes = plt.subplots(len(ues), 1, figsize=(14, 2.5 * len(ues)), sharex=True)
    if len(ues) == 1:
        axes = [axes]
    _style_fig(fig)
    fig.suptitle(f"URLLC — Per-UE Latency Timeline ({title_suffix})", color=TEXT_COLOR,
                 fontsize=13, fontweight="bold", y=0.99)

    for i, ue in enumerate(ues):
        ax = axes[i]
        _style_ax(ax, f"UE {ue}", "Latency (ms)")
        ue_data = df[df["ip"] == ue].sort_values("seq")
        ax.plot(ue_data["seq"].values, ue_data["latency_ms"].values,
                color=COLORS["lstm"], linewidth=0.5, alpha=0.8)
        ax.axhline(1.0, color="#e74c3c", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.axhline(2.0, color="#f39c12", linestyle="--", alpha=0.4, linewidth=0.8)

    axes[-1].set_xlabel("Ping Sequence", color=TEXT_COLOR, fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig(fig, "urllc_per_ue_latency.png")


# ── 10. Decision action distribution (pie / bar) ─────────────────────────

def plot_decision_distribution(all_dec_summaries):
    """Side-by-side action distributions for reactive vs LSTM."""
    slices_with_data = []
    for sk in ["embb", "mmtc", "urllc"]:
        has_data = False
        for ctrl in ["reactive", "lstm"]:
            if (sk, ctrl) in all_dec_summaries and all_dec_summaries[(sk, ctrl)]:
                has_data = True
        if has_data:
            slices_with_data.append(sk)

    if not slices_with_data:
        return

    n_slices = len(slices_with_data)
    fig, axes = plt.subplots(n_slices, 2, figsize=(12, 4 * n_slices))
    if n_slices == 1:
        axes = axes.reshape(1, -1)
    _style_fig(fig)
    fig.suptitle("Controller Decision Distribution — Reactive vs LSTM",
                 color=TEXT_COLOR, fontsize=13, fontweight="bold", y=0.99)

    action_colors = {
        "expand":   "#2ecc71",
        "contract": "#e74c3c",
        "hold":     "#3498db",
        "elevate":  "#f39c12",
        "relax":    "#1abc9c",
    }

    for row, sk in enumerate(slices_with_data):
        for col, ctrl in enumerate(["reactive", "lstm"]):
            ax = axes[row, col]
            _style_ax(ax, f"{sk.upper()} — {LABELS[ctrl]}")
            summary = all_dec_summaries.get((sk, ctrl), {})
            if not summary or summary.get("total_decisions", 0) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        color=TEXT_COLOR, fontsize=12, transform=ax.transAxes)
                continue

            action_keys = [k for k in summary if k.startswith("action_") and k.endswith("_count")]
            if not action_keys:
                continue

            action_names  = [k.replace("action_", "").replace("_count", "") for k in action_keys]
            action_counts = [summary[k] for k in action_keys]
            colors_pie    = [action_colors.get(a, "#95a5a6") for a in action_names]

            wedges, texts, autotexts = ax.pie(
                action_counts, labels=action_names, colors=colors_pie,
                autopct="%1.1f%%", textprops={"color": TEXT_COLOR, "fontsize": 8},
                startangle=90
            )
            for t in autotexts:
                t.set_fontsize(7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, "decision_distribution.png")


# ── 11. Decision bandwidth/capacity timeline ─────────────────────────────

def plot_decision_bw_timeline(all_dec_summaries):
    """Plot allocated bandwidth / capacity over decision index for each slice."""
    fig_parts = []

    # eMBB BW timeline
    embb_data = []
    for ctrl in ["reactive", "lstm"]:
        s = all_dec_summaries.get(("embb", ctrl), {})
        if s and "bw_dl_series" in s:
            embb_data.append((ctrl, s["bw_dl_series"]))

    if embb_data:
        fig, ax = plt.subplots(figsize=(14, 5))
        _style_fig(fig)
        _style_ax(ax, "eMBB — Allocated DL Bandwidth Over Decisions",
                   "Bandwidth (Mbps)", "Decision Index")
        ax.axhline(STATIC_DEFAULTS["embb"]["slice_bw_dl_mbps"], color=COLORS["static"],
                   linestyle="--", linewidth=1.5, alpha=0.7,
                   label=f"Static ({STATIC_DEFAULTS['embb']['slice_bw_dl_mbps']} Mbps)")
        for ctrl, series in embb_data:
            ax.plot(series, label=LABELS[ctrl], color=COLORS[ctrl], linewidth=1.2)
        _add_legend(ax)
        _savefig(fig, "embb_decision_bw_timeline.png")

    # mMTC capacity timeline
    mmtc_data = []
    for ctrl in ["reactive", "lstm"]:
        s = all_dec_summaries.get(("mmtc", ctrl), {})
        if s and "capacity_series" in s:
            mmtc_data.append((ctrl, s["capacity_series"]))

    if mmtc_data:
        fig, ax = plt.subplots(figsize=(14, 5))
        _style_fig(fig)
        _style_ax(ax, "mMTC — Allocated Capacity Over Decisions",
                   "Capacity (pps)", "Decision Index")
        ax.axhline(STATIC_DEFAULTS["mmtc"]["capacity_pps"], color=COLORS["static"],
                   linestyle="--", linewidth=1.5, alpha=0.7,
                   label=f"Static ({STATIC_DEFAULTS['mmtc']['capacity_pps']} pps)")
        for ctrl, series in mmtc_data:
            ax.plot(series, label=LABELS[ctrl], color=COLORS[ctrl], linewidth=1.2)
        _add_legend(ax)
        _savefig(fig, "mmtc_decision_capacity_timeline.png")


# ── 12.  URLLC: Decision latency timeline ────────────────────────────────

def plot_urllc_decision_latency(all_dec_summaries):
    """Observed latency over decision steps for URLLC."""
    fig, ax = plt.subplots(figsize=(14, 5))
    _style_fig(fig)
    _style_ax(ax, "URLLC — Observed Latency per Controller Decision",
               "Latency (ms)", "Decision Index")

    plotted = False
    for ctrl in ["reactive", "lstm"]:
        s = all_dec_summaries.get(("urllc", ctrl), {})
        if s and "observed_latency_series" in s:
            ax.plot(s["observed_latency_series"], label=LABELS[ctrl],
                    color=COLORS[ctrl], linewidth=0.8, alpha=0.85)
            plotted = True

    if plotted:
        ax.axhline(2.0, color="#e74c3c", linestyle="--", alpha=0.5, label="2ms threshold")
        ax.axhline(1.0, color="#f39c12", linestyle="--", alpha=0.5, label="1ms threshold")
        _add_legend(ax)
        _savefig(fig, "urllc_decision_latency_timeline.png")
    else:
        plt.close(fig)


# ── 13.  Cross-slice radar chart ──────────────────────────────────────────

def plot_radar_chart(all_embb, all_mmtc, all_urllc):
    """Radar chart summarising normalised performance across slices."""
    # Collect metrics for radar
    controllers_available = []
    for c in ["static", "reactive", "lstm"]:
        if ((c in all_embb and all_embb[c]) or
            (c in all_mmtc and all_mmtc[c]) or
            (c in all_urllc and all_urllc[c])):
            controllers_available.append(c)

    if len(controllers_available) < 2:
        return

    # Build radar dimensions (lower is better for all except utilization)
    dimensions = []
    vals_by_ctrl: Dict[str, list] = {c: [] for c in controllers_available}

    # eMBB SLA violations
    for c in controllers_available:
        v = all_embb.get(c, {}).get("sla_violation_pct", 0)
        vals_by_ctrl[c].append(v)
    dimensions.append("eMBB SLA\nViolation %")

    # eMBB waste
    for c in controllers_available:
        v = all_embb.get(c, {}).get("waste_pct", 0)
        vals_by_ctrl[c].append(v)
    dimensions.append("eMBB Resource\nWaste %")

    # mMTC SLA violations
    for c in controllers_available:
        v = all_mmtc.get(c, {}).get("sla_violation_pct", 0)
        vals_by_ctrl[c].append(v)
    dimensions.append("mMTC SLA\nViolation %")

    # URLLC latency (mean, from stats or decisions)
    for c in controllers_available:
        lat = all_urllc.get(c, {}).get("latency_mean", 0)
        vals_by_ctrl[c].append(lat)
    dimensions.append("URLLC Mean\nLatency (ms)")

    # URLLC >2ms violation
    for c in controllers_available:
        v = all_urllc.get(c, {}).get("sla_violation_2.0ms_pct", 0)
        vals_by_ctrl[c].append(v)
    dimensions.append("URLLC >2ms\nViolation %")

    n_dims = len(dimensions)
    if n_dims < 3:
        return

    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    _style_fig(fig)
    ax.set_facecolor(PANEL_BG)
    ax.set_title("Cross-Slice Performance Radar\n(lower is better)",
                 color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=20)

    # Normalise each dimension to [0, 1]
    for i in range(n_dims):
        col_vals = [vals_by_ctrl[c][i] for c in controllers_available]
        max_val = max(col_vals) if max(col_vals) > 0 else 1
        for c in controllers_available:
            vals_by_ctrl[c][i] = vals_by_ctrl[c][i] / max_val

    for ctrl in controllers_available:
        values = vals_by_ctrl[ctrl] + vals_by_ctrl[ctrl][:1]
        ax.plot(angles, values, color=COLORS[ctrl], linewidth=2, label=LABELS[ctrl])
        ax.fill(angles, values, color=COLORS[ctrl], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, color=TEXT_COLOR, fontsize=8)
    ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax.grid(color=GRID_COLOR, alpha=0.3)
    ax.spines["polar"].set_color(GRID_COLOR)
    leg = ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
                    fontsize=9, facecolor=PANEL_BG, edgecolor=GRID_COLOR)
    for t in leg.get_texts():
        t.set_color(TEXT_COLOR)

    _savefig(fig, "cross_slice_radar.png")


# ── 14.  Combined summary table (text image) ─────────────────────────────

def plot_summary_table(all_embb, all_mmtc, all_urllc, all_dec_summaries):
    """Create a table image summarising key numbers."""
    rows = []
    # eMBB rows
    for c in ["static", "reactive", "lstm"]:
        m = all_embb.get(c, {})
        if not m:
            continue
        rows.append([
            f"eMBB / {LABELS[c]}",
            f"{m.get('throughput_mean', 0):.1f}",
            f"{m.get('utilization_pct', 0):.1f}%",
            f"{m.get('sla_violation_pct', 0):.1f}%",
            f"{m.get('waste_pct', 0):.1f}%",
            f"{m.get('unmet_demand_pct', 0):.1f}%",
        ])
    # mMTC rows
    for c in ["static", "reactive", "lstm"]:
        m = all_mmtc.get(c, {})
        if not m:
            continue
        rows.append([
            f"mMTC / {LABELS[c]}",
            f"{m.get('packet_rate_mean', 0):.1f}",
            "—",
            f"{m.get('sla_violation_pct', 0):.1f}%",
            f"{m.get('waste_pct', 0):.1f}%",
            f"{m.get('unmet_demand_pct', 0):.1f}%",
        ])
    # URLLC rows
    for c in ["static", "reactive", "lstm"]:
        m = all_urllc.get(c, {})
        if not m:
            continue
        rows.append([
            f"URLLC / {LABELS[c]}",
            f"{m.get('latency_mean', 0):.2f} ms",
            "—",
            f"{m.get('sla_violation_2.0ms_pct', 0):.1f}%",
            "—",
            "—",
        ])

    if not rows:
        return

    col_labels = ["Slice / Controller", "Avg Metric", "Utilization", "SLA Violation",
                  "Waste", "Unmet Demand"]

    fig, ax = plt.subplots(figsize=(14, 0.6 * len(rows) + 2))
    _style_fig(fig)
    ax.axis("off")
    ax.set_title("Cross-Controller Performance Summary", color=TEXT_COLOR,
                 fontsize=14, fontweight="bold", pad=15)

    table = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    for (r, c_idx), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        if r == 0:
            cell.set_facecolor("#1a3a5c")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor(PANEL_BG)
            cell.set_text_props(color="white")

    _savefig(fig, "summary_table.png")


# ── 15.  Reaction time / decision speed comparison ───────────────────────

def plot_decision_reaction_analysis(all_dec_summaries):
    """Bar chart: total decisions, expand/contract ratio, and duration."""
    slices = ["embb", "mmtc", "urllc"]
    ctrls  = ["reactive", "lstm"]

    # Total decisions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    _style_fig(fig)
    fig.suptitle("Decision Volume & Action Breakdown", color=TEXT_COLOR,
                 fontsize=13, fontweight="bold")

    for i, sk in enumerate(slices):
        ax = axes[i]
        _style_ax(ax, sk.upper())
        x_pos = 0
        tick_labels = []
        tick_positions = []
        for ctrl in ctrls:
            s = all_dec_summaries.get((sk, ctrl), {})
            if not s or s.get("total_decisions", 0) == 0:
                continue
            total = s["total_decisions"]
            actions = {}
            for k in s:
                if k.startswith("action_") and k.endswith("_count"):
                    a_name = k.replace("action_", "").replace("_count", "")
                    actions[a_name] = s[k]

            bottom = 0
            action_colors = {
                "expand": "#2ecc71", "contract": "#e74c3c", "hold": "#3498db",
                "elevate": "#f39c12", "relax": "#1abc9c",
            }
            for a_name, count in sorted(actions.items()):
                ax.bar(x_pos, count, bottom=bottom, width=0.6,
                       color=action_colors.get(a_name, "#95a5a6"),
                       label=a_name if i == 0 and ctrl == ctrls[0] else "")
                bottom += count

            tick_labels.append(LABELS[ctrl])
            tick_positions.append(x_pos)
            x_pos += 1

        if tick_positions:
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=7, color=TEXT_COLOR, rotation=15)
        ax.set_ylabel("Number of Decisions", color=TEXT_COLOR, fontsize=8)

    # shared legend from first axes
    handles, labels = [], []
    for a, c in [("expand", "#2ecc71"), ("contract", "#e74c3c"), ("hold", "#3498db"),
                 ("elevate", "#f39c12"), ("relax", "#1abc9c")]:
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=c))
        labels.append(a)
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8,
               facecolor=PANEL_BG, edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR)

    plt.tight_layout(rect=[0, 0.06, 1, 0.93])
    _savefig(fig, "decision_volume_breakdown.png")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Cross-Controller Performance Comparison")
    print("  Static  ·  Reactive  ·  LSTM (Proactive)")
    print("=" * 70)

    # ── Load all data ─────────────────────────────────────────────────────
    print("\n▸ Loading data …")

    # Stats CSVs
    stats = {}
    for sk in ["embb", "mmtc", "urllc"]:
        for ctrl in ["lstm", "reactive", "static"]:
            df = _find_stats_csv(sk, ctrl)
            if df is not None:
                # For mMTC, only keep the 14-column portion
                if sk == "mmtc" and ctrl == "lstm":
                    expected_cols = {"timestamp", "packets", "bytes", "throughput_mbps",
                                     "packet_rate", "active_devices", "dropped_packets",
                                     "retries", "send_errors", "burst_active",
                                     "temp_sensors", "meter_sensors", "alarm_sensors",
                                     "tracker_sensors"}
                    if expected_cols.issubset(set(df.columns)):
                        # keep only rows where all expected columns are numeric
                        for col in expected_cols - {"timestamp"}:
                            df[col] = pd.to_numeric(df[col], errors="coerce")
                        df.dropna(subset=["packet_rate"], inplace=True)

                print(f"    [{sk}/{ctrl}] stats: {len(df)} rows")
            stats[(sk, ctrl)] = df

    # Decisions
    decisions = {}
    dec_summaries = {}
    for sk in ["embb", "mmtc", "urllc"]:
        for ctrl in ["reactive", "lstm"]:
            dec = _load_decisions(ctrl, sk)
            decisions[(sk, ctrl)] = dec
            if dec:
                dec_summaries[(sk, ctrl)] = _decisions_summary(dec, sk)
                print(f"    [{sk}/{ctrl}] decisions: {len(dec)} entries")
            else:
                dec_summaries[(sk, ctrl)] = {}

    # ── Compute metrics ──────────────────────────────────────────────────
    print("\n▸ Computing metrics …")

    # --- eMBB ---
    all_embb: Dict[str, dict] = {}

    lstm_embb_df = stats.get(("embb", "lstm"))
    if lstm_embb_df is not None:
        all_embb["lstm_raw"] = _extract_embb_metrics(lstm_embb_df)
        traffic_embb = pd.to_numeric(lstm_embb_df["throughput_mbps"], errors="coerce").fillna(0).values
    else:
        traffic_embb = None

    reactive_embb_df = stats.get(("embb", "reactive"))
    if reactive_embb_df is not None:
        all_embb["reactive_raw"] = _extract_embb_metrics(reactive_embb_df)

    # Simulate over common traffic
    if traffic_embb is not None:
        all_embb["static"] = _simulate_static_embb(lstm_embb_df)
        lstm_dec_embb = decisions.get(("embb", "lstm"))
        if lstm_dec_embb:
            all_embb["lstm"] = _simulate_controller_embb(lstm_dec_embb, traffic_embb)
            all_embb["lstm"]["throughput_mean"] = all_embb.get("lstm_raw", {}).get("throughput_mean", 0)
            all_embb["lstm"]["throughput_series"] = traffic_embb

        reactive_dec_embb = decisions.get(("embb", "reactive"))
        if reactive_dec_embb:
            all_embb["reactive"] = _simulate_controller_embb(reactive_dec_embb, traffic_embb)
            all_embb["reactive"]["throughput_series"] = traffic_embb
        elif reactive_embb_df is not None:
            # Use raw reactive stats if decisions are missing
            all_embb["reactive"] = all_embb.get("reactive_raw", {})

    # --- mMTC ---
    all_mmtc: Dict[str, dict] = {}

    lstm_mmtc_df = stats.get(("mmtc", "lstm"))
    if lstm_mmtc_df is not None:
        all_mmtc["lstm_raw"] = _extract_mmtc_metrics(lstm_mmtc_df)
        if "packet_rate" in lstm_mmtc_df.columns:
            traffic_mmtc = pd.to_numeric(lstm_mmtc_df["packet_rate"], errors="coerce").fillna(0).values
        else:
            traffic_mmtc = None
    else:
        traffic_mmtc = None

    reactive_mmtc_df = stats.get(("mmtc", "reactive"))
    if reactive_mmtc_df is not None:
        all_mmtc["reactive_raw"] = _extract_mmtc_metrics(reactive_mmtc_df)

    if traffic_mmtc is not None:
        all_mmtc["static"] = _simulate_static_mmtc(lstm_mmtc_df)
        lstm_dec_mmtc = decisions.get(("mmtc", "lstm"))
        if lstm_dec_mmtc:
            all_mmtc["lstm"] = _simulate_controller_mmtc(lstm_dec_mmtc, traffic_mmtc)
            all_mmtc["lstm"]["packet_rate_series"] = traffic_mmtc

        reactive_dec_mmtc = decisions.get(("mmtc", "reactive"))
        if reactive_dec_mmtc:
            all_mmtc["reactive"] = _simulate_controller_mmtc(reactive_dec_mmtc, traffic_mmtc)
            all_mmtc["reactive"]["packet_rate_series"] = traffic_mmtc

    # --- URLLC ---
    all_urllc: Dict[str, dict] = {}

    lstm_urllc_df = stats.get(("urllc", "lstm"))
    if lstm_urllc_df is not None:
        all_urllc["lstm"] = _extract_urllc_metrics(lstm_urllc_df)
        print(f"    [urllc/lstm] mean latency = {all_urllc['lstm'].get('latency_mean', 0):.2f} ms")

    reactive_urllc_df = stats.get(("urllc", "reactive"))
    if reactive_urllc_df is not None:
        all_urllc["reactive"] = _extract_urllc_metrics(reactive_urllc_df)

    # Static URLLC: same as LSTM data since no QoS change occurs
    if "lstm" in all_urllc:
        all_urllc["static"] = dict(all_urllc["lstm"])

    # ── Generate visualizations ───────────────────────────────────────────
    print(f"\n▸ Generating visualisations → {OUT}/")

    # eMBB
    plot_embb_throughput_timeseries(
        all_embb.get("lstm_raw") or all_embb.get("lstm"),
        all_embb.get("reactive_raw") or all_embb.get("reactive"),
        all_embb.get("static"),
    )
    plot_embb_allocation_vs_demand(
        decisions.get(("embb", "lstm")),
        decisions.get(("embb", "reactive")),
        lstm_embb_df,
    )
    plot_embb_bar_summary(all_embb)

    # mMTC
    plot_mmtc_packet_rate(
        all_mmtc.get("lstm_raw") or all_mmtc.get("lstm"),
        all_mmtc.get("reactive_raw") or all_mmtc.get("reactive"),
        all_mmtc.get("static"),
    )
    plot_mmtc_capacity_timeline(
        decisions.get(("mmtc", "lstm")),
        decisions.get(("mmtc", "reactive")),
        lstm_mmtc_df,
    )
    plot_mmtc_bar_summary(all_mmtc)

    # URLLC
    plot_urllc_latency_cdf(all_urllc.get("lstm"), all_urllc.get("reactive"))
    plot_urllc_latency_boxplot(all_urllc.get("lstm"), all_urllc.get("reactive"))
    plot_urllc_sla_violations(all_urllc.get("lstm"), all_urllc.get("reactive"))
    plot_urllc_per_ue_latency(lstm_urllc_df, "LSTM")
    if reactive_urllc_df is not None:
        plot_urllc_per_ue_latency(reactive_urllc_df, "Reactive")

    # Cross-controller
    plot_decision_distribution(dec_summaries)
    plot_decision_bw_timeline(dec_summaries)
    plot_urllc_decision_latency(dec_summaries)
    plot_radar_chart(all_embb, all_mmtc, all_urllc)
    plot_summary_table(all_embb, all_mmtc, all_urllc, dec_summaries)
    plot_decision_reaction_analysis(dec_summaries)

    # ── Print console summary ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CONSOLE SUMMARY")
    print("=" * 70)

    for sk, label, metrics in [("embb", "eMBB", all_embb),
                                ("mmtc", "mMTC", all_mmtc),
                                ("urllc", "URLLC", all_urllc)]:
        print(f"\n  ── {label} {'─' * (50 - len(label))}")
        for ctrl in ["static", "reactive", "lstm"]:
            m = metrics.get(ctrl, {})
            if not m:
                print(f"    {LABELS[ctrl]:<25s}  (no data)")
                continue
            parts = []
            if "throughput_mean" in m:
                parts.append(f"Avg Tput={m['throughput_mean']:.1f} Mbps")
            if "packet_rate_mean" in m:
                parts.append(f"Avg PktRate={m['packet_rate_mean']:.1f} pps")
            if "latency_mean" in m:
                parts.append(f"Avg Lat={m['latency_mean']:.2f} ms")
            if "utilization_pct" in m:
                parts.append(f"Util={m['utilization_pct']:.1f}%")
            if "sla_violation_pct" in m:
                parts.append(f"SLA Viol={m['sla_violation_pct']:.1f}%")
            if "waste_pct" in m:
                parts.append(f"Waste={m['waste_pct']:.1f}%")
            if "unmet_demand_pct" in m:
                parts.append(f"Unmet={m['unmet_demand_pct']:.1f}%")
            if "sla_violation_2.0ms_pct" in m:
                parts.append(f">2ms={m['sla_violation_2.0ms_pct']:.1f}%")
            print(f"    {LABELS[ctrl]:<25s}  {' | '.join(parts)}")

    # Decision summaries
    print(f"\n  ── Decision Counts {'─' * 40}")
    for sk in ["embb", "mmtc", "urllc"]:
        for ctrl in ["reactive", "lstm"]:
            s = dec_summaries.get((sk, ctrl), {})
            if s and s.get("total_decisions", 0) > 0:
                actions_str = ", ".join(
                    f"{k.replace('action_', '').replace('_count', '')}="
                    f"{s[k]}"
                    for k in sorted(s) if k.startswith("action_") and k.endswith("_count")
                )
                dur = f", dur={s['duration_s']:.0f}s" if "duration_s" in s else ""
                print(f"    [{sk.upper()}/{LABELS[ctrl]}] "
                      f"total={s['total_decisions']} | {actions_str}{dur}")

    # ── Save numeric results as JSON ──────────────────────────────────────
    def _clean(d):
        """Remove numpy arrays before JSON serialisation."""
        return {k: v for k, v in d.items() if not isinstance(v, np.ndarray) and not isinstance(v, list)}

    results = {
        "embb":  {c: _clean(all_embb.get(c, {})) for c in ["static", "reactive", "lstm"]},
        "mmtc":  {c: _clean(all_mmtc.get(c, {})) for c in ["static", "reactive", "lstm"]},
        "urllc": {c: _clean(all_urllc.get(c, {})) for c in ["static", "reactive", "lstm"]},
    }
    results_path = os.path.join(OUT, "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  ✓ Numeric results saved → {results_path}")
    print(f"  ✓ All visualisations saved → {OUT}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
