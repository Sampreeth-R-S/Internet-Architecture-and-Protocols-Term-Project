"""
Baseline Comparison: Static vs Reactive vs Proactive (LSTM) mMTC Slicing.
Evaluates connection capacity utilization, packet delivery, and device density handling.

mMTC-specific metrics:
  - Packet delivery ratio (PDR): fraction of sensor packets successfully delivered
  - Connection density utilization: how well we match capacity to device count
  - Resource waste: over-provisioned capacity
  - Congestion events: when demand exceeds capacity

Usage:
    python3 compare_baselines.py
"""
import numpy as np
import pandas as pd
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
VIZ_DIR = os.path.join(BASE_DIR, '..', 'visualization')

# mMTC capacity levels (packets per second)
STATIC_CAPACITY = 300         # Fixed allocation
EXPANDED_CAPACITY = 800       # After expansion
CONTRACTED_CAPACITY = 200     # After contraction
NORMAL_CAPACITY = 500         # Default

# SLA: 95% packet delivery ratio required
SLA_PDR_THRESHOLD = 0.95


def simulate_static(traffic):
    """Static allocation: fixed capacity, no adjustment."""
    allocated = np.full(len(traffic), STATIC_CAPACITY, dtype=float)
    delivered = np.minimum(traffic, allocated)
    return allocated, delivered


def simulate_reactive(traffic, react_delay=6):
    """Reactive: adjusts AFTER congestion detected (with delay)."""
    allocated = np.full(len(traffic), NORMAL_CAPACITY, dtype=float)
    current_cap = NORMAL_CAPACITY

    for i in range(len(traffic)):
        if i >= react_delay:
            past = traffic[i - react_delay]
            if past > 0.7 * current_cap:
                current_cap = EXPANDED_CAPACITY
            elif past < 0.3 * current_cap:
                current_cap = CONTRACTED_CAPACITY
            else:
                current_cap = NORMAL_CAPACITY
        allocated[i] = current_cap

    delivered = np.minimum(traffic, allocated)
    return allocated, delivered


def simulate_proactive(traffic, decisions_file=None):
    """Proactive LSTM: adjusts BEFORE congestion using predictions."""
    allocated = np.full(len(traffic), NORMAL_CAPACITY, dtype=float)

    if decisions_file and os.path.exists(decisions_file):
        with open(decisions_file, 'r') as f:
            decisions = json.load(f)

        current_cap = NORMAL_CAPACITY
        decision_idx = 0
        for i in range(len(traffic)):
            if decision_idx < len(decisions):
                action = decisions[decision_idx]['action']
                if action == 'expand':
                    current_cap = EXPANDED_CAPACITY
                elif action == 'contract':
                    current_cap = CONTRACTED_CAPACITY
                decision_idx += 1
            allocated[i] = current_cap
    else:
        # Simulate proactive behavior with moving average
        window = 24
        current_cap = NORMAL_CAPACITY
        for i in range(window, len(traffic)):
            predicted_peak = np.max(traffic[i-window:i]) * 1.1
            predicted_avg = np.mean(traffic[i-window:i])

            if predicted_peak > 0.7 * NORMAL_CAPACITY:
                current_cap = EXPANDED_CAPACITY
            elif predicted_avg < 0.3 * NORMAL_CAPACITY:
                current_cap = CONTRACTED_CAPACITY
            else:
                current_cap = NORMAL_CAPACITY
            allocated[i] = current_cap

    delivered = np.minimum(traffic, allocated)
    return allocated, delivered


def compute_metrics(traffic, allocated, delivered):
    """Compute mMTC-specific performance metrics."""
    # Packet delivery ratio
    pdr = np.mean(delivered / np.maximum(traffic, 1)) * 100

    # Capacity utilization
    utilization = np.mean(delivered / allocated) * 100

    # SLA violations: PDR < threshold
    per_step_pdr = delivered / np.maximum(traffic, 1)
    sla_violations = np.sum(per_step_pdr < SLA_PDR_THRESHOLD)
    sla_violation_rate = (sla_violations / len(traffic)) * 100

    # Congestion events: when demand exceeds capacity
    congestion_events = np.sum(traffic > allocated)
    congestion_rate = (congestion_events / len(traffic)) * 100

    # Packet loss
    excess = np.clip(traffic - allocated, 0, None)
    packet_loss = np.sum(excess) / np.sum(traffic) * 100

    # Wasted capacity
    wasted = np.clip(allocated - traffic, 0, None)
    waste_pct = np.sum(wasted) / np.sum(allocated) * 100

    return {
        'pdr_pct': round(pdr, 2),
        'utilization_pct': round(utilization, 2),
        'sla_violations': int(sla_violations),
        'sla_violation_rate_pct': round(sla_violation_rate, 2),
        'congestion_events': int(congestion_events),
        'congestion_rate_pct': round(congestion_rate, 2),
        'packet_loss_pct': round(packet_loss, 2),
        'resource_waste_pct': round(waste_pct, 2)
    }


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, 'mmtc_traffic_timeseries.csv'))
    traffic = df['packet_rate'].values.astype(float)

    decisions_file = os.path.join(DATA_DIR, 'controller_decisions.json')

    print("=" * 65)
    print("  mMTC Baseline Comparison: Static vs Reactive vs Proactive")
    print("=" * 65)

    static_alloc, static_deliv = simulate_static(traffic)
    reactive_alloc, reactive_deliv = simulate_reactive(traffic)
    proactive_alloc, proactive_deliv = simulate_proactive(traffic, decisions_file)

    static_metrics = compute_metrics(traffic, static_alloc, static_deliv)
    reactive_metrics = compute_metrics(traffic, reactive_alloc, reactive_deliv)
    proactive_metrics = compute_metrics(traffic, proactive_alloc, proactive_deliv)

    print(f"\n{'Metric':<30} {'Static':>10} {'Reactive':>10} {'Proactive':>10}")
    print("-" * 62)
    for key in static_metrics:
        print(f"{key:<30} {static_metrics[key]:>10} {reactive_metrics[key]:>10} {proactive_metrics[key]:>10}")

    results = {
        'static': static_metrics,
        'reactive': reactive_metrics,
        'proactive': proactive_metrics
    }
    results_path = os.path.join(DATA_DIR, 'baseline_comparison.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {results_path}")

    # Plot
    os.makedirs(VIZ_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('mMTC Baseline Comparison: Static vs Reactive vs Proactive', fontsize=14)

    labels = ['Static', 'Reactive', 'Proactive']
    colors = ['#e74c3c', '#f39c12', '#2ecc71']

    # 1. SLA Violation Rate
    vals = [static_metrics['sla_violation_rate_pct'],
            reactive_metrics['sla_violation_rate_pct'],
            proactive_metrics['sla_violation_rate_pct']]
    axes[0, 0].bar(labels, vals, color=colors)
    axes[0, 0].set_title('SLA Violation Rate (%)')
    axes[0, 0].set_ylabel('%')

    # 2. Packet Delivery Ratio
    vals = [static_metrics['pdr_pct'],
            reactive_metrics['pdr_pct'],
            proactive_metrics['pdr_pct']]
    axes[0, 1].bar(labels, vals, color=colors)
    axes[0, 1].set_title('Packet Delivery Ratio (%)')
    axes[0, 1].set_ylabel('%')

    # 3. Congestion Rate
    vals = [static_metrics['congestion_rate_pct'],
            reactive_metrics['congestion_rate_pct'],
            proactive_metrics['congestion_rate_pct']]
    axes[1, 0].bar(labels, vals, color=colors)
    axes[1, 0].set_title('Congestion Rate (%)')
    axes[1, 0].set_ylabel('%')

    # 4. Resource Waste
    vals = [static_metrics['resource_waste_pct'],
            reactive_metrics['resource_waste_pct'],
            proactive_metrics['resource_waste_pct']]
    axes[1, 1].bar(labels, vals, color=colors)
    axes[1, 1].set_title('Resource Waste (%)')
    axes[1, 1].set_ylabel('%')

    plt.tight_layout()
    plot_path = os.path.join(VIZ_DIR, 'baseline_comparison.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved → {plot_path}")


if __name__ == '__main__':
    main()
