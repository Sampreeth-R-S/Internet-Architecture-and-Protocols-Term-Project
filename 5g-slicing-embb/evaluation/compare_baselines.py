"""
Baseline Comparison: Static vs Reactive vs Proactive (LSTM) Slicing.
Evaluates throughput utilization, SLA violations, and latency proxies.

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


STATIC_BW = 60       # Fixed allocation
EXPANDED_BW = 200    # After expansion
CONTRACTED_BW = 50   # After contraction
NORMAL_BW = 100      # Default

# SLA threshold
SLA_THRESHOLD = 0.90  # Demand must be met at 90%+ to avoid violation


def simulate_static(traffic):
    """Static allocation: fixed bandwidth, no adjustment."""
    allocated = np.full(len(traffic), STATIC_BW, dtype=float)
    delivered = np.minimum(traffic, allocated)
    return allocated, delivered


def simulate_reactive(traffic, react_delay=6):
    """Reactive: adjusts AFTER congestion is detected (with delay)."""
    allocated = np.full(len(traffic), NORMAL_BW, dtype=float)
    current_bw = NORMAL_BW

    for i in range(len(traffic)):
        # React to congestion detected `react_delay` steps ago
        if i >= react_delay:
            past = traffic[i - react_delay]
            if past > 0.7 * current_bw:
                current_bw = EXPANDED_BW
            elif past < 0.3 * current_bw:
                current_bw = CONTRACTED_BW
            else:
                current_bw = NORMAL_BW
        allocated[i] = current_bw

    delivered = np.minimum(traffic, allocated)
    return allocated, delivered


def simulate_proactive(traffic, decisions_file=None):
    """Proactive LSTM: adjusts BEFORE congestion using predictions."""
    allocated = np.full(len(traffic), NORMAL_BW, dtype=float)

    if decisions_file and os.path.exists(decisions_file):
        with open(decisions_file, 'r') as f:
            decisions = json.load(f)

        # Map decisions to time steps
        current_bw = NORMAL_BW
        decision_idx = 0
        for i in range(len(traffic)):
            if decision_idx < len(decisions):
                action = decisions[decision_idx]['action']
                if action == 'expand':
                    current_bw = EXPANDED_BW
                elif action == 'contract':
                    current_bw = CONTRACTED_BW
                else:
                    pass  # hold
                decision_idx += 1
            allocated[i] = current_bw
    else:
        # Simulate proactive behavior: predict and pre-allocate
        from sklearn.preprocessing import MinMaxScaler
        window = 24
        current_bw = NORMAL_BW
        for i in range(window, len(traffic)):
            # Simple moving average as prediction proxy
            predicted_peak = np.max(traffic[i-window:i]) * 1.1
            predicted_avg = np.mean(traffic[i-window:i])

            if predicted_peak > 70 and current_bw != EXPANDED_BW:
                current_bw = EXPANDED_BW
            elif predicted_avg < 30 and current_bw != CONTRACTED_BW:
                current_bw = CONTRACTED_BW
            elif 30 <= predicted_avg <= 70:
                current_bw = NORMAL_BW
            allocated[i] = current_bw

    delivered = np.minimum(traffic, allocated)
    return allocated, delivered


def compute_metrics(traffic, allocated, delivered):
    """Compute performance metrics."""
    # Throughput utilization
    utilization = np.mean(delivered / allocated) * 100  # % of allocated BW used

    # SLA violations: when delivered < SLA_THRESHOLD * demand
    sla_met = delivered >= SLA_THRESHOLD * traffic
    sla_violations = np.sum(~sla_met)
    sla_violation_rate = (sla_violations / len(traffic)) * 100

    # Latency proxy: higher when congested (demand > allocated)
    congestion_ratio = np.clip(traffic / allocated, 0, 2)
    latency_proxy = np.mean(congestion_ratio) * 10  # ms scale

    # Packet loss proxy: when demand exceeds allocation
    excess = np.clip(traffic - allocated, 0, None)
    packet_loss = np.sum(excess) / np.sum(traffic) * 100

    # Wasted resources: allocated but unused
    wasted = np.clip(allocated - traffic, 0, None)
    waste_pct = np.sum(wasted) / np.sum(allocated) * 100

    return {
        'utilization_pct': round(utilization, 2),
        'sla_violations': int(sla_violations),
        'sla_violation_rate_pct': round(sla_violation_rate, 2),
        'avg_latency_proxy_ms': round(latency_proxy, 2),
        'packet_loss_pct': round(packet_loss, 2),
        'resource_waste_pct': round(waste_pct, 2)
    }


def main():
    # Load traffic data
    df = pd.read_csv(os.path.join(DATA_DIR, 'embb_traffic_timeseries.csv'))
    traffic = df['throughput_mbps'].values

    # Decisions file from controller
    decisions_file = os.path.join(DATA_DIR, 'controller_decisions.json')

    # Run simulations
    print("=" * 60)
    print("  Baseline Comparison: Static vs Reactive vs Proactive")
    print("=" * 60)

    static_alloc, static_deliv = simulate_static(traffic)
    reactive_alloc, reactive_deliv = simulate_reactive(traffic)
    proactive_alloc, proactive_deliv = simulate_proactive(traffic, decisions_file)

    static_metrics = compute_metrics(traffic, static_alloc, static_deliv)
    reactive_metrics = compute_metrics(traffic, reactive_alloc, reactive_deliv)
    proactive_metrics = compute_metrics(traffic, proactive_alloc, proactive_deliv)

    # Print results
    print(f"\n{'Metric':<30} {'Static':>10} {'Reactive':>10} {'Proactive':>10}")
    print("-" * 62)
    for key in static_metrics:
        print(f"{key:<30} {static_metrics[key]:>10} {reactive_metrics[key]:>10} {proactive_metrics[key]:>10}")

    # Save results
    results = {
        'static': static_metrics,
        'reactive': reactive_metrics,
        'proactive': proactive_metrics
    }
    results_path = os.path.join(DATA_DIR, 'baseline_comparison.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {results_path}")

    # Plot comparison
    os.makedirs(VIZ_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Baseline Comparison: Static vs Reactive vs Proactive Slicing', fontsize=14)

    labels = ['Static', 'Reactive', 'Proactive']
    colors = ['#e74c3c', '#f39c12', '#2ecc71']

    # 1. SLA Violation Rate
    vals = [static_metrics['sla_violation_rate_pct'],
            reactive_metrics['sla_violation_rate_pct'],
            proactive_metrics['sla_violation_rate_pct']]
    axes[0, 0].bar(labels, vals, color=colors)
    axes[0, 0].set_title('SLA Violation Rate (%)')
    axes[0, 0].set_ylabel('%')

    # 2. Resource Utilization
    vals = [static_metrics['utilization_pct'],
            reactive_metrics['utilization_pct'],
            proactive_metrics['utilization_pct']]
    axes[0, 1].bar(labels, vals, color=colors)
    axes[0, 1].set_title('Resource Utilization (%)')
    axes[0, 1].set_ylabel('%')

    # 3. Packet Loss
    vals = [static_metrics['packet_loss_pct'],
            reactive_metrics['packet_loss_pct'],
            proactive_metrics['packet_loss_pct']]
    axes[1, 0].bar(labels, vals, color=colors)
    axes[1, 0].set_title('Packet Loss (%)')
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
