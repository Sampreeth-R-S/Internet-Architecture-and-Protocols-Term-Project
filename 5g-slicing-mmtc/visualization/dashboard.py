"""
Visualization Dashboard for mMTC Network Slicing Project.
Generates all analysis plots:
  1. Traffic time-series (packet rate, device count, throughput)
  2. Controller decision timeline
  3. Baseline comparison charts
  4. Device density heatmap

Usage:
    python3 dashboard.py
"""
import numpy as np
import pandas as pd
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
VIZ_DIR = BASE_DIR

os.makedirs(VIZ_DIR, exist_ok=True)


def plot_traffic_timeseries():
    """Plot 1: mMTC traffic time-series showing IoT patterns."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'mmtc_traffic_timeseries.csv'))
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle('mMTC Traffic Time-Series Analysis', fontsize=14, fontweight='bold')

    # Packet rate
    axes[0].plot(df['timestamp'], df['packet_rate'], color='#e74c3c', linewidth=0.5, alpha=0.8)
    axes[0].fill_between(df['timestamp'], df['packet_rate'], alpha=0.2, color='#e74c3c')
    axes[0].set_ylabel('Packet Rate (pps)')
    axes[0].set_title('Sensor Packet Rate Over Time')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=500, color='red', linestyle='--', alpha=0.5, label='High threshold (500 pps)')
    axes[0].axhline(y=150, color='green', linestyle='--', alpha=0.5, label='Low threshold (150 pps)')
    axes[0].legend(loc='upper right')

    # Device count
    axes[1].plot(df['timestamp'], df['device_count'], color='#3498db', linewidth=0.8, alpha=0.8)
    axes[1].set_ylabel('Total Devices')
    axes[1].set_title('Connected IoT Device Count Over Time')
    axes[1].grid(True, alpha=0.3)

    # Throughput (low for mMTC)
    axes[2].plot(df['timestamp'], df['throughput_mbps'], color='#2ecc71', linewidth=0.5, alpha=0.8)
    axes[2].fill_between(df['timestamp'], df['throughput_mbps'], alpha=0.2, color='#2ecc71')
    axes[2].set_ylabel('Throughput (Mbps)')
    axes[2].set_title('Aggregate mMTC Throughput (Low BW, Many Devices)')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel('Time')

    plt.tight_layout()
    path = os.path.join(VIZ_DIR, 'traffic_timeseries.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✓ Traffic time-series → {path}")


def plot_controller_decisions():
    """Plot 2: Controller decision timeline for mMTC slice."""
    decisions_path = os.path.join(DATA_DIR, 'controller_decisions.json')
    if not os.path.exists(decisions_path):
        print("✗ No controller decisions file found, skipping...")
        return

    with open(decisions_path, 'r') as f:
        decisions = json.load(f)

    df = pd.read_csv(os.path.join(DATA_DIR, 'mmtc_traffic_timeseries.csv'))

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('mMTC Zero-Touch Controller Decision Timeline', fontsize=14, fontweight='bold')

    # Packet rate with decision overlay
    axes[0].plot(df['packet_rate'].values, color='#e74c3c', linewidth=0.5, alpha=0.7,
                 label='Actual packet rate')
    axes[0].axhline(y=400, color='red', linestyle='--', alpha=0.4, label='High threshold')
    axes[0].axhline(y=150, color='green', linestyle='--', alpha=0.4, label='Low threshold')

    expand_idxs = [i for i, d in enumerate(decisions) if d['action'] == 'expand']
    contract_idxs = [i for i, d in enumerate(decisions) if d['action'] == 'contract']

    if expand_idxs:
        axes[0].scatter(expand_idxs, [decisions[i]['trigger_value'] for i in expand_idxs],
                        color='red', marker='^', s=20, alpha=0.6, label='Expand', zorder=5)
    if contract_idxs:
        axes[0].scatter(contract_idxs, [decisions[i]['trigger_value'] for i in contract_idxs],
                        color='green', marker='v', s=20, alpha=0.6, label='Contract', zorder=5)

    axes[0].set_ylabel('Packet Rate (pps)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Decision action bar
    action_colors = {'expand': '#e74c3c', 'contract': '#2ecc71', 'hold': '#95a5a6'}
    colors = [action_colors.get(d['action'], '#95a5a6') for d in decisions]
    axes[1].bar(range(len(decisions)), [1] * len(decisions), color=colors, width=1.0)
    axes[1].set_ylabel('Action')
    axes[1].set_xlabel('Time Step')
    axes[1].set_yticks([])
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Expand'),
        Patch(facecolor='#2ecc71', label='Contract'),
        Patch(facecolor='#95a5a6', label='Hold')
    ]
    axes[1].legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    path = os.path.join(VIZ_DIR, 'controller_decisions.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✓ Controller decisions → {path}")


def plot_baseline_comparison():
    """Plot 3: Baseline comparison bar charts for mMTC."""
    results_path = os.path.join(DATA_DIR, 'baseline_comparison.json')
    if not os.path.exists(results_path):
        print("✗ No baseline comparison file found, skipping...")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)

    metrics = ['sla_violation_rate_pct', 'pdr_pct', 'congestion_rate_pct', 'resource_waste_pct']
    titles = ['SLA Violation Rate (%)', 'Packet Delivery Ratio (%)',
              'Congestion Rate (%)', 'Resource Waste (%)']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('mMTC Performance: Static vs Reactive vs Proactive', fontsize=14, fontweight='bold')

    labels = ['Static', 'Reactive', 'Proactive']
    colors = ['#e74c3c', '#f39c12', '#2ecc71']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        vals = [results['static'][metric], results['reactive'][metric], results['proactive'][metric]]
        bars = ax.bar(labels, vals, color=colors)
        ax.set_title(title)
        ax.set_ylabel('%')

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    path = os.path.join(VIZ_DIR, 'baseline_comparison_dashboard.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✓ Baseline comparison → {path}")


def plot_device_density_heatmap():
    """Plot 4: Device density / packet rate heatmap over time."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'mmtc_traffic_timeseries.csv'))
    packet_rate = df['packet_rate'].values

    # Reshape into 2D (hours x samples_per_hour)
    samples_per_hour = 720  # 3600/5
    n_hours = len(packet_rate) // samples_per_hour
    data = packet_rate[:n_hours * samples_per_hour].reshape(n_hours, samples_per_hour)

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=np.percentile(packet_rate, 99))
    ax.set_xlabel('Time within hour (samples)')
    ax.set_ylabel('Hour')
    ax.set_title('mMTC Device Density Heatmap (Packet Rate pps)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Packet Rate (pps)')

    plt.tight_layout()
    path = os.path.join(VIZ_DIR, 'device_density_heatmap.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✓ Device density heatmap → {path}")


def main():
    print("=" * 60)
    print("  mMTC Network Slicing — Visualization Dashboard")
    print("=" * 60)
    print()

    plot_traffic_timeseries()
    plot_controller_decisions()
    plot_baseline_comparison()
    plot_device_density_heatmap()

    print(f"\n✓ All plots saved to {VIZ_DIR}/")


if __name__ == '__main__':
    main()
