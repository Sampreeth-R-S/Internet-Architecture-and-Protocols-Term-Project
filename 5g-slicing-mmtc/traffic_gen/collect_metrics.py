"""
Collects per-slice traffic metrics at fixed intervals from the ogstun interface.
Adapted for mMTC slice — tracks packet rate, device count, and small-packet throughput.

Usage:
    python3 collect_metrics.py              # Run with default 1s interval
    python3 collect_metrics.py --interval 2 # Custom interval
"""
import csv
import time
import os
import argparse

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUTPUT_FILE = os.path.join(DATA_DIR, 'mmtc_traffic_timeseries.csv')


def read_interface_stats():
    """Read /proc/net/dev and return per-interface counters."""
    stats = {}
    try:
        with open('/proc/net/dev', 'r') as f:
            for line in f:
                if ':' not in line:
                    continue
                name, data = line.split(':', 1)
                iface = name.strip()
                parts = data.split()
                if len(parts) < 10:
                    continue
                stats[iface] = (int(parts[0]), int(parts[8]), int(parts[1]), int(parts[9]))
    except Exception as e:
        print(f"Error reading interface stats: {e}")
    return stats


def discover_monitor_interfaces(interface_arg):
    """Resolve which interfaces to monitor."""
    all_stats = read_interface_stats()
    all_ifaces = list(all_stats.keys())
    ue_ifaces = sorted([i for i in all_ifaces if i.startswith('uesimtun')])

    if interface_arg == 'auto':
        candidates = []
        if 'ogstun' in all_ifaces:
            candidates.append('ogstun')
        if 'lo' in all_ifaces:
            candidates.append('lo')
        candidates.extend(ue_ifaces)
        return candidates if candidates else ['lo']

    if ',' in interface_arg:
        return [i.strip() for i in interface_arg.split(',') if i.strip()]

    return [interface_arg]


def select_active_interface(current_stats, previous_stats, candidates):
    """Pick interface with the highest byte delta for current interval."""
    best_iface = None
    best_delta = -1
    for iface in candidates:
        if iface not in current_stats or iface not in previous_stats:
            continue
        rx, tx, _, _ = current_stats[iface]
        prx, ptx, _, _ = previous_stats[iface]
        delta = max(0, (rx - prx) + (tx - ptx))
        if delta > best_delta:
            best_delta = delta
            best_iface = iface
    return best_iface


def get_active_ues():
    """Count active UERANSIM users by matching 'uesimtun' interfaces."""
    try:
        count = 0
        with open('/proc/net/dev', 'r') as f:
            for line in f:
                if 'uesimtun' in line:
                    count += 1
        return count
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(description='Collect mMTC slice traffic metrics')
    parser.add_argument('--interval', type=int, default=1, help='Collection interval in seconds')
    parser.add_argument('--interface', type=str, default='auto',
                        help='Network interface to monitor (single, comma-list, or auto)')
    parser.add_argument('--output', type=str, default=OUTPUT_FILE, help='Output CSV file')
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    monitor_ifaces = discover_monitor_interfaces(args.interface)

    print(f"=== Collecting mMTC metrics from '{args.interface}' every {args.interval}s ===")
    print(f"Resolved monitor interfaces: {', '.join(monitor_ifaces)}")
    print(f"Output: {args.output}")
    print("Press Ctrl+C to stop.\n")

    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'rx_bytes', 'tx_bytes', 'rx_packets',
                         'tx_packets', 'throughput_mbps', 'packet_rate',
                         'active_devices', 'selected_interface'])

        prev_stats = read_interface_stats()

        try:
            while True:
                time.sleep(args.interval)
                curr_stats = read_interface_stats()
                active_devices = get_active_ues()

                iface = select_active_interface(curr_stats, prev_stats, monitor_ifaces)
                if iface and iface in curr_stats:
                    rx, tx, rxp, txp = curr_stats[iface]
                    prx, ptx, prxp, ptxp = prev_stats.get(iface, (rx, tx, rxp, txp))
                else:
                    iface = 'none'
                    rx, tx, rxp, txp = 0, 0, 0, 0
                    prx, ptx, prxp, ptxp = 0, 0, 0, 0

                throughput = (max(0, (rx - prx) + (tx - ptx))) * 8 / (args.interval * 1e6)
                pkt_rate = max(0, (rxp - prxp) + (txp - ptxp)) / args.interval

                writer.writerow([time.time(), rx, tx, rxp, txp,
                                 round(throughput, 4), round(pkt_rate, 2),
                                 active_devices, iface])
                f.flush()

                prev_stats = curr_stats
                print(f"[{time.strftime('%H:%M:%S')}] Throughput: {throughput:.4f} Mbps | "
                      f"Pkt Rate: {pkt_rate:.0f} pps | Devices: {active_devices} | Iface: {iface}")
        except KeyboardInterrupt:
            print("\n=== Collection stopped ===")


if __name__ == '__main__':
    main()
