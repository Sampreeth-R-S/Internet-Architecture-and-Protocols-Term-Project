"""
Collects per-slice traffic metrics at fixed intervals from the ogstun interface.
Logs: timestamp, rx_bytes, tx_bytes, rx_packets, tx_packets, throughput_mbps, packet_rate

Usage:
    python3 collect_metrics.py              # Run with default 1s interval
    python3 collect_metrics.py --interval 2 # Custom interval
"""
import csv
import time
import os
import argparse

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUTPUT_FILE = os.path.join(DATA_DIR, 'embb_traffic_timeseries.csv')
MONITORED_UE_IFACES = [f'uesimtun{i}' for i in range(5)]


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
    """Resolve monitored interfaces and restrict aggregation to uesimtun0..uesimtun4."""
    _ = interface_arg
    return MONITORED_UE_IFACES


def aggregate_interface_deltas(current_stats, previous_stats, candidates):
    """Sum byte and packet deltas across all monitored interfaces."""
    total_rx_bytes = 0
    total_tx_bytes = 0
    total_rx_packets = 0
    total_tx_packets = 0

    for iface in candidates:
        if iface not in current_stats or iface not in previous_stats:
            continue
        rx, tx, rxp, txp = current_stats[iface]
        prx, ptx, prxp, ptxp = previous_stats[iface]
        total_rx_bytes += max(0, rx - prx)
        total_tx_bytes += max(0, tx - ptx)
        total_rx_packets += max(0, rxp - prxp)
        total_tx_packets += max(0, txp - ptxp)

    return total_rx_bytes, total_tx_bytes, total_rx_packets, total_tx_packets


def get_active_ues():
    """Count active UERANSIM users by matching 'uesimtun' interfaces."""
    try:
        count = 0
        with open('/proc/net/dev', 'r') as f:
            for line in f:
                if 'uesimtun' in line:
                    count += 1
        return count
    except Exception as e:
        print(f"Error reading active UEs: {e}")
    return 0


def main():
    parser = argparse.ArgumentParser(description='Collect per-slice traffic metrics')
    parser.add_argument('--interval', type=int, default=1, help='Collection interval in seconds')
    parser.add_argument('--interface', type=str, default='auto',
                        help='Ignored: throughput is always aggregated from uesimtun0..uesimtun4')
    parser.add_argument('--output', type=str, default=OUTPUT_FILE, help='Output CSV file')
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    monitor_ifaces = discover_monitor_interfaces(args.interface)

    print(f"=== Collecting metrics from uesimtun0..uesimtun4 every {args.interval}s ===")
    print(f"Resolved monitor interfaces: {', '.join(monitor_ifaces)}")
    print(f"Output: {args.output}")
    print("Press Ctrl+C to stop.\n")

    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'rx_bytes', 'tx_bytes', 'rx_packets',
                         'tx_packets', 'throughput_mbps', 'packet_rate', 'active_ues', 'selected_interface'])

        prev_stats = read_interface_stats()

        try:
            while True:
                time.sleep(args.interval)
                curr_stats = read_interface_stats()
                active_ues = get_active_ues()

                rx_delta, tx_delta, rxp_delta, txp_delta = aggregate_interface_deltas(
                    curr_stats, prev_stats, monitor_ifaces
                )

                # Calculate aggregate rates across all monitored interfaces.
                throughput = (rx_delta + tx_delta) * 8 / (args.interval * 1e6)
                pkt_rate = (rxp_delta + txp_delta) / args.interval
                iface = 'aggregate'

                if rx_delta == 0 and tx_delta == 0 and rxp_delta == 0 and txp_delta == 0:
                    iface = 'none'

                writer.writerow([time.time(), rx_delta, tx_delta, rxp_delta, txp_delta,
                                 round(throughput, 4), round(pkt_rate, 2), active_ues, iface])
                f.flush()

                prev_stats = curr_stats
                print(f"[{time.strftime('%H:%M:%S')}] Throughput: {throughput:.2f} Mbps | "
                      f"Pkt Rate: {pkt_rate:.0f} pps | UEs: {active_ues} | Iface: {iface}")
        except KeyboardInterrupt:
            print("\n=== Collection stopped ===")


if __name__ == '__main__':
    main()
