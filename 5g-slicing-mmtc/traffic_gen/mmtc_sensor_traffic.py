"""
mMTC Sensor Traffic Simulator — Massive IoT UDP Packet Generator.
Simulates thousands of IoT sensors sending small UDP packets to a
central collector, mimicking real mMTC slice traffic patterns.

Sensor types:
  - Temperature/Humidity  (64 B payload,  every 30s)
  - Smart Meter           (128 B payload, every 60s)
  - Motion/Alarm          (48 B payload,  event-driven bursts)
  - Asset Tracker GPS     (96 B payload,  every 120s)

Usage:
    python3 mmtc_sensor_traffic.py                        # Default 1000 sensors
    python3 mmtc_sensor_traffic.py --sensors 5000         # 5000 sensors
    python3 mmtc_sensor_traffic.py --target 10.45.0.1     # Remote target
    python3 mmtc_sensor_traffic.py --duration 600         # 10 minutes
    python3 mmtc_sensor_traffic.py --live                 # Send real UDP packets
"""
import socket
import struct
import time
import random
import csv
import os
import argparse
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

# --- Sensor type definitions ---
SENSOR_TYPES = {
    'temperature': {
        'payload_bytes': 64,
        'interval_s': 30,
        'priority': 'low',
        'fraction': 0.40,
        'jitter_s': 5,
    },
    'smart_meter': {
        'payload_bytes': 128,
        'interval_s': 60,
        'priority': 'low',
        'fraction': 0.25,
        'jitter_s': 10,
    },
    'motion_alarm': {
        'payload_bytes': 48,
        'interval_s': 0,  # event-driven
        'priority': 'medium',
        'fraction': 0.15,
        'jitter_s': 0,
    },
    'asset_tracker': {
        'payload_bytes': 96,
        'interval_s': 120,
        'priority': 'low',
        'fraction': 0.20,
        'jitter_s': 15,
    },
}

# Event burst parameters for motion/alarm sensors
BURST_PROBABILITY = 0.02   # 2% chance per second that a burst event begins
BURST_DURATION_S = (5, 20) # burst lasts 5-20 seconds
BURST_SENSORS_RATIO = 0.5  # 50% of alarm sensors fire during a burst


def build_sensor_payload(sensor_id, sensor_type, seq_num):
    """Build a compact binary payload mimicking real sensor data."""
    cfg = SENSOR_TYPES[sensor_type]
    size = cfg['payload_bytes']

    # Header: sensor_id (4B) + seq (4B) + timestamp (8B) + type_id (1B) = 17B
    type_ids = {'temperature': 1, 'smart_meter': 2, 'motion_alarm': 3, 'asset_tracker': 4}
    header = struct.pack('!IIdB', sensor_id, seq_num, time.time(), type_ids[sensor_type])

    # Fill remaining bytes with simulated sensor readings
    data_len = max(0, size - len(header))
    data = bytes(random.getrandbits(8) for _ in range(data_len))

    return header + data


def is_sensor_awake(sensor, sim_time):
    """Return whether a sensor is in its active window for this simulation tick."""
    if not sensor.get('sleep_enabled', False):
        return True

    period = sensor['sleep_period_s']
    window = sensor['awake_window_s']
    phase = sensor['sleep_phase_s']
    in_cycle = (sim_time + phase) % period
    return in_cycle < window


def transmit_with_impairments(sock, payload, target_ip, target_port, live, loss_rate, max_retries, retry_delay_s):
    """Apply optional packet-loss and retry behavior, then transmit if allowed."""
    retries_used = 0
    attempts = max_retries + 1

    for attempt in range(attempts):
        if random.random() < loss_rate:
            if attempt < max_retries:
                retries_used += 1
                if live and retry_delay_s > 0:
                    time.sleep(retry_delay_s)
                continue
            return False, retries_used, 'dropped'

        if not live:
            return True, retries_used, 'ok'

        try:
            sock.sendto(payload, (target_ip, target_port))
            return True, retries_used, 'ok'
        except OSError:
            if attempt < max_retries:
                retries_used += 1
                if retry_delay_s > 0:
                    time.sleep(retry_delay_s)
                continue
            return False, retries_used, 'send_error'

    return False, retries_used, 'send_error'


def bind_socket_to_interface(sock, interface_name):
    """Bind a UDP socket to a specific Linux network interface."""
    if not interface_name:
        return

    if not hasattr(socket, 'SO_BINDTODEVICE'):
        raise RuntimeError('SO_BINDTODEVICE is not available on this platform')

    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, interface_name.encode() + b'\0')


def simulate_traffic(args):
    """Main simulation loop — generates mMTC traffic patterns."""
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, 'mmtc_sensor_session.csv')
    timeseries_path = os.path.join(DATA_DIR, 'mmtc_traffic_timeseries.csv')

    num_sensors = args.sensors
    duration = args.duration
    target_ip = args.target
    target_port = args.port
    live = args.live
    interface_name = args.interface

    if args.seed is not None:
        random.seed(args.seed)

    # Assign sensor types
    sensors = []
    sensor_id = 0
    for stype, cfg in SENSOR_TYPES.items():
        count = int(num_sensors * cfg['fraction'])
        for _ in range(count):
            sensors.append({
                'id': sensor_id,
                'type': stype,
                'next_tx': random.uniform(0, min(cfg['interval_s'], 10)) if cfg['interval_s'] > 0 else float('inf'),
                'seq': 0,
                'interval': cfg['interval_s'],
                'payload_bytes': cfg['payload_bytes'],
                'jitter': cfg['jitter_s'],
                'sleep_enabled': (stype != 'motion_alarm' and random.random() < args.sleep_ratio),
                'sleep_period_s': args.sleep_period_s,
                'awake_window_s': min(args.sleep_window_s, args.sleep_period_s),
                'sleep_phase_s': random.uniform(0, max(1, args.sleep_period_s)),
            })
            sensor_id += 1

    # Separate alarm sensors for burst simulation
    alarm_sensors = [s for s in sensors if s['type'] == 'motion_alarm']
    periodic_sensors = [s for s in sensors if s['type'] != 'motion_alarm']

    print("=" * 60)
    print("  mMTC Sensor Traffic Simulator")
    print("=" * 60)
    print(f"  Total sensors:    {len(sensors)}")
    for stype, cfg in SENSOR_TYPES.items():
        count = sum(1 for s in sensors if s['type'] == stype)
        print(f"    {stype:20s}: {count:5d} sensors ({cfg['payload_bytes']}B, "
              f"{'event-driven' if cfg['interval_s'] == 0 else str(cfg['interval_s']) + 's interval'})")
    print(f"  Duration:         {duration}s")
    print(f"  Target:           {target_ip}:{target_port}")
    print(f"  Mode:             {'LIVE (sending UDP)' if live else 'SIMULATION (log only)'}")
    print(f"  Loss model:       {args.loss_rate * 100:.1f}% packet loss, max {args.max_retries} retries")
    print(f"  Duty cycle:       {args.sleep_ratio * 100:.1f}% periodic devices use {args.sleep_window_s}/{args.sleep_period_s}s awake/sleep cycle")
    print()

    # Open UDP socket if live mode
    sock = None
    if live:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            bind_socket_to_interface(sock, interface_name)
            if interface_name:
                print(f"  Bound live socket to interface: {interface_name}")
                print()
        except PermissionError:
            raise SystemExit(f"Failed to bind to interface '{interface_name}': root privileges required for SO_BINDTODEVICE")
        except OSError as exc:
            raise SystemExit(f"Failed to bind to interface '{interface_name}': {exc}")

    if live and target_ip.startswith('10.45.'):
        print("  Note: 10.45.x.x target typically uses UE tunnel; ensure uesimtun0 is up.")
        print()

    # Per-second aggregation for time-series
    timeseries_rows = []
    session_rows = []

    # Burst state
    burst_active = False
    burst_end_time = 0

    total_retries = 0
    total_dropped = 0
    total_send_errors = 0

    start_time = time.time()
    sim_time = 0.0
    tick_interval = 1.0  # 1-second simulation ticks

    try:
        while sim_time < duration:
            tick_start = time.time()
            packets_this_tick = 0
            bytes_this_tick = 0
            active_devices = 0
            types_count = defaultdict(int)
            dropped_this_tick = 0
            retries_this_tick = 0
            send_errors_this_tick = 0

            # Check for burst events (alarm sensors)
            if not burst_active and random.random() < BURST_PROBABILITY:
                burst_active = True
                burst_duration = random.uniform(*BURST_DURATION_S)
                burst_end_time = sim_time + burst_duration
                burst_count = int(len(alarm_sensors) * BURST_SENSORS_RATIO)
                burst_subset = random.sample(alarm_sensors, min(burst_count, len(alarm_sensors)))
                print(f"  [{sim_time:.0f}s] ⚡ BURST EVENT: {len(burst_subset)} alarm sensors triggered "
                      f"(duration: {burst_duration:.0f}s)")
                # Set alarm sensors to fire every 0.5-2s during burst
                for s in burst_subset:
                    s['next_tx'] = sim_time
                    s['interval'] = random.uniform(0.5, 2.0)

            if burst_active and sim_time >= burst_end_time:
                burst_active = False
                # Reset alarm sensors to dormant
                for s in alarm_sensors:
                    s['next_tx'] = float('inf')
                    s['interval'] = 0

            # Process periodic sensors
            for s in periodic_sensors:
                if sim_time >= s['next_tx'] and is_sensor_awake(s, sim_time):
                    payload = build_sensor_payload(s['id'], s['type'], s['seq'])

                    ok, retries_used, status = transmit_with_impairments(
                        sock, payload, target_ip, target_port, live,
                        args.loss_rate, args.max_retries, args.retry_delay_ms / 1000.0
                    )
                    retries_this_tick += retries_used
                    total_retries += retries_used

                    if ok:
                        packets_this_tick += 1
                        bytes_this_tick += len(payload)
                        active_devices += 1
                        types_count[s['type']] += 1
                        s['seq'] += 1
                    elif status == 'dropped':
                        dropped_this_tick += 1
                        total_dropped += 1
                    else:
                        send_errors_this_tick += 1
                        total_send_errors += 1

                    jitter = random.uniform(-s['jitter'], s['jitter']) if s['jitter'] > 0 else 0
                    s['next_tx'] = sim_time + s['interval'] + jitter

                    session_rows.append({
                        'timestamp': start_time + sim_time,
                        'sensor_id': s['id'],
                        'sensor_type': s['type'],
                        'payload_bytes': len(payload),
                        'seq_num': s['seq'],
                        'status': status,
                    })

            # Process alarm sensors during burst
            if burst_active:
                for s in alarm_sensors:
                    if s['interval'] > 0 and sim_time >= s['next_tx']:
                        payload = build_sensor_payload(s['id'], s['type'], s['seq'])

                        ok, retries_used, status = transmit_with_impairments(
                            sock, payload, target_ip, target_port, live,
                            args.loss_rate, args.max_retries, args.retry_delay_ms / 1000.0
                        )
                        retries_this_tick += retries_used
                        total_retries += retries_used

                        if ok:
                            packets_this_tick += 1
                            bytes_this_tick += len(payload)
                            active_devices += 1
                            types_count[s['type']] += 1
                            s['seq'] += 1
                        elif status == 'dropped':
                            dropped_this_tick += 1
                            total_dropped += 1
                        else:
                            send_errors_this_tick += 1
                            total_send_errors += 1
                        s['next_tx'] = sim_time + s['interval']

                        session_rows.append({
                            'timestamp': start_time + sim_time,
                            'sensor_id': s['id'],
                            'sensor_type': s['type'],
                            'payload_bytes': len(payload),
                            'seq_num': s['seq'],
                            'status': status,
                        })

            # Calculate throughput for this tick
            throughput_mbps = (bytes_this_tick * 8) / (tick_interval * 1e6)
            packet_rate = packets_this_tick / tick_interval

            timeseries_rows.append({
                'timestamp': start_time + sim_time,
                'packets': packets_this_tick,
                'bytes': bytes_this_tick,
                'throughput_mbps': round(throughput_mbps, 6),
                'packet_rate': round(packet_rate, 2),
                'active_devices': active_devices,
                'dropped_packets': dropped_this_tick,
                'retries': retries_this_tick,
                'send_errors': send_errors_this_tick,
                'burst_active': 1 if burst_active else 0,
                'temp_sensors': types_count.get('temperature', 0),
                'meter_sensors': types_count.get('smart_meter', 0),
                'alarm_sensors': types_count.get('motion_alarm', 0),
                'tracker_sensors': types_count.get('asset_tracker', 0),
            })

            # Progress output every 30 seconds
            if int(sim_time) % 30 == 0 and sim_time > 0:
                print(f"  [{sim_time:.0f}s] pkts={packets_this_tick} | "
                      f"{throughput_mbps:.4f} Mbps | "
                      f"devices={active_devices} | "
                        f"drops={dropped_this_tick} | "
                        f"burst={'YES' if burst_active else 'no'}")

            sim_time += tick_interval

            # In live mode, sleep to maintain real-time pacing
            if live:
                elapsed = time.time() - tick_start
                sleep_time = max(0, tick_interval - elapsed)
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n=== Simulation stopped by user ===")
    finally:
        if sock:
            sock.close()

    # Save session CSV
    with open(csv_path, 'w', newline='') as f:
        if session_rows:
            writer = csv.DictWriter(f, fieldnames=session_rows[0].keys())
            writer.writeheader()
            writer.writerows(session_rows)
    print(f"\nSession log ({len(session_rows)} packets) → {csv_path}")

    # Save time-series CSV
    with open(timeseries_path, 'w', newline='') as f:
        if timeseries_rows:
            writer = csv.DictWriter(f, fieldnames=timeseries_rows[0].keys())
            writer.writeheader()
            writer.writerows(timeseries_rows)
    print(f"Time-series ({len(timeseries_rows)} ticks) → {timeseries_path}")

    # Summary
    total_packets = sum(r['packets'] for r in timeseries_rows)
    total_bytes = sum(r['bytes'] for r in timeseries_rows)
    burst_ticks = sum(1 for r in timeseries_rows if r['burst_active'])
    print(f"\nSummary:")
    print(f"  Total packets:    {total_packets}")
    print(f"  Total bytes:      {total_bytes} ({total_bytes / 1e6:.2f} MB)")
    print(f"  Avg packet rate:  {total_packets / max(1, len(timeseries_rows)):.1f} pps")
    print(f"  Burst ticks:      {burst_ticks}/{len(timeseries_rows)}")
    print(f"  Dropped packets:  {total_dropped}")
    print(f"  Retry attempts:   {total_retries}")
    print(f"  Send errors:      {total_send_errors}")


def main():
    parser = argparse.ArgumentParser(description='mMTC Sensor Traffic Simulator')
    parser.add_argument('--sensors', type=int, default=1000,
                        help='Total number of simulated IoT sensors (default: 1000)')
    parser.add_argument('--duration', type=int, default=300,
                        help='Simulation duration in seconds (default: 300)')
    parser.add_argument('--target', type=str, default='127.0.0.1',
                        help='Target IP for UDP packets (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=9999,
                        help='Target UDP port (default: 9999)')
    parser.add_argument('--live', action='store_true',
                        help='Send real UDP packets (default: simulation only)')
    parser.add_argument('--interface', type=str, default='uesimtun0',
                        help='Linux network interface to bind live UDP traffic to (default: uesimtun0)')
    parser.add_argument('--loss-rate', type=float, default=0.0,
                        help='Packet loss probability in [0,1] applied before each send attempt (default: 0.0)')
    parser.add_argument('--max-retries', type=int, default=0,
                        help='Retry attempts after a lost/failed send (default: 0)')
    parser.add_argument('--retry-delay-ms', type=int, default=20,
                        help='Delay between retries in milliseconds (default: 20)')
    parser.add_argument('--sleep-ratio', type=float, default=0.6,
                        help='Fraction of periodic sensors that follow sleep/wake cycles (default: 0.6)')
    parser.add_argument('--sleep-period-s', type=int, default=300,
                        help='Sleep cycle period in seconds (default: 300)')
    parser.add_argument('--sleep-window-s', type=int, default=120,
                        help='Awake window per cycle in seconds (default: 120)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Optional random seed for reproducible runs')
    args = parser.parse_args()

    args.loss_rate = min(1.0, max(0.0, args.loss_rate))
    args.max_retries = max(0, args.max_retries)
    args.sleep_ratio = min(1.0, max(0.0, args.sleep_ratio))
    args.sleep_period_s = max(1, args.sleep_period_s)
    args.sleep_window_s = min(args.sleep_period_s, max(1, args.sleep_window_s))
    args.retry_delay_ms = max(0, args.retry_delay_ms)

    simulate_traffic(args)


if __name__ == '__main__':
    main()
