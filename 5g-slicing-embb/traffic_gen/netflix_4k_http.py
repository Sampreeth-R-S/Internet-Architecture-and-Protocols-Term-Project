#!/usr/bin/env python3
"""
Netflix-style 4K Streaming Simulator with proper UE interface binding.
Uses UDP packets to simulate HTTP Range requests, following the pattern
of mmtc_sensor_traffic.py with SO_BINDTODEVICE for reliable interface routing.

Usage:
    sudo python3 netflix_4k_http.py
    sudo python3 netflix_4k_http.py --segment-count 30
    sudo python3 netflix_4k_http.py --target 10.45.0.1
"""
import socket
import struct
import time
import random
import csv
import os
import argparse
import subprocess
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MEDIA_DIR = os.path.join(BASE_DIR, '..', 'data', 'media')

# ABR-like per-user profiles in Mbps (5 users)
UE0_PROFILE = [24, 30, 36, 28, 34, 26, 38, 29, 35, 27]
UE1_PROFILE = [18, 22, 27, 21, 25, 19, 28, 20, 24, 18]
UE2_PROFILE = [26, 33, 40, 31, 37, 29, 42, 32, 38, 30]
UE3_PROFILE = [16, 20, 24, 18, 22, 17, 25, 19, 23, 16]
UE4_PROFILE = [20, 26, 32, 24, 30, 22, 34, 25, 31, 21]

PROFILES = [UE0_PROFILE, UE1_PROFILE, UE2_PROFILE, UE3_PROFILE, UE4_PROFILE]


def get_ue_interfaces():
    """Discover 5 uesimtun interfaces and their IPs."""
    try:
        result = subprocess.run(['ip', '-4', '-o', 'addr', 'show'], 
                              capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        ifaces = {}
        for line in lines:
            if 'uesimtun' in line:
                parts = line.split()
                iface_name = parts[1]
                ip_addr = parts[3].split('/')[0]
                if len(ifaces) < 5:
                    ifaces[iface_name] = ip_addr
        if len(ifaces) < 5:
            print(f"Error: Need 5 UE tunnels. Found {len(ifaces)}")
            exit(1)
        return ifaces
    except Exception as e:
        print(f"Error discovering interfaces: {e}")
        exit(1)


def bind_socket_to_interface(sock, iface_name):
    """Bind socket to specific interface using SO_BINDTODEVICE."""
    if not hasattr(socket, 'SO_BINDTODEVICE'):
        raise RuntimeError('SO_BINDTODEVICE not available')
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, 
                   iface_name.encode() + b'\0')


def build_request_payload(segment_num, ue_id, start_byte, end_byte, rate_mbps):
    """Build a simulated HTTP Range request as binary payload."""
    # Minimal HTTP Range request simulation
    # Format: segment_id(4) | ue_id(4) | start_byte(8) | end_byte(8) | rate_mbps(4) = 28B header
    # Followed by dummy data to reach the desired byte count
    header = struct.pack('!IIQQI', segment_num, ue_id, start_byte, end_byte, rate_mbps)
    return header


def send_udp_burst(iface_name, target_ip, target_port, num_bytes, segment_num, ue_id, timeout=15):
    """Send UDP packets to simulate HTTP streaming through specific interface."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        bind_socket_to_interface(sock, iface_name)
        sock.settimeout(timeout)
        
        # Calculate number of packets needed (UDP max ~1472 payload per packet)
        packet_size = 1400  # Conservative UDP payload
        num_packets = (num_bytes + packet_size - 1) // packet_size
        
        bytes_sent = 0
        t0 = time.time()
        
        for pkt_num in range(num_packets):
            start_byte = pkt_num * packet_size
            end_byte = min(start_byte + packet_size - 1, num_bytes - 1)
            payload_size = end_byte - start_byte + 1
            
            # Build a simulated Range request
            header = build_request_payload(segment_num, ue_id, start_byte, end_byte, 0)
            
            # Pad with dummy data to reach target byte count per packet
            dummy_data = os.urandom(max(0, payload_size - len(header)))
            packet = header + dummy_data
            
            sock.sendto(packet, (target_ip, target_port))
            bytes_sent += len(packet)
        
        sock.close()
        duration = time.time() - t0
        return True, bytes_sent, duration
    
    except Exception as e:
        return False, 0, 0.0


def prepare_media_file(media_path, size_mb):
    """Create media file if it doesn't exist."""
    os.makedirs(os.path.dirname(media_path), exist_ok=True)
    target_bytes = size_mb * 1024 * 1024
    
    if not os.path.exists(media_path):
        print(f"Creating media file: {media_path} ({size_mb}MB)")
        try:
            with open(media_path, 'wb') as f:
                f.truncate(target_bytes)
        except Exception as e:
            print(f"Error creating media file: {e}")
            exit(1)
    else:
        print(f"Using existing media file: {media_path}")
    
    return target_bytes


def main():
    parser = argparse.ArgumentParser(description='Netflix 4K UDP Streaming Simulator')
    parser.add_argument('--segment-count', type=int, default=90,
                       help='Number of segments to stream (default: 90)')
    parser.add_argument('--segment-duration', type=int, default=4,
                       help='Duration per segment in seconds (default: 4)')
    parser.add_argument('--target', type=str, default='10.45.0.1',
                       help='Target IP for UDP packets (default: 10.45.0.1)')
    parser.add_argument('--port', type=int, default=9999,
                       help='Target UDP port (default: 9999)')
    parser.add_argument('--media-size-mb', type=int, default=768,
                       help='Media file size in MB (default: 768)')
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, 'netflix_4k_session.csv')
    
    # Setup
    print("=" * 70)
    print("  Netflix-style 4K Streaming Simulator (UDP + Interface Binding)")
    print("=" * 70)
    
    ue_ifaces = get_ue_interfaces()
    print(f"  UE Interfaces: {', '.join(ue_ifaces.keys())}")
    print(f"  Target: {args.target}:{args.port}")
    print(f"  Segments: {args.segment_count} × {args.segment_duration}s")
    print(f"  Mode: UDP burst (SO_BINDTODEVICE per interface)")
    print()
    
    media_file = 'netflix_4k_source.bin'
    media_path = os.path.join(MEDIA_DIR, media_file)
    file_size = prepare_media_file(media_path, args.media_size_mb)
    print()
    
    # Per-user offsets
    offsets = [
        0,
        file_size // 5,
        file_size * 2 // 5,
        file_size * 3 // 5,
        file_size * 4 // 5,
    ]
    
    # Streaming simulation
    iface_list = list(ue_ifaces.keys())
    csv_rows = []
    
    try:
        for segment in range(args.segment_count):
            idx = segment % len(PROFILES[0])
            rates = [p[idx] for p in PROFILES]
            
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"Segment {segment+1}/{args.segment_count} | "
                  f"rates: {' '.join(f'{r}' for r in rates)} Mbps")
            
            for ue, iface in enumerate(iface_list):
                rate_mbps = rates[ue]
                bytes_to_download = (rate_mbps * 1000000 // 8) * args.segment_duration
                
                start = offsets[ue]
                if start >= file_size:
                    start = start % file_size
                
                end = min(start + bytes_to_download - 1, file_size - 1)
                bytes_to_download = end - start + 1
                
                offsets[ue] = end + 1
                if offsets[ue] >= file_size:
                    offsets[ue] = offsets[ue] % file_size
                
                success, bytes_sent, duration = send_udp_burst(
                    iface, args.target, args.port, bytes_to_download,
                    segment + 1, ue + 1, timeout=args.segment_duration + 8
                )
                
                duration_ms = int(duration * 1000)
                status = 'ok' if success else 'fail'
                
                csv_rows.append({
                    'timestamp': int(time.time()),
                    'segment': segment + 1,
                    'ue': ue + 1,
                    'iface': iface,
                    'rate_mbps': rate_mbps,
                    'bytes': bytes_to_download,
                    'status': status,
                    'duration_ms': duration_ms,
                })
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n=== Simulation stopped by user ===")
    finally:
        pass
    
    # Save CSV
    with open(csv_path, 'w', newline='') as f:
        if csv_rows:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
    
    print(f"\n=== Streaming Simulation Complete ===")
    print(f"Session metrics: {csv_path}")


if __name__ == '__main__':
    main()
