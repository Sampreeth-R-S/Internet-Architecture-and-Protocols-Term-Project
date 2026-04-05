# custom_udp_client.py

import socket
import time
import threading
import random
import struct
import os
import subprocess
import math

SERVER_IP = "10.46.0.10"
BASE_PORT = 5202

UE_IFACES = [f"uesimtun{i}" for i in range(5, 10)]
NUM_USERS = len(UE_IFACES)
TOTAL_DURATION = 3000

PACKET_SIZES = [64, 128, 256, 512, 1024] * 20
PACKET_RATES = [10, 20, 50, 100, 200]

end_time = time.time() + TOTAL_DURATION

mu = math.log(2.5)   # ~2-3 ms median
sigma = 0.4          # low jitter


def get_interface_ip(iface_name):
    """Return IPv4 address for the given interface."""
    try:
        result = subprocess.run(
            ["ip", "-4", "-o", "addr", "show", "dev", iface_name],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to query IP for {iface_name}: {e}")

    for line in result.stdout.strip().splitlines():
        parts = line.split()
        if "inet" in parts:
            inet_idx = parts.index("inet")
            if inet_idx + 1 < len(parts):
                return parts[inet_idx + 1].split("/")[0]

    raise RuntimeError(f"No IPv4 address found on {iface_name}")


def resolve_ue_ips(ifaces):
    """Resolve source IPs for uesimtun5..uesimtun9."""
    iface_to_ip = {}
    missing_ifaces = [iface for iface in ifaces if not os.path.exists(f"/sys/class/net/{iface}")]
    if missing_ifaces:
        raise RuntimeError(f"Missing interfaces: {', '.join(missing_ifaces)}")

    for iface in ifaces:
        iface_to_ip[iface] = get_interface_ip(iface)

    return iface_to_ip


def busy_wait_delay(delay_sec):
    start = time.perf_counter()
    end = start + delay_sec

    while time.perf_counter() < end:
        pass


def bind_socket_to_interface(sock, iface_name):
    """Bind socket to specific interface using SO_BINDTODEVICE."""
    if not hasattr(socket, "SO_BINDTODEVICE"):
        raise RuntimeError("SO_BINDTODEVICE not available")
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, iface_name.encode() + b"\0")


UE_IPS_BY_IFACE = resolve_ue_ips(UE_IFACES)
print("Interfaces:", UE_IFACES)
print("Resolved UE IPs:", ", ".join(f"{iface}={UE_IPS_BY_IFACE[iface]}" for iface in UE_IFACES))


def run_ue(ue_index):
    iface = UE_IFACES[ue_index]
    ue_ip = UE_IPS_BY_IFACE[iface]

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind source IP first, then lock socket to interface.
    sock.bind((ue_ip, 0))

    try:
        bind_socket_to_interface(sock, iface)
    except Exception as e:
        print(f"[UE{ue_index + 1}] Interface bind failed: {e}")

    print(f"[UE{ue_index + 1}] IP={ue_ip}, IFACE={iface}")

    seq = 0

    while time.time() < end_time:
        pkt_size = random.choice(PACKET_SIZES)
        pkt_rate = random.choice(PACKET_RATES)

        interval = 1.0 / pkt_rate

        ts = time.time()

        payload_size = max(0, pkt_size - 12)
        payload = os.urandom(payload_size)

        packet = struct.pack("!Id", seq, ts) + payload

        delay_ms = random.lognormvariate(mu, sigma)
        delay_ms = min(delay_ms, 10)
        delay_ms = max(0, min(10, delay_ms))
        # busy_wait_delay(delay_ms / 1000.0)

        try:
            sock.sendto(packet, (SERVER_IP, BASE_PORT))
        except Exception as e:
            print(f"[UE{ue_index + 1}] Send error: {e}")

        seq += 1
        time.sleep(interval)


threads = []

for i in range(NUM_USERS):
    t = threading.Thread(target=run_ue, args=(i,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
