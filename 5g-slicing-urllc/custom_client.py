# custom_udp_client.py

import socket
import time
import threading
import random
import struct
import os

SERVER_IP = "10.46.0.10"
BASE_PORT = 5202

NUM_USERS = 5
TOTAL_DURATION = 3000

UE_IPS = [
    "10.45.0.2",
    "10.45.0.3",
    "10.45.0.4",
    "10.45.0.5",
    "10.45.0.6"
]

PACKET_SIZES = [64, 128, 256, 512, 1024] * 20
PACKET_RATES = [10, 20, 50, 100, 200]

end_time = time.time() + TOTAL_DURATION

# Get UE interfaces
interfaces = sorted([
    iface for iface in os.listdir("/sys/class/net/")
    if iface.startswith("uesimtun")
])

UE_IFACES = interfaces[:NUM_USERS]

print("Interfaces:", UE_IFACES)

def busy_wait_delay(delay_sec):
    start = time.perf_counter()
    end = start + delay_sec

    while time.perf_counter() < end:
        pass
        
import math

mu = math.log(2.5)   # ~2–3 ms median
sigma = 0.4          # low jitter

def bind_socket_to_interface(sock, iface_name):
    """Bind socket to specific interface using SO_BINDTODEVICE."""
    if not hasattr(socket, 'SO_BINDTODEVICE'):
        raise RuntimeError('SO_BINDTODEVICE not available')
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, 
                   iface_name.encode() + b'\0')


def run_ue(ue_index):
    ue_ip = UE_IPS[ue_index]
    iface = UE_IFACES[ue_index]

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # ✅ Step 1: Bind to UE IP
    # sock.bind((ue_ip, 0))

    # ✅ Step 2: Bind to interface (AFTER IP bind)
    try:
    #    pass
        # sock.setsockopt(socket.SOL_SOCKET, 25, iface.encode())
        bind_socket_to_interface(sock,iface)
    except Exception as e:
        print(f"[UE{ue_index+1}] Interface bind failed: {e}")

    print(f"[UE{ue_index+1}] IP={ue_ip}, IFACE={iface}")

    seq = 0

    while time.time() < end_time:

        pkt_size = random.choice(PACKET_SIZES)
        pkt_rate = random.choice(PACKET_RATES)

        interval = 1.0 / pkt_rate

        ts = time.time()

        payload_size = max(0, pkt_size - 12)
        payload = os.urandom(payload_size)

        packet = struct.pack("!Id", seq, ts) + payload
        # 🔥 Random delay between 0–10 ms
        
        delay_ms = random.lognormvariate(mu, sigma)

        # clamp to max 10 ms
        delay_ms = min(delay_ms, 10)

        #if random.random() < 0.9:
    # normal operation (90%)
            #delay_ms = random.gauss(2, 0.5)
        #else:
    # rare spike (10%)
            #delay_ms = random.uniform(5, 10)

        delay_ms = max(0, min(10, delay_ms))
        # busy_wait_delay(delay_ms / 1000.0)


        try:
            sock.sendto(packet, (SERVER_IP, BASE_PORT))
        except Exception as e:
            print(f"[UE{ue_index+1}] Send error: {e}")

        seq += 1
        time.sleep(interval)


threads = []

for i in range(NUM_USERS):
    t = threading.Thread(target=run_ue, args=(i,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
