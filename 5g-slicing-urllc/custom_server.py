# custom_udp_server.py

import socket
import struct
import time
from collections import defaultdict

SERVER_IP = "10.45.0.10"
PORT = 5202

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((SERVER_IP, PORT))

print(f"Server listening on {SERVER_IP}:{PORT}")

last_seq = defaultdict(lambda: -1)
received = defaultdict(int)
lost = defaultdict(int)

while True:
    data, addr = sock.recvfrom(4096)

    recv_time = time.time()

    if len(data) < 12:
        continue

    seq, ts = struct.unpack("!Id", data[:12])

    latency_ms = (recv_time - ts) * 1000

    client = addr[0]

    # packet loss tracking
    if last_seq[client] != -1 and seq != last_seq[client] + 1:
        lost[client] += (seq - last_seq[client] - 1)

    last_seq[client] = seq
    received[client] += 1

    print(f"[{client}] seq={seq} latency={latency_ms:.2f} ms "
          f"recv={received[client]} lost={lost[client]}")
