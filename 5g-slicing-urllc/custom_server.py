# custom_udp_server.py

import socket
import struct
import time
from collections import defaultdict

<<<<<<< HEAD
SERVER_IP = "10.45.0.10"
=======
SERVER_IP = "10.46.0.10"
>>>>>>> d7e5ce9f407c4039a9955cd4e8aa4162b84a16e1
PORT = 5202

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((SERVER_IP, PORT))

<<<<<<< HEAD
print(f"Server listening on {SERVER_IP}:{PORT}")
=======
# print(f"Server listening on {SERVER_IP}:{PORT}")
>>>>>>> d7e5ce9f407c4039a9955cd4e8aa4162b84a16e1

last_seq = defaultdict(lambda: -1)
received = defaultdict(int)
lost = defaultdict(int)

<<<<<<< HEAD
=======
print(f"ip,seq,latency_ms,recv,lost")
>>>>>>> d7e5ce9f407c4039a9955cd4e8aa4162b84a16e1
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

<<<<<<< HEAD
    print(f"[{client}] seq={seq} latency={latency_ms:.2f} ms "
          f"recv={received[client]} lost={lost[client]}")
=======
    print(f"{client},{seq},{latency_ms:.2f},{received[client]},{lost[client]}")
>>>>>>> d7e5ce9f407c4039a9955cd4e8aa4162b84a16e1
