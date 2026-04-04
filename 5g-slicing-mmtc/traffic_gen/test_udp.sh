#!/bin/bash
# Simple test of UDP send/receive for mMTC sensor simulation

echo "Starting UDP listener on port 9999..."
timeout 15 python3 -c "
import socket, time
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 9999))
sock.settimeout(10)
count = 0
try:
    while count < 20:
        data, addr = sock.recvfrom(256)
        count += 1
        print(f'  Received {len(data)}B from {addr}')
except socket.timeout:
    pass
print(f'Total received: {count} packets')
sock.close()
" &
SRV_PID=$!
echo "Listener PID: $SRV_PID"

sleep 2

echo ""
echo "Sending test UDP packets..."
python3 -c "
import socket, struct, time
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
for i in range(20):
    payload = struct.pack('!IIdB', i, i, time.time(), 1) + b'\x00' * 47
    sock.sendto(payload, ('127.0.0.1', 9999))
    time.sleep(0.1)
print(f'Sent 20 test packets (64B each)')
sock.close()
"

wait $SRV_PID 2>/dev/null || true
echo ""
echo "=== UDP test complete ==="
