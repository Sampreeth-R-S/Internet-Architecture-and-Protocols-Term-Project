#!/bin/bash
# Simple test of server+client pattern

echo "Starting server on port 5401..."
timeout 15 iperf3 -s -p 5401 > /tmp/srv.log 2>&1 &
SRV_PID=$!
echo "Server PID: $SRV_PID"

sleep 2
echo "Listening sockets:"
ss -ltnup | grep 5401

echo ""
echo "Attempting client connection..."
timeout 5 iperf3 -c 10.45.0.1 -p 5401 -t 3 -u -b 20M
RC=$?
echo "Client exit code: $RC"

wait $SRV_PID 2>/dev/null || true
echo "Server log:"
cat /tmp/srv.log
