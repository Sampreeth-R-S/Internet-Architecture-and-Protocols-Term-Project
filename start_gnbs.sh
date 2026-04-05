#!/usr/bin/env bash
# Start all 3 gNB servers (eMBB, mMTC, URLLC) in the background.
# Usage:  sudo bash start_gnbs.sh
# Stop:   Ctrl+C  (sends SIGINT to all three)

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
NR_GNB="$ROOT/5g-slicing-embb/ueransim-src/build/nr-gnb"

GNB_EMBB="$ROOT/5g-slicing-embb/ueransim/gnb-embb.yaml"
GNB_MMTC="$ROOT/5g-slicing-mmtc/ueransim/gnb-mmtc.yaml"
GNB_URLLC="$ROOT/5g-slicing-urllc/ueransim/gnb-urllc.yaml"

# Verify binary exists
if [[ ! -x "$NR_GNB" ]]; then
    echo "[ERROR] nr-gnb binary not found at $NR_GNB"
    exit 1
fi

cleanup() {
    echo ""
    echo "[*] Stopping all gNB processes..."
    kill -- -$$ 2>/dev/null || true
    wait 2>/dev/null
    echo "[*] All gNBs stopped."
}
trap cleanup EXIT INT TERM

echo "=========================================="
echo "  Starting 3 gNB servers"
echo "=========================================="
echo "  Binary : $NR_GNB"
echo "  eMBB   : $GNB_EMBB"
echo "  mMTC   : $GNB_MMTC"
echo "  URLLC  : $GNB_URLLC"
echo "=========================================="
echo ""

echo "[eMBB]  Launching gNB..."
"$NR_GNB" -c "$GNB_EMBB" &
PID_EMBB=$!

echo "[mMTC]  Launching gNB..."
"$NR_GNB" -c "$GNB_MMTC" &
PID_MMTC=$!

echo "[URLLC] Launching gNB..."
"$NR_GNB" -c "$GNB_URLLC" &
PID_URLLC=$!

echo ""
echo "[*] gNB PIDs — eMBB=$PID_EMBB  mMTC=$PID_MMTC  URLLC=$PID_URLLC"
echo "[*] Press Ctrl+C to stop all."
echo ""

wait
