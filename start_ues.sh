#!/usr/bin/env bash
# Start all UEs for all 3 slices (eMBB, mMTC, URLLC).
# Usage:  sudo bash start_ues.sh
# Stop:   Ctrl+C  (or: sudo killall nr-ue)

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
NR_UE="$ROOT/5g-slicing-embb/ueransim-src/build/nr-ue"
MMTC_UE_YAML="$ROOT/5g-slicing-mmtc/ueransim/ue-mmtc.yaml"

if [[ ! -x "$NR_UE" ]]; then
    echo "[ERROR] nr-ue binary not found at $NR_UE"
    exit 1
fi

echo "=========================================="
echo "  Starting all UEs (eMBB + mMTC + URLLC)"
echo "=========================================="
echo ""

# ── eMBB UEs (5) ────────────────────────────────────────────
echo "── Launching 5 eMBB UEs ──"
bash "$ROOT/5g-slicing-embb/ueransim/launch-5-ues.sh" &
PID_EMBB=$!
sleep 3

# ── URLLC UEs (5) ───────────────────────────────────────────
echo ""
echo "── Launching 5 URLLC UEs ──"
bash "$ROOT/5g-slicing-urllc/ueransim/launch-5-ues.sh" &
PID_URLLC=$!
sleep 3

# ── mMTC UE (1) ─────────────────────────────────────────────
echo ""
echo "── Launching mMTC UE ──"
cd "$ROOT/5g-slicing-embb/ueransim-src"
sudo "$NR_UE" -c "$MMTC_UE_YAML" &
PID_MMTC=$!

echo ""
echo "=========================================="
echo "  All UEs launched"
echo "  eMBB  script PID : $PID_EMBB"
echo "  URLLC script PID : $PID_URLLC"
echo "  mMTC  UE PID     : $PID_MMTC"
echo "=========================================="
echo ""
echo "Press Ctrl+C to stop all, or run:  sudo killall nr-ue"

wait
