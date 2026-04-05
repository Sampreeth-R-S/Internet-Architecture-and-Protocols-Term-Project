#!/bin/bash
# Launch 5 URLLC UEs in parallel

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use shared UERANSIM build from eMBB slice.
BUILD_DIR="/home/sam/Desktop/Internet-Architecture-and-Protocols-Term-Project/5g-slicing-embb/ueransim-src"
NR_UE_BIN="$BUILD_DIR/build/nr-ue"

if [ ! -x "$NR_UE_BIN" ]; then
    echo "ERROR: nr-ue binary not found or not executable: $NR_UE_BIN"
    exit 1
fi

echo "=========================================="
echo "  Launching 5 URLLC UEs"
echo "=========================================="
echo ""

cd "$BUILD_DIR"

# Launch each UE in background
for i in {1..5}; do
    config="$SCRIPT_DIR/ue-urllc${i}.yaml"
    if [ ! -f "$config" ]; then
        echo "WARNING: Config not found: $config"
        continue
    fi
    echo "  [UE-$i] Starting with config: $(basename "$config")"
    sudo "$NR_UE_BIN" -c "$config" &
    PIDS[$i]=$!
    sleep 0.5
done

echo ""
echo "All UEs started. PIDs: ${PIDS[@]}"
echo ""
echo "To stop all UEs, run:"
echo "  sudo killall nr-ue"
echo ""

# Wait for all background processes
wait
