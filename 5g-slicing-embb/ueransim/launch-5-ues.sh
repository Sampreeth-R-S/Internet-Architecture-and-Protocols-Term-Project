#!/bin/bash
#
# Launch 5 eMBB UEs in parallel
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../ueransim-src"

if [ ! -x "$BUILD_DIR/build/nr-ue" ]; then
    echo "ERROR: nr-ue binary not found at $BUILD_DIR/build/nr-ue"
    exit 1
fi

echo "=========================================="
echo "  Launching 5 eMBB UEs"
echo "=========================================="
echo ""

cd "$BUILD_DIR"

# Launch each UE in background
for i in {1..5}; do
    config="$SCRIPT_DIR/ue-embb-00${i}.yaml"
    if [ ! -f "$config" ]; then
        echo "WARNING: Config not found: $config"
        continue
    fi
    echo "  [UE-$i] Starting with config: $(basename $config)"
    sudo ./build/nr-ue -c "$config" &
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
