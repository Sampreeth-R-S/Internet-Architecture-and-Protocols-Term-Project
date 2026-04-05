#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_unified_controller.sh
# Launches unified_controller.py in LIVE mode.
#
# Each slice controller polls its own CSV file (written by the traffic
# generators / custom_client.py) and pushes QoS updates to Open5GS.
#
# Usage:
#   chmod +x run_unified_controller.sh
#   ./run_unified_controller.sh
#
# Optional env overrides (export before running):
#   URLLC_DATA   – path to URLLC rolling CSV  (latency_ms column)
#   MMTC_DATA    – path to mMTC  rolling CSV  (packet_rate column)
#   EMBB_DATA    – path to eMBB  rolling CSV  (throughput_mbps column)
#   URLLC_MODEL  – path to trained URLLC .pth
#   MMTC_MODEL   – path to trained mMTC  .pth
#   EMBB_MODEL   – path to trained eMBB  .pth
#   PYTHON       – python interpreter to use  (default: python3)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Resolve project root (directory containing this script) ──────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults (relative to project root) ─────────────────────────────────────
URLLC_DATA="${URLLC_DATA:-${SCRIPT_DIR}/5g-slicing-urllc/data/urllc_timeseries.csv}"
MMTC_DATA="${MMTC_DATA:-${SCRIPT_DIR}/5g-slicing-mmtc/data/mmtc_traffic_timeseries.csv}"
EMBB_DATA="${EMBB_DATA:-${SCRIPT_DIR}/5g-slicing-embb/data/embb_traffic_timeseries.csv}"

URLLC_MODEL="${URLLC_MODEL:-${SCRIPT_DIR}/5g-slicing-urllc/saved/lstm_urllc.pth}"
MMTC_MODEL="${MMTC_MODEL:-${SCRIPT_DIR}/5g-slicing-mmtc/models/saved/lstm_mmtc.pth}"
EMBB_MODEL="${EMBB_MODEL:-${SCRIPT_DIR}/5g-slicing-embb/models/saved/lstm_embb.pth}"

PYTHON="${PYTHON:-python3}"
CONTROLLER="${SCRIPT_DIR}/unified_controller.py"

# ── Pre-flight checks ────────────────────────────────────────────────────────
echo "============================================================"
echo "  Unified Zero-Touch 5G Slice Controller  —  LIVE MODE"
echo "============================================================"

if [[ ! -f "${CONTROLLER}" ]]; then
    echo "[ERROR] Controller not found: ${CONTROLLER}"
    exit 1
fi

if ! command -v "${PYTHON}" &>/dev/null; then
    echo "[ERROR] Python interpreter not found: ${PYTHON}"
    echo "        Set the PYTHON env variable to the correct path."
    exit 1
fi

# Warn if a model file is missing (controller will also error, but this gives
# an early, readable message).
for label_model in "URLLC:${URLLC_MODEL}" "mMTC:${MMTC_MODEL}" "eMBB:${EMBB_MODEL}"; do
    label="${label_model%%:*}"
    model="${label_model##*:}"
    if [[ ! -f "${model}" ]]; then
        echo "[WARN]  ${label} model not found: ${model}"
        echo "        Train the model first (run lstm_predictor.py in the slice directory)."
    fi
done

# Warn if a live CSV is not yet present (expected — traffic generators create it)
for label_csv in "URLLC:${URLLC_DATA}" "mMTC:${MMTC_DATA}" "eMBB:${EMBB_DATA}"; do
    label="${label_csv%%:*}"
    csvfile="${label_csv##*:}"
    if [[ ! -f "${csvfile}" ]]; then
        echo "[WARN]  ${label} data file not yet present: ${csvfile}"
        echo "        Make sure the traffic generator is running and writing to that path."
    fi
done

echo ""
echo "  Python      : ${PYTHON}"
echo "  Controller  : ${CONTROLLER}"
echo ""
echo "  Slice   SST  Data file"
echo "  ------  ---  ---------"
echo "  eMBB    1    ${EMBB_DATA}"
echo "  URLLC   2    ${URLLC_DATA}"
echo "  mMTC    3    ${MMTC_DATA}"
echo ""
echo "  Models"
echo "  ------  -----------------------------------------------"
echo "  URLLC   ${URLLC_MODEL}"
echo "  mMTC    ${MMTC_MODEL}"
echo "  eMBB    ${EMBB_MODEL}"
echo ""
echo "  Output JSON files will be written to:"
echo "    ${SCRIPT_DIR}/unified_decisions_urllc.json"
echo "    ${SCRIPT_DIR}/unified_decisions_mmtc.json"
echo "    ${SCRIPT_DIR}/unified_decisions_embb.json"
echo ""
echo "  Press Ctrl+C to stop all three slice controllers."
echo "============================================================"
echo ""

# ── Launch ───────────────────────────────────────────────────────────────────
exec "${PYTHON}" "${CONTROLLER}" \
    --mode       live            \
    --urllc-data "${URLLC_DATA}" \
    --mmtc-data  "${MMTC_DATA}"  \
    --embb-data  "${EMBB_DATA}"  \
    --urllc-model "${URLLC_MODEL}" \
    --mmtc-model  "${MMTC_MODEL}"  \
    --embb-model  "${EMBB_MODEL}"
