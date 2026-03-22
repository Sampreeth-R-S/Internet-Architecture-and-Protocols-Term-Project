#!/bin/bash
# Netflix-style 4K Streaming Simulator (5 concurrent users)
# Uses HTTP Range requests over a large media file.

set -u

SERVER_IP="${SERVER_IP:-10.45.0.1}"
SERVER_PORT="${SERVER_PORT:-8080}"
MEDIA_DIR="${MEDIA_DIR:-../data/media}"
MEDIA_FILE="${MEDIA_FILE:-netflix_4k_source.bin}"
MEDIA_SIZE_MB="${MEDIA_SIZE_MB:-768}"

SEGMENT_DURATION=4 # seconds per video chunk
SEGMENT_COUNT=90 # ~6 minutes
THINK_TIME=1 # seconds between chunks

OUTPUT_DIR="../data"
CSV_OUT="$OUTPUT_DIR/netflix_4k_session.csv"
SERVER_LOG="$OUTPUT_DIR/netflix_http_server.log"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$MEDIA_DIR"

INTERFACES=$(ls -1 /sys/class/net/ | grep '^uesimtun' || true)
if [ -z "$INTERFACES" ]; then
 echo "Error: No 'uesimtun' interfaces found. Is UERANSIM running?"
 exit 1
fi

mapfile -t UE_IFACES < <(echo "$INTERFACES" | head -n 5)
if [ "${#UE_IFACES[@]}" -lt 5 ]; then
 echo "Error: Need 5 UE tunnels (uesimtun0..uesimtun4). Found ${#UE_IFACES[@]}"
 exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
 echo "Error: curl is required. Install it and retry."
 exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
 echo "Error: python3 is required. Install it and retry."
 exit 1
fi

MEDIA_PATH="$MEDIA_DIR/$MEDIA_FILE"
TARGET_SIZE_BYTES=$((MEDIA_SIZE_MB * 1024 * 1024))

if [ ! -f "$MEDIA_PATH" ]; then
 echo "Preparing media source file: $MEDIA_PATH (${MEDIA_SIZE_MB}MB)"
else
 echo "Reusing media source file: $MEDIA_PATH"
fi

if ! truncate -s "$TARGET_SIZE_BYTES" "$MEDIA_PATH" 2>/dev/null; then
 echo "Error: Unable to size media file to ${MEDIA_SIZE_MB}MB. Check free disk space."
 exit 1
fi

avail_bytes=$(df --output=avail -B1 "$MEDIA_DIR" | tail -n 1 | tr -d ' ')
if [ -n "$avail_bytes" ] && [ "$avail_bytes" -lt $((64 * 1024 * 1024)) ]; then
 echo "Warning: Less than 64MB free on filesystem hosting $MEDIA_DIR."
 echo "Streaming may fail due to low disk space."
fi

FILE_SIZE=$(stat -c%s "$MEDIA_PATH")
if [ -z "$FILE_SIZE" ] || [ "$FILE_SIZE" -le 0 ]; then
 echo "Error: Could not read media file size."
 exit 1
fi

SERVER_IS_LOCAL=0
if ip -4 addr show | grep -q "inet ${SERVER_IP}/"; then
 SERVER_IS_LOCAL=1
fi

HTTP_SERVER_PID=""
cleanup() {
 if [ -n "$HTTP_SERVER_PID" ]; then
 kill "$HTTP_SERVER_PID" 2>/dev/null || true
 fi
}
trap cleanup EXIT SIGINT SIGTERM

if [ "$SERVER_IS_LOCAL" -eq 1 ]; then
 echo "Starting local HTTP server on ${SERVER_IP}:${SERVER_PORT}"
 python3 -m http.server "$SERVER_PORT" --bind "$SERVER_IP" --directory "$MEDIA_DIR" > "$SERVER_LOG" 2>&1 &
 HTTP_SERVER_PID=$!
 sleep 2
 if ! kill -0 "$HTTP_SERVER_PID" 2>/dev/null; then
 echo "Error: Failed to start local HTTP server. Check $SERVER_LOG"
 exit 1
 fi
else
 echo "Using external HTTP server at ${SERVER_IP}:${SERVER_PORT}"
 echo "Ensure $MEDIA_FILE exists on remote server path."
fi

URL="http://${SERVER_IP}:${SERVER_PORT}/${MEDIA_FILE}"

# ABR-like per-user profiles in Mbps (5 users)
UE0_PROFILE=(24 30 36 28 34 26 38 29 35 27)
UE1_PROFILE=(18 22 27 21 25 19 28 20 24 18)
UE2_PROFILE=(26 33 40 31 37 29 42 32 38 30)
UE3_PROFILE=(16 20 24 18 22 17 25 19 23 16)
UE4_PROFILE=(20 26 32 24 30 22 34 25 31 21)
PROFILE_LEN=${#UE0_PROFILE[@]}

# Per-user file offsets to emulate independent viewers at different play positions.
OFFSETS=(
 0
 $((FILE_SIZE / 5))
 $((FILE_SIZE * 2 / 5))
 $((FILE_SIZE * 3 / 5))
 $((FILE_SIZE * 4 / 5))
)

echo "=== Netflix-style 4K Streaming Started ==="
echo "Target URL: $URL"
echo "UE interfaces: ${UE_IFACES[*]}"
echo "Segment duration: ${SEGMENT_DURATION}s | Segments: ${SEGMENT_COUNT}"
if [ "$SERVER_IS_LOCAL" -eq 1 ]; then
 echo "Mode: Local server (UE bind disabled for local endpoint stability)"
else
 echo "Mode: Remote server (UE interface binding enabled)"
fi
echo ""

echo "timestamp,segment,ue,iface,rate_mbps,bytes,status,duration_ms" > "$CSV_OUT"

for ((segment=0; segment<SEGMENT_COUNT; segment++)); do
 idx=$((segment % PROFILE_LEN))
 rates=("${UE0_PROFILE[$idx]}" "${UE1_PROFILE[$idx]}" "${UE2_PROFILE[$idx]}" "${UE3_PROFILE[$idx]}" "${UE4_PROFILE[$idx]}")

 echo "[$(date '+%Y-%m-%d %H:%M:%S')] Segment $((segment+1))/${SEGMENT_COUNT} | rates: ${rates[*]} Mbps"

 CLIENT_PIDS=()
 for ue in 0 1 2 3 4; do
 iface="${UE_IFACES[$ue]}"
 rate_mbps="${rates[$ue]}"
 bytes=$((rate_mbps * 1000000 / 8 * SEGMENT_DURATION))

 start="${OFFSETS[$ue]}"
 if [ "$start" -ge "$FILE_SIZE" ]; then
 start=$((start % FILE_SIZE))
 fi

 end=$((start + bytes - 1))
 if [ "$end" -ge "$FILE_SIZE" ]; then
 end=$((FILE_SIZE - 1))
 bytes=$((end - start + 1))
 fi

 OFFSETS[$ue]=$((end + 1))
 if [ "${OFFSETS[$ue]}" -ge "$FILE_SIZE" ]; then
 OFFSETS[$ue]=$((OFFSETS[$ue] % FILE_SIZE))
 fi

 (
 ts=$(date +%s)
 t0_ms=$(date +%s%3N)
 status="ok"

 if [ "$SERVER_IS_LOCAL" -eq 1 ]; then
 if ! curl --silent --show-error --fail \
 --max-time $((SEGMENT_DURATION + 8)) \
 --range "${start}-${end}" \
 "$URL" -o /dev/null; then
 status="fail"
 fi
 else
 if ! curl --silent --show-error --fail \
 --max-time $((SEGMENT_DURATION + 8)) \
 --interface "$iface" \
 --range "${start}-${end}" \
 "$URL" -o /dev/null; then
 status="fail"
 fi
 fi

 t1_ms=$(date +%s%3N)
 dur_ms=$((t1_ms - t0_ms))
 echo "${ts},$((segment+1)),$((ue+1)),${iface},${rate_mbps},${bytes},${status},${dur_ms}" >> "$CSV_OUT"
 ) &

 CLIENT_PIDS+=("$!")
 done

 for pid in "${CLIENT_PIDS[@]}"; do
 wait "$pid"
 done

 sleep "$THINK_TIME"
done

echo "=== Streaming Simulation Complete ==="
echo "Session metrics: $CSV_OUT"