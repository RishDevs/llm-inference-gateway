#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
OUT="${OUT:-results}"
PORT="${PORT:-8000}"
LEVELS="${LEVELS:-1,8,16,32}"
PER_USER="${PER_USER:-6}"
PRESETS="${PRESETS:-naive unbucketed gateway}"
BURST="${BURST:-1}"
PROFILE="${PROFILE:-1}"
BURST_PRESETS="${BURST_PRESETS:-gateway unguarded}"
SERVER_ARGS=()
if [[ -n "${MAX_MODEL_LEN:-}" ]]; then
  SERVER_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
fi
DEVICE_ARGS=()
if [[ -n "${DEVICE:-}" ]]; then
  DEVICE_ARGS=(--device "$DEVICE")
fi
URL="http://localhost:${PORT}"

mkdir -p "$OUT/logs"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

wait_up() {
  for _ in $(seq 1 180); do
    if curl -sf "$URL/health" > /dev/null; then
      return 0
    fi
    sleep 2
  done
  echo "server did not come up, see $1" >&2
  return 1
}

serve() {
  local preset="$1"
  if curl -sf "$URL/health" > /dev/null; then
    echo "something is already listening on $URL, stop it first" >&2
    exit 1
  fi
  python -m gateway.server --preset "$preset" --model "$MODEL" --port "$PORT" "${DEVICE_ARGS[@]}" "${SERVER_ARGS[@]}" \
    > "$OUT/logs/server-$preset.log" 2>&1 &
  SERVER_PID=$!
  wait_up "$OUT/logs/server-$preset.log"
}

halt() {
  kill "$SERVER_PID" 2> /dev/null || true
  wait "$SERVER_PID" 2> /dev/null || true
  sleep 3
}
trap 'kill ${SERVER_PID:-0} 2> /dev/null || true' EXIT

for preset in $PRESETS; do
  echo "== steady sweep: $preset"
  serve "$preset"
  python -m bench.load_test --url "$URL" --model "$MODEL" --label "$preset" \
    --concurrency "$LEVELS" --requests-per-user "$PER_USER" \
    --prompt-buckets "${PROMPTS:-64,128,256,512}" --output-buckets "${OUTPUTS:-32,64,128,256}" \
    --out "$OUT/$preset.json"
  halt
done

if [[ "$BURST" == "1" ]]; then
  for preset in $BURST_PRESETS; do
    echo "== burst: $preset"
    serve "$preset"
    python -m bench.load_test --url "$URL" --model "$MODEL" --label "burst-$preset" --burst \
      --base "${BURST_BASE:-32}" --peak "${BURST_PEAK:-64}" --phases "${PHASES:-30,60,30}" \
      --prompt-buckets "${BURST_PROMPTS:-256,512,1024,1536}" --output-buckets "${BURST_OUTPUTS:-64,128,256,384}" \
      --out "$OUT/burst-$preset.json"
    halt
  done
fi

if [[ "$PROFILE" == "1" ]]; then
  echo "== profiler"
  python -m bench.profile_decode --model "$MODEL" "${DEVICE_ARGS[@]}" --trace-requests 6 --out "$OUT/profile"
fi

python -m bench.report "$OUT"
