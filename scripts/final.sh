#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-results-final}"
HERE="$(dirname "$0")"

OUT="$ROOT/short" REPEATS=3 LEVELS=32 PER_USER=3 PRESETS="naive gateway" BURST=0 PROFILE=0 \
  bash "$HERE/run_benchmarks.sh"

OUT="$ROOT/long" REPEATS=3 LEVELS=32 PER_USER=3 MAX_MODEL_LEN=4096 \
  PROMPTS=128,512,1024,2048 OUTPUTS=64,128,256,512 PRESETS="unbucketed gateway" \
  BURST=0 PROFILE=1 TRACE_REQUESTS=0 \
  bash "$HERE/run_benchmarks.sh"

OUT="$ROOT/burst" MAX_MODEL_LEN=4096 PRESETS="" BURST=1 BURST_PRESETS="gateway worstcase unguarded" \
  BURST_BASE=32 BURST_PEAK=128 BURST_PROMPTS=1024,2048,3072 BURST_OUTPUTS=256,512 PROFILE=0 \
  bash "$HERE/run_benchmarks.sh"

python -m bench.aggregate "$ROOT/short"
python -m bench.aggregate "$ROOT/long" --baseline unbucketed
