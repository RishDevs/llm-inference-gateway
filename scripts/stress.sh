#!/usr/bin/env bash
set -euo pipefail

export OUT="${OUT:-results-stress}"
export MAX_MODEL_LEN=4096
export PRESETS="unbucketed gateway"
export LEVELS=32
export PER_USER="${PER_USER:-4}"
export PROMPTS=128,512,1024,2048
export OUTPUTS=64,128,256,512
export BURST=1
export BURST_BASE=32
export BURST_PEAK=128
export BURST_PROMPTS=1024,2048,3072
export BURST_OUTPUTS=256,512
export PROFILE="${PROFILE:-1}"

exec bash "$(dirname "$0")/run_benchmarks.sh"
