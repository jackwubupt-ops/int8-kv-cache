#!/usr/bin/env bash
# Per-layer K/V quantization error of FP8 (scale 1.0) vs INT8 per-token-head on real activations.
# Usage: bash scripts/stage4/fp8_probe.sh [model_dir_name]   (default Qwen2.5-7B-Instruct)
set -uo pipefail
source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"
require_env
exec 9>/tmp/int8kv_stage4_test.lock
flock -n 9 || { echo "Another stage 4 GPU job is running; wait for it." >&2; exit 1; }
NAME="${1:-Qwen2.5-7B-Instruct}"
"$PY" experiments/stage4/fp8_probe.py --model "$MODELS/$NAME" --out "results/stage4/$NAME/fp8_probe.json" 2>&1 \
    | grep -a -E "^PROBE|Error|Traceback"
