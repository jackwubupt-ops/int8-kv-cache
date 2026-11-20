#!/usr/bin/env bash
# FP8 study on real activations, in pure PyTorch (no SGLang server):
#   1. fp8_probe.py   per-layer K/V error of FP8 (scale 1.0), calibrated FP8 and INT8; writes calibrated scales
#   2. kv_format_sim.py  model-level comparison of the same formats (top-1, needle, optional GSM8K)
# Usage: bash scripts/stage4/fp8_probe.sh [model_dir_name]   (default Qwen2.5-7B-Instruct)
# Environment: SIM_GSM8K (GSM8K questions per format in step 2, default 0), SIM_ARGS (extra kv_format_sim args)
# Resumable: a step whose output file exists is skipped.
set -uo pipefail
source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"
require_env
exec 9>/tmp/int8kv_stage4_test.lock
flock -n 9 || { echo "Another stage 4 GPU job is running; wait for it." >&2; exit 1; }
NAME="${1:-Qwen2.5-7B-Instruct}"
OUT="$RESULTS/$NAME"
SUFFIX="${FP8_SUFFIX:-}"
PROBE="$OUT/fp8_probe$SUFFIX.json"; CALIB="$OUT/fp8_calib$SUFFIX.json"; SIM="$OUT/format_sim$SUFFIX.json"
mkdir -p "$OUT"
failed=0

if [[ -s $PROBE && -s $CALIB ]]; then
    echo "skip fp8 probe (exists: $PROBE)"
else
    "$PY" experiments/stage4/fp8_probe.py --model "$MODELS/$NAME" --out "$PROBE" --calib-out "$CALIB" 2>&1 \
        | grep -a -E "^PROBE|Error|Traceback"
    [[ -s $PROBE && -s $CALIB ]] || { echo "FAILED fp8 probe"; failed=1; }
fi

if [[ -s $SIM ]] && grep -q '"int8_per_token_head"' "$SIM"; then
    echo "skip format simulation (exists: $SIM)"
elif [[ -s $CALIB ]]; then
    # shellcheck disable=SC2086
    "$PY" experiments/stage4/kv_format_sim.py --model "$MODELS/$NAME" --calib "$CALIB" --gsm8k "${SIM_GSM8K:-0}" \
        ${SIM_ARGS:-} --out "$SIM" 2>&1 | grep -a -E "^SIM|Error|Traceback"
    [[ -s $SIM ]] && grep -q '"int8_per_token_head"' "$SIM" || { echo "FAILED format simulation"; failed=1; }
else
    echo "FAILED format simulation (no calibration file)"; failed=1
fi
exit $failed
