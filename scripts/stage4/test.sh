#!/usr/bin/env bash
# Stage 4 correctness checks on the GPU server: L0 pytest, then the L1 model check.
# Usage: bash scripts/stage4/test.sh [l1_configs]
# Prints a SUMMARY block at the end; the full log is in results/stage4/test_<time>.log.
set -uo pipefail
source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"
require_env

CONFIGS="${1:-int8_per_token_head:plain:on,int8_per_token_head:plain:off}"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=results/stage4/test_$STAMP
mkdir -p "$OUT"
LOG="$OUT/test.log"

echo "=== L0: pytest tests_stage4 ===" | tee "$LOG"
"$PY" -m pytest -q tests_stage4 -p no:cacheprovider >>"$LOG" 2>&1
L0=$?
echo "=== L1: model check ($CONFIGS) ===" | tee -a "$LOG"
"$PY" experiments/stage4/l1_model_check.py --model "$MODELS/Qwen2.5-1.5B-Instruct" \
    --configs "$CONFIGS" --out "$OUT/l1.json" >>"$LOG" 2>&1
L1=$?

{
    echo "===== SUMMARY ($STAMP, patch $(git rev-parse --short HEAD)) ====="
    echo "L0 pytest: $([[ $L0 == 0 ]] && echo PASS || echo FAIL) -- $(grep -E '[0-9]+ (passed|failed)' "$LOG" | tail -1)"
    grep -E "^(FAILED|ERROR) " "$LOG" | head -20
    echo "L1: $([[ $L1 == 0 ]] && echo PASS || echo FAIL)"
    grep -E "^L1 " "$LOG"
    [[ $L1 == 0 ]] || grep -E "Error|error:|Traceback" "$LOG" | tail -5
    echo "full log: $LOG"
} | tee "$OUT/summary.txt"
exit $(( L0 || L1 ))
