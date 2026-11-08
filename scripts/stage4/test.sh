#!/usr/bin/env bash
# Stage 4 correctness checks on the GPU server: L0 pytest, then the L1 model check.
# Usage: bash scripts/stage4/test.sh [l1_configs]
# Prints a SUMMARY block at the end; the full log is in results/stage4/test_<time>.log.
set -uo pipefail
source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"
require_env
# Two runs at once share the GPU and fail each other with out-of-memory errors.
exec 9>/tmp/int8kv_stage4_test.lock
flock -n 9 || { echo "Another stage 4 test.sh is already running; wait for it (check with: ps aux | grep test.sh)." >&2; exit 1; }
"$PY" -c "import pytest" 2>/dev/null || "$PY" -m pip install -q "pytest>=8"

CONFIGS="${1:-auto:plain:on,fp8_e4m3:plain:on,int8_per_token_head:plain:on,int8_per_token_head:plain:off}"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=results/stage4/test_$STAMP
mkdir -p "$OUT"
LOG="$OUT/test.log"

echo "=== L0: pytest tests_stage4 ===" | tee "$LOG"
"$PY" -m pytest -q tests_stage4 -p no:cacheprovider >"$OUT/l0.log" 2>&1
L0=$?
cat "$OUT/l0.log" >>"$LOG"
echo "=== L1: model check ($CONFIGS) ===" | tee -a "$LOG"
"$PY" experiments/stage4/l1_model_check.py --model "$MODELS/Qwen2.5-1.5B-Instruct" \
    --configs "$CONFIGS" --out "$OUT/l1.json" >"$OUT/l1.log" 2>&1
L1=$?
cat "$OUT/l1.log" >>"$LOG"

{
    echo "===== SUMMARY ($STAMP, patch $(git rev-parse --short HEAD)) ====="
    echo "L0 pytest: $([[ $L0 == 0 ]] && echo PASS || echo FAIL) -- $(grep -E '[0-9]+ (passed|failed|skipped|error)' "$OUT/l0.log" | tail -1)"
    if [[ $L0 != 0 ]]; then grep -E "^(FAILED|ERROR) |^E " "$OUT/l0.log" | head -20; tail -n 5 "$OUT/l0.log"; fi
    echo "L1: $([[ $L1 == 0 ]] && echo PASS || echo FAIL)"
    grep -E "^L1 " "$OUT/l1.log"
    [[ $L1 == 0 ]] || grep -vE "it/s\]|s/it\]" "$OUT/l1.log" | tail -n 25
    echo "full log: $LOG"
} | tee "$OUT/summary.txt"
exit $(( L0 || L1 ))
