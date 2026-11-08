#!/usr/bin/env bash
# Runs stages 1-3 and writes results under results/<ModelName>. Run from the repository root.
# Usage: scripts/run_stages.sh [model] [all|1|2|3|gsm8k] [gsm8k_samples]
set -euo pipefail
MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
STAGE="${2:-all}"
GSM_SAMPLES="${3:-200}"
PY=.venv/bin/python
NAME="${MODEL##*/}"

case "$STAGE" in all|1|2|3|gsm8k) ;; *) echo "stage must be all, 1, 2, 3 or gsm8k" >&2; exit 2 ;; esac

run() {
    local script=$1 dir=$2; shift 2
    echo "=== $script ==="
    "$PY" "experiments/$script" --model "$MODEL" --device cuda --out "results/$NAME/$dir" "$@"
}

if [[ $STAGE == all || $STAGE == 1 ]]; then run stage1_kv_distribution.py stage1; fi
if [[ $STAGE == all || $STAGE == 2 ]]; then run stage2_operator.py stage2; fi
if [[ $STAGE == all || $STAGE == 3 ]]; then run stage3_model.py stage3; fi
if [[ $STAGE == all || $STAGE == gsm8k ]]; then run stage3_gsm8k.py stage3 --samples "$GSM_SAMPLES"; fi
echo "Done. Results are in results/$NAME"
