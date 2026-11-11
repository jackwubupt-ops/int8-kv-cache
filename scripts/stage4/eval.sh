#!/usr/bin/env bash
# Stage 4 accuracy on the GPU server: Qwen2.5-7B-Instruct, 5 KV configs x 3 tasks, then a paired summary.
# Usage: bash scripts/stage4/eval.sh [configs] [tasks]
#   configs: comma list of kv_dtype:impl[:attention_backend] (backend defaults to triton)
#   tasks:   comma list (default gsm8k,needle150,needle600)
# Resumable: a (task, config) whose result JSON exists is skipped; delete it to rerun.
# Smoke run: EVAL_N=20 bash scripts/stage4/eval.sh  (first 20 items per task, written to accuracy_smoke20/)
# Run in the background: nohup bash scripts/stage4/eval.sh > eval.log 2>&1 &
set -uo pipefail
source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"
require_env
exec 9>/tmp/int8kv_stage4_test.lock
flock -n 9 || { echo "Another stage 4 GPU job is running; wait for it." >&2; exit 1; }

MODEL="$MODELS/Qwen2.5-7B-Instruct"
CONFIGS="${1:-auto:plain,fp8_e4m3:plain,fp8_e5m2:plain,int8_per_token_head:plain,int8_per_token_head:fused}"
TASKS="${2:-gsm8k,needle150,needle600}"
OUT="$RESULTS/$(basename "$MODEL")/accuracy"
N_ARGS=()
if [[ -n ${EVAL_N:-} ]]; then OUT="${OUT}_smoke$EVAL_N"; N_ARGS=(--n "$EVAL_N"); fi
mkdir -p "$OUT"
echo "=== stage 4 accuracy: $MODEL, patch $(git rev-parse --short HEAD), $(date) ==="

name() {  # name <kv> <impl> <backend> -> config name used in result files (matches eval_accuracy.config_name)
    local n
    case $1 in auto) n=bf16 ;; int8_per_token_head) n="int8_$2" ;; *) n="$1" ;; esac
    [[ $3 == triton ]] && echo "$n" || echo "${n}_$3"
}

failed=0
IFS=, read -ra tasks <<<"$TASKS"
IFS=, read -ra configs <<<"$CONFIGS"
for task in "${tasks[@]}"; do
    for cfg in "${configs[@]}"; do
        IFS=: read -r kv impl backend <<<"$cfg"; backend=${backend:-triton}
        result="$OUT/${task}_$(name "$kv" "$impl" "$backend").json"
        if [[ -f $result ]]; then echo "skip $task $cfg (exists: $result)"; continue; fi
        echo "--- $task $cfg ($(date +%T))"
        if ! "$PY" experiments/stage4/eval_accuracy.py --model "$MODEL" --kv-dtype "$kv" --impl "$impl" --attention-backend "$backend" \
                --task "$task" --out "$OUT" ${N_ARGS[@]+"${N_ARGS[@]}"} 2>&1 | grep -a -E "^EVAL|Error|error:|Traceback"; then
            :
        fi
        [[ -f $result ]] || { echo "FAILED $task $cfg"; failed=1; }
    done
done

echo "===== ACCURACY SUMMARY ($(date +%T)) ====="
"$PY" experiments/stage4/eval_accuracy.py --summarize "$OUT"
for f in "$OUT"/*.json; do
    [[ $(basename "$f") == accuracy_summary.json ]] && continue
    "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(f\"RUN {d['task']:9s} {d['config']:11s} acc={d['accuracy']:.4f} ({d['correct']}/{d['total']}) {d['seconds']:.0f}s chunk={d.get('chunked_prefill_size')}\")" "$f"
done
exit $failed
