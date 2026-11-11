#!/usr/bin/env bash
# Stage 4 checkpoint 3: everything the report needs, in one background run on the GPU server.
#   nohup bash scripts/stage4/run_all.sh > run_all.log 2>&1 &
#   RUN_ALL_SMOKE=1 bash scripts/stage4/run_all.sh        # ~20-minute dry run with tiny sizes, separate outputs
# Steps (a failed step is reported and the run continues; finished work is skipped on a rerun):
#   1. l0     pytest tests_stage4
#   2. fp8    fp8_probe.sh: FP8 error probe + calibrated scales + pure-PyTorch format simulation
#   3. eval   eval.sh: GSM8K 1319, needle150, needle600 for BF16 / FP8 e4m3 / FP8 e5m2 / INT8 plain / INT8 fused
#   4. bench  bench.sh: capacity, serving throughput, long-request concurrency, decode-kernel micro-benchmark
#   5. pack   results into one tarball to download
# Usage: bash scripts/stage4/run_all.sh [steps]   (default l0,fp8,eval,bench,pack)
set -uo pipefail
source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"
require_env

STEPS="${1:-l0,fp8,eval,bench,pack}"
MODEL_NAME=Qwen2.5-7B-Instruct
SMOKE="${RUN_ALL_SMOKE:-}"
STAMP=$(date +%Y%m%d_%H%M%S)
LOGS="$RESULTS/run_all${SMOKE:+_smoke}/$STAMP"
mkdir -p "$LOGS"
echo "=== stage 4 run_all (${SMOKE:+smoke, }steps: $STEPS), patch $(git rev-parse --short HEAD), $(date) ==="
echo "logs: $LOGS"

declare -a summary=()
step() {  # step <name> <command...>: run, log, record the outcome, never abort the whole run
    local name=$1; shift
    case ",$STEPS," in *",$name,"*) ;; *) return 0 ;; esac
    echo "##### $name start ($(date +%T))"
    local start=$SECONDS
    if "$@" >"$LOGS/$name.log" 2>&1; then
        summary+=("$name: OK ($(( (SECONDS - start) / 60 )) min)")
    else
        summary+=("$name: FAILED (exit $?, see $LOGS/$name.log)")
    fi
    tail -n 40 "$LOGS/$name.log"
    # Failures may have scrolled out of the tail; repeat them so they are not missed.
    grep -a -E "^FAILED|FAILED |^INCOMPLETE" "$LOGS/$name.log" | sort -u | sed 's/^/!! /'
    echo "##### $name end ($(date +%T))"
}

if [[ -n $SMOKE ]]; then
    step l0 bash scripts/stage4/l0.sh tests_stage4 -x -k "not backend_wiring"
    step fp8 env FP8_SUFFIX=_smoke SIM_GSM8K=3 SIM_ARGS="--needles 2 --max-prompts 2 --max-new-tokens 16 --gsm8k-max-new-tokens 64" \
        bash scripts/stage4/fp8_probe.sh "$MODEL_NAME"
    step eval env EVAL_N=20 bash scripts/stage4/eval.sh
    step bench env BENCH_SMOKE=1 bash scripts/stage4/bench.sh
else
    step l0 bash scripts/stage4/l0.sh tests_stage4
    step fp8 env SIM_GSM8K="${SIM_GSM8K:-100}" bash scripts/stage4/fp8_probe.sh "$MODEL_NAME"
    step eval bash scripts/stage4/eval.sh
    step bench bash scripts/stage4/bench.sh
fi

pack() {
    local tarball="$RESULTS/../stage4_results${SMOKE:+_smoke}_$STAMP.tar.gz"
    git rev-parse HEAD >"$RESULTS/COMMIT"
    nvidia-smi >"$RESULTS/nvidia-smi.txt" 2>&1 || true
    "$PY" -m pip freeze >"$RESULTS/pip-freeze.txt" 2>/dev/null || true
    # pack.log is still being written while tar runs; leave it out so tar does not report a changed file.
    tar -czf "$tarball" --exclude=pack.log -C "$(dirname "$RESULTS")" "$(basename "$RESULTS")" && ls -la "$tarball"
}
step pack pack

echo "===== RUN_ALL SUMMARY ($(date +%T)) ====="
printf '%s\n' ${summary[@]+"${summary[@]}"}
case "${summary[*]-}" in *FAILED*) exit 1 ;; esac
exit 0
