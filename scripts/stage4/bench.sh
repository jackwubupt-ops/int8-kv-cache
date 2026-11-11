#!/usr/bin/env bash
# Stage 4 performance on the GPU server: Qwen2.5-7B-Instruct, KV capacity, serving throughput, long-request
# concurrency and a decode-kernel micro-benchmark.
# Usage: bash scripts/stage4/bench.sh [configs]
#   configs: comma list of kv_dtype:impl:attention_backend:mode, mode = full | light
#     full  = capacity + serving (1024,256) (4096,512) (8192,512) x RUNS + long-request test
#     light = capacity + serving (1024,256) x RUNS
#   default: BF16, FP8 e4m3, INT8 fused (full), INT8 plain (light), BF16 on FlashInfer (full, reference)
# Environment: MEM_FRACTION (0.85), RUNS (3), LONG_REQUESTS (128), PORT (30000), BENCH_SMOKE=1 (tiny, to bench_smoke/)
# Resumable: finished runs are skipped; delete a run's .jsonl (or the config directory) to redo it.
# Run in the background: nohup bash scripts/stage4/bench.sh > bench.log 2>&1 &
set -uo pipefail
source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"
require_env
exec 9>/tmp/int8kv_stage4_test.lock
flock -n 9 || { echo "Another stage 4 GPU job is running; wait for it." >&2; exit 1; }
# The server and the load generator talk over localhost; a proxy in between breaks that.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost

MODEL="$MODELS/Qwen2.5-7B-Instruct"
CONFIGS="${1:-auto:plain:triton:full,fp8_e4m3:plain:triton:full,int8_per_token_head:fused:triton:full,int8_per_token_head:plain:triton:light,auto:plain:flashinfer:full}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
RUNS="${RUNS:-3}"
LONG_REQUESTS="${LONG_REQUESTS:-128}"
PORT="${PORT:-30000}"
OUT="$RESULTS/$(basename "$MODEL")/bench"
# shape = input_len:output_len:num_prompts
SHAPES_FULL=(1024:256:256 4096:512:64 8192:512:32)
SHAPES_LIGHT=(1024:256:256)
LONG_SHAPE="8192:512:$LONG_REQUESTS"
KERNEL_CONTEXTS="1024,4096,16384,32768"
KERNEL_ITERS=100
if [[ -n ${BENCH_SMOKE:-} ]]; then
    OUT="${OUT}_smoke"; RUNS=1
    SHAPES_FULL=(1024:64:16); SHAPES_LIGHT=(1024:64:16); LONG_SHAPE="4096:64:24"
    KERNEL_CONTEXTS="1024,4096"; KERNEL_ITERS=20
fi
mkdir -p "$OUT"
echo "=== stage 4 bench: $MODEL, patch $(git rev-parse --short HEAD), mem_fraction=$MEM_FRACTION, $(date) ==="

name() {  # name <kv> <impl> <backend> -> config name (matches eval_accuracy.config_name)
    local n
    case $1 in auto) n=bf16 ;; int8_per_token_head) n="int8_$2" ;; *) n="$1" ;; esac
    [[ $3 == triton ]] && echo "$n" || echo "${n}_$3"
}

SERVER_PID=""
stop_server() {
    [[ -n $SERVER_PID ]] || return 0
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    SERVER_PID=""
    # Wait until the GPU memory is back before the next configuration starts.
    for _ in $(seq 1 60); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        [[ -z $used || $used -lt 1500 ]] && break
        sleep 2
    done
}
trap stop_server EXIT

start_server() {  # start_server <kv> <impl> <backend> <dir>; leaves SERVER_PID set, returns 1 on failure
    local kv=$1 impl=$2 backend=$3 dir=$4
    SGLANG_INT8_KV_IMPL="$impl" setsid "$PY" -m sglang.launch_server --model-path "$MODEL" \
        --attention-backend "$backend" --kv-cache-dtype "$kv" --mem-fraction-static "$MEM_FRACTION" \
        --host 127.0.0.1 --port "$PORT" >"$dir/server.log" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 600); do  # up to 20 minutes: weight loading plus CUDA graph capture
        if grep -a -q "ready to roll" "$dir/server.log"; then return 0; fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "server exited during startup:"; tail -n 15 "$dir/server.log"; SERVER_PID=""; return 1
        fi
        sleep 2
    done
    echo "server not ready after 20 minutes:"; tail -n 15 "$dir/server.log"
    stop_server
    return 1
}

run_bench() {  # run_bench <out.jsonl> <input_len> <output_len> <num_prompts> <seed>
    local out=$1
    rm -f "$out.tmp"
    "$PY" -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port "$PORT" --model "$MODEL" \
        --dataset-name random-ids --tokenize-prompt --random-input-len "$2" --random-output-len "$3" \
        --random-range-ratio 1.0 --num-prompts "$4" --request-rate inf --seed "$5" --flush-cache \
        --disable-tqdm --output-file "$out.tmp" >"${out%.jsonl}.log" 2>&1
    if [[ -s $out.tmp ]] && grep -q '"completed"' "$out.tmp"; then mv "$out.tmp" "$out"; return 0; fi
    echo "  bench_serving failed, see ${out%.jsonl}.log:"; tail -n 5 "${out%.jsonl}.log"
    return 1
}

failed=0

echo "--- decode kernel micro-benchmark ($(date +%T))"
for dtype in bf16 fp8 int8 int8_plain; do
    kernel_out="$OUT/decode_kernel_$dtype.json"
    if [[ -s $kernel_out ]]; then echo "skip kernel $dtype (exists)"; continue; fi
    "$PY" experiments/stage4/bench_decode_kernel.py --dtype "$dtype" --contexts "$KERNEL_CONTEXTS" \
        --iters "$KERNEL_ITERS" --out "$kernel_out" 2>&1 | grep -a -E "^KERNEL|Error|Traceback"
    [[ -s $kernel_out ]] || { echo "FAILED kernel $dtype"; failed=1; }
done

IFS=, read -ra configs <<<"$CONFIGS"
for cfg in "${configs[@]}"; do
    IFS=: read -r kv impl backend mode <<<"$cfg"
    cname=$(name "$kv" "$impl" "$backend")
    dir="$OUT/$cname"
    mkdir -p "$dir"
    if [[ $mode == light ]]; then shapes=("${SHAPES_LIGHT[@]}"); long=""; else shapes=("${SHAPES_FULL[@]}"); long=$LONG_SHAPE; fi

    pending=()
    for shape in "${shapes[@]}"; do
        IFS=: read -r in_len out_len num <<<"$shape"
        for run in $(seq 1 "$RUNS"); do
            f="$dir/serving_${in_len}x${out_len}_run$run.jsonl"
            [[ -s $f ]] || pending+=("serving:$in_len:$out_len:$num:$run")
        done
    done
    if [[ -n $long ]]; then
        IFS=: read -r in_len out_len num <<<"$long"
        [[ -s $dir/long_${in_len}x${out_len}_run1.jsonl ]] || pending+=("long:$in_len:$out_len:$num:1")
    fi
    if (( ${#pending[@]} == 0 )) && grep -a -q "max_total_num_tokens=" "$dir/server.log" 2>/dev/null; then
        echo "skip $cname (all runs exist)"; continue
    fi

    echo "--- $cname: starting server ($(date +%T))"
    # Keep the log of a finished earlier session; this session's log replaces it only once the server is up.
    [[ -f $dir/server.log ]] && mv "$dir/server.log" "$dir/server.prev.log"
    if ! start_server "$kv" "$impl" "$backend" "$dir"; then echo "FAILED $cname (server)"; failed=1; continue; fi
    grep -a -o "max_total_num_tokens=[0-9]*" "$dir/server.log" | head -1
    for item in ${pending[@]+"${pending[@]}"}; do
        IFS=: read -r kind in_len out_len num run <<<"$item"
        f="$dir/${kind}_${in_len}x${out_len}_run$run.jsonl"
        start_bytes=$(wc -c <"$dir/server.log")
        echo "  $kind ${in_len}x${out_len} n=$num run $run ($(date +%T))"
        if ! run_bench "$f" "$in_len" "$out_len" "$num" "$run"; then echo "FAILED $cname $kind ${in_len}x${out_len}"; failed=1; fi
        if [[ $kind == long ]]; then
            tail -c "+$((start_bytes + 1))" "$dir/server.log" >"$dir/long_${in_len}x${out_len}_server.log"
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "FAILED $cname (server died)"; tail -n 15 "$dir/server.log"; failed=1; break; fi
    done
    stop_server
done

echo "===== BENCH SUMMARY ($(date +%T)) ====="
"$PY" experiments/stage4/parse_logs.py "$OUT" || failed=1
exit $failed
