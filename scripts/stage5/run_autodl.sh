#!/usr/bin/env bash
# Stage 5 on an AutoDL GPU box: per_head / per_channel INT8 KV, implementation A vs B, benchmark scores.
# Run from the repository root:
#   bash scripts/stage5/run_autodl.sh setup        # venv, datasets, model (once)
#   bash scripts/stage5/run_autodl.sh check        # 5-minute end-to-end check on 1.5B, a few problems
#   nohup bash scripts/stage5/run_autodl.sh all > stage5.log 2>&1 < /dev/null &
# Steps: setup | check | calibrate | operator | humaneval | aime | gsm8k | all (= calibrate..gsm8k)
# Env: MODEL (default Qwen2.5-7B-Instruct), MODELS, VENV, GSM8K_SAMPLES (default 500), CONFIGS.
set -euo pipefail
cd "$(dirname "$0")/../.."
STEP="${1:-all}"
MODELS="${MODELS:-/root/autodl-tmp/models}"
MODEL_NAME="${MODEL:-Qwen2.5-7B-Instruct}"
MODEL_PATH="$MODELS/$MODEL_NAME"
VENV="${VENV:-/root/autodl-tmp/venv-int8kv}"
PY="$VENV/bin/python"
OUT="results/stage5/$MODEL_NAME"
GSM8K_SAMPLES="${GSM8K_SAMPLES:-500}"
CONFIGS="${CONFIGS:-per_head:dequant:dynamic,per_channel:dequant:dynamic,per_channel:int8:dynamic,per_channel:dequant:static,per_channel:int8:static,per_token_head:dequant:write}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf}"

setup() {
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    if [[ ! -x $PY ]]; then
        # Reuse the image's CUDA PyTorch (>= 2.6) instead of downloading it again.
        if python3 -c "import sys, torch; v = tuple(int(p) for p in torch.__version__.split('+')[0].split('.')[:2]); sys.exit(0 if torch.cuda.is_available() and v >= (2, 6) else 1)"; then
            python3 -m venv --system-site-packages "$VENV"
        else
            echo "The image's PyTorch is missing or older than 2.6; pick an AutoDL image with PyTorch >= 2.6" >&2
            exit 1
        fi
    fi
    "$PY" -m pip install -q -i https://mirrors.aliyun.com/pypi/simple -e ".[dev]" modelscope
    "$PY" -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, torch.cuda.get_device_name(0))"
    "$PY" scripts/stage5/fetch_data.py
    for model in Qwen2.5-1.5B-Instruct "$MODEL_NAME"; do
        if [[ ! -f $MODELS/$model/config.json ]]; then
            "$VENV/bin/modelscope" download --model "Qwen/$model" --local_dir "$MODELS/$model"
        fi
    done
    "$PY" -m pytest -q -p no:warnings tests
}

run() { echo "=== $* ($(date '+%F %T')) ==="; "$PY" "$@" --device cuda; }

calibrate() { run experiments/stage5_calibrate.py --model "$MODEL_PATH" --out "$OUT" --data-file data/gsm8k/train.jsonl --samples 128 --max-new-tokens 512; }
operator() {
    run experiments/stage5_operator.py --model "$MODEL_PATH" --out "$OUT" --scales "$OUT/scales.pt" --max-new-tokens 64
    run experiments/stage5_operator.py --model "$MODEL_PATH" --out "$OUT" --scales "$OUT/scales.pt" --max-new-tokens 64 --quantile-scales \
        --configs per_channel:dequant:static,per_channel:int8:static,per_head:dequant:static,per_head:int8:static
}
evaluate() {
    local task=$1 file=$2; shift 2
    run experiments/stage5_eval.py --task "$task" --model "$MODEL_PATH" --out "$OUT" --scales "$OUT/scales.pt" \
        --data-file "$file" --configs "$CONFIGS" "$@"
}

check() {
    local m="$MODELS/Qwen2.5-1.5B-Instruct" o="results/stage5/check"
    run experiments/stage5_calibrate.py --model "$m" --out "$o" --data-file data/gsm8k/train.jsonl --samples 4 --max-new-tokens 64
    run experiments/stage5_operator.py --model "$m" --out "$o" --scales "$o/scales.pt" --max-new-tokens 8 --max-prompts 1 --needle-facts 20
    for t in humaneval:data/humaneval/test.jsonl aime:data/aime/aime_2022_2024.jsonl gsm8k:data/gsm8k/test.jsonl; do
        run experiments/stage5_eval.py --task "${t%%:*}" --model "$m" --out "$o" --scales "$o/scales.pt" \
            --data-file "${t#*:}" --samples 3 --max-new-tokens 256 --configs per_channel:int8:static
    done
    echo "Check passed. Results in $o (not meaningful: 3 problems each)."
}

case "$STEP" in
    setup) setup ;;
    check) check ;;
    calibrate) calibrate ;;
    operator) operator ;;
    humaneval) evaluate humaneval data/humaneval/test.jsonl ;;
    aime) evaluate aime data/aime/aime_2022_2024.jsonl ;;
    gsm8k) evaluate gsm8k data/gsm8k/test.jsonl --samples "$GSM8K_SAMPLES" ;;
    all) calibrate; operator; evaluate humaneval data/humaneval/test.jsonl;
         evaluate gsm8k data/gsm8k/test.jsonl --samples "$GSM8K_SAMPLES"; evaluate aime data/aime/aime_2022_2024.jsonl ;;
    *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
echo "Done: $STEP ($(date '+%F %T'))"
