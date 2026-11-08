#!/usr/bin/env bash
# Stage 4 server setup: venv with the pinned SGLang release, patched source on top, models.
# Re-running only re-applies the patches and skips finished steps.
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"

echo "=== GPU / driver ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
VERSION=$(sed -n 's/^version=//p' sglang_patch/BASE_VERSION)
TORCH_VERSION=2.11.0
KERNEL_VERSION=0.4.2.post2

# GitHub needs AutoDL's academic proxy; pip mirrors are faster without it.
with_proxy() { ( if [[ -f /etc/network_turbo ]]; then source /etc/network_turbo >/dev/null; fi; "$@" ) }

if [[ ! -x $PY ]]; then
    echo "=== venv + sglang ${VERSION#v} ==="
    # Driver 570 tops out at CUDA 12.8: reuse the image's cu128 torch when it is the pinned version.
    if python3 -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('$TORCH_VERSION') and torch.cuda.is_available() else 1)" 2>/dev/null; then
        python3 -m venv --system-site-packages "$STAGE4_VENV"
    else
        python3 -m venv "$STAGE4_VENV"
        "$PY" -m pip install "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cu128
    fi
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install "sglang==${VERSION#v}" modelscope
    # PyPI ships the CUDA 13 build of sglang-kernel; driver 570 needs the cu129 build.
    with_proxy "$PY" -m pip install --force-reinstall --no-deps "sglang-kernel==$KERNEL_VERSION+cu129" \
        --find-links https://sgl-project.github.io/whl/cu129/sglang-kernel/
fi

echo "=== pinned dependencies (checked on every run) ==="
# pip can replace the cu128 torch with the PyPI build (CUDA 13), which driver 570 cannot run.
if [[ $("$PY" -c "import torch; print(torch.__version__)" 2>/dev/null) != "$TORCH_VERSION+cu128" ]]; then
    "$PY" -m pip install "torch==$TORCH_VERSION+cu128" "torchaudio==$TORCH_VERSION+cu128" \
        "torchvision==0.26.0+cu128" --index-url https://download.pytorch.org/whl/cu128
fi
# sglang leaves `kernels` unpinned; transformers 5.6.0 only works with kernels 0.12.x.
"$PY" -m pip install "kernels>=0.12,<0.13" ninja
command -v nvcc >/dev/null || { echo "nvcc not found under $CUDA_HOME/bin; SGLang needs the CUDA toolkit to JIT-compile kernels" >&2; exit 1; }
nvcc --version | tail -n 2
"$PY" -c "import torch, kernels; print('torch', torch.__version__, 'cuda ok:', torch.cuda.is_available(), '| kernels', kernels.__version__)"

echo "=== patched source ==="
with_proxy bash sglang_patch/apply.sh third_party
# The source tree needs Rust to build; overlay the patched .py files onto the installed wheel instead.
"$PY" sglang_patch/overlay.py apply --checkout third_party
"$PY" -m pip install -e ".[dev]" --no-deps
"$PY" -m pip install "pytest>=8"

echo "=== models ==="
for model in Qwen2.5-1.5B-Instruct Qwen2.5-7B-Instruct; do
    if [[ ! -f $MODELS/$model/config.json ]]; then
        "$STAGE4_VENV/bin/modelscope" download --model "Qwen/$model" --local_dir "$MODELS/$model"
    fi
done

require_env
echo "=== smoke test (triton backend, bf16 and fp8 KV) ==="
"$PY" experiments/stage4/smoke.py --model "$MODELS/Qwen2.5-1.5B-Instruct"
echo "Setup complete."
