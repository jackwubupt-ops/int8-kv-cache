#!/usr/bin/env bash
# Stage 4 server setup: venv with the pinned SGLang release, patched source on top, models.
# Re-running only re-applies the patches and skips finished steps.
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"

echo "=== GPU / driver ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
VERSION=$(sed -n 's/^version=//p' sglang_patch/BASE_VERSION)

if [[ ! -x $PY ]]; then
    echo "=== venv + sglang ${VERSION#v} ==="
    python3 -m venv "$STAGE4_VENV"
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install "sglang==${VERSION#v}"
    "$PY" -m pip install modelscope
fi

echo "=== patched source ==="
bash sglang_patch/apply.sh third_party
"$PY" -m pip install -e third_party/sglang/python --no-deps
"$PY" -m pip install -e ".[dev]" --no-deps

echo "=== models ==="
for model in Qwen2.5-1.5B-Instruct Qwen2.5-7B-Instruct; do
    if [[ ! -f $MODELS/$model/config.json ]]; then
        "$STAGE4_VENV/bin/modelscope" download --model "Qwen/$model" --local_dir "$MODELS/$model"
    fi
done

require_env
"$PY" -c "import sglang, torch; print('sglang', sglang.__file__); print('torch', torch.__version__, torch.cuda.get_device_name(0))"
echo "Setup complete."
