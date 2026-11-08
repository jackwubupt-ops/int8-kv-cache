#!/usr/bin/env bash
# Creates .venv on a Linux GPU server. Run from the repository root.
# Reuses the server's preinstalled PyTorch when it already has CUDA and is >= 2.6;
# otherwise installs PyTorch from the CUDA_INDEX wheel index (default cu128).
set -euo pipefail
PYTHON="${PYTHON:-python3}"
CUDA_INDEX="${CUDA_INDEX:-cu128}"

if "$PYTHON" - <<'PY' 2>/dev/null
import sys, torch
major, minor = (int(p) for p in torch.__version__.split(".")[:2])
sys.exit(0 if torch.cuda.is_available() and (major, minor) >= (2, 6) else 1)
PY
then
    echo "Reusing preinstalled PyTorch from $PYTHON"
    "$PYTHON" -m venv --system-site-packages .venv
    .venv/bin/python -m pip install --upgrade pip
else
    echo "Installing PyTorch ($CUDA_INDEX)"
    "$PYTHON" -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install torch --index-url "https://download.pytorch.org/whl/$CUDA_INDEX"
fi

.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -c "import torch, transformers; assert torch.cuda.is_available(), 'CUDA is not available'; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), 'GiB')"
.venv/bin/python -m pytest -q
echo "Setup complete."
