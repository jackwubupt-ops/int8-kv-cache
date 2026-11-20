#!/usr/bin/env bash
# Run L0 pytest on the GPU server with the stage 4 environment (venv python, ninja and nvcc on PATH).
# Usage: bash scripts/stage4/l0.sh [pytest args]   (default: tests_stage4)
# Calling the venv python directly skips common.sh: SGLang then cannot JIT-compile kernels (no ninja),
# its scheduler child dies, and the parent is killed with SIGQUIT (exit 137), which looks like an OOM kill.
set -uo pipefail
source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"
require_env
exec 9>/tmp/int8kv_stage4_test.lock
flock -n 9 || { echo "Another stage 4 GPU job is running; wait for it." >&2; exit 1; }
"$PY" -m pytest -q -p no:cacheprovider "${@:-tests_stage4}"
