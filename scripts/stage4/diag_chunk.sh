#!/usr/bin/env bash
# Chunked-prefill needle diagnosis on the GPU server (see experiments/stage4/chunk_diag.py).
# Usage: bash scripts/stage4/diag_chunk.sh [facts] [seeds]
set -uo pipefail
source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"
require_env
exec 9>/tmp/int8kv_stage4_test.lock
flock -n 9 || { echo "Another stage 4 GPU job is running; wait for it." >&2; exit 1; }
OUT=results/stage4/diag_chunk_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
"$PY" experiments/stage4/chunk_diag.py --model "$MODELS/Qwen2.5-1.5B-Instruct" --facts "${1:-600}" \
    --seeds "${2:-4}" --out "$OUT/diag.json" 2>&1 | tee "$OUT/diag.log" | grep -a -E "^(DIAG|---|[0-9]+ needles)"
