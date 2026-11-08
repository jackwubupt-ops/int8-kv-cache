# Shared settings for the stage 4 server scripts. Source it; do not run it.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE4_VENV="${STAGE4_VENV:-/root/autodl-tmp/venv-sglang}"
MODELS="${MODELS:-/root/autodl-tmp/models}"
PY="$STAGE4_VENV/bin/python"

# Exit unless the stage 4 venv exists and imports the patched SGLang checkout.
require_env() {
    if [[ ! -x $PY ]]; then
        echo "No venv at $STAGE4_VENV. Run: bash scripts/stage4/setup.sh" >&2
        exit 1
    fi
    local loaded
    loaded=$("$PY" -c "import os, sglang; print(os.path.realpath(sglang.__file__))" 2>/dev/null || true)
    if [[ $loaded != "$REPO_ROOT/third_party/sglang/"* ]]; then
        echo "sglang is loaded from '$loaded', expected the patched checkout under $REPO_ROOT/third_party/sglang/. Re-run setup.sh." >&2
        exit 1
    fi
}
