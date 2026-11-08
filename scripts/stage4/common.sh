# Shared settings for the stage 4 server scripts. Source it; do not run it.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE4_VENV="${STAGE4_VENV:-/root/autodl-tmp/venv-sglang}"
MODELS="${MODELS:-/root/autodl-tmp/models}"
PY="$STAGE4_VENV/bin/python"

# Exit unless the stage 4 venv exists and its installed sglang carries the current overlay.
require_env() {
    if [[ ! -x $PY ]]; then
        echo "No venv at $STAGE4_VENV. Run: bash scripts/stage4/setup.sh" >&2
        exit 1
    fi
    if ! "$PY" "$REPO_ROOT/sglang_patch/overlay.py" verify --checkout "$REPO_ROOT/third_party"; then
        echo "The installed sglang does not match the patched checkout. Re-run: bash scripts/stage4/setup.sh" >&2
        exit 1
    fi
}
