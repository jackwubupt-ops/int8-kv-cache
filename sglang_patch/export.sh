#!/usr/bin/env bash
# Export commits made on top of the base in <target>/sglang to sglang_patch/0*.patch.
# Usage: sglang_patch/export.sh [target_dir]   (default: third_party)
set -euo pipefail
PATCH_DIR="${SGLANG_PATCH_DIR:-$(cd "$(dirname "$0")" && pwd)}"
TARGET="${1:-third_party}"
COMMIT=$(sed -n 's/^commit=//p' "$PATCH_DIR/BASE_VERSION")
[[ -n $COMMIT ]] || { echo "BASE_VERSION needs a commit= line" >&2; exit 2; }

rm -f "$PATCH_DIR"/0*.patch
git -C "$TARGET/sglang" format-patch -q --no-signature "$COMMIT..HEAD" -o "$PATCH_DIR"
ls "$PATCH_DIR"/0*.patch 2>/dev/null || echo "no commits on top of $COMMIT"
