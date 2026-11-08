#!/usr/bin/env bash
# Clone the pinned SGLang base into <target>/sglang-pristine and <target>/sglang,
# then apply sglang_patch/0*.patch on top of <target>/sglang.
# Usage: sglang_patch/apply.sh [target_dir]   (default: third_party)
set -euo pipefail
PATCH_DIR="${SGLANG_PATCH_DIR:-$(cd "$(dirname "$0")" && pwd)}"
REPO_URL="${SGLANG_REPO_URL:-https://github.com/sgl-project/sglang.git}"
TARGET="${1:-third_party}"

VERSION=$(sed -n 's/^version=//p' "$PATCH_DIR/BASE_VERSION")
COMMIT=$(sed -n 's/^commit=//p' "$PATCH_DIR/BASE_VERSION")
[[ -n $VERSION && -n $COMMIT ]] || { echo "BASE_VERSION needs version= and commit= lines" >&2; exit 2; }

checkout() {  # checkout <dir>: fresh clone, or reset an existing one to the base commit
    local dir=$1
    if [[ -d $dir/.git ]]; then
        git -C "$dir" am --abort >/dev/null 2>&1 || true
        if ! git -C "$dir" cat-file -e "$COMMIT^{commit}" 2>/dev/null; then
            git -C "$dir" fetch -q --depth 1 origin "refs/tags/$VERSION:refs/tags/$VERSION"
        fi
        git -C "$dir" checkout -q --detach "$COMMIT"
        git -C "$dir" reset -q --hard "$COMMIT"
        git -C "$dir" clean -qfd
    else
        git clone -q --depth 1 --branch "$VERSION" "$REPO_URL" "$dir"
    fi
    local head
    head=$(git -C "$dir" rev-parse HEAD)
    [[ $head == "$COMMIT" ]] || { echo "$dir is at commit $head, expected $COMMIT for $VERSION" >&2; exit 3; }
}

mkdir -p "$TARGET"
checkout "$TARGET/sglang-pristine"
checkout "$TARGET/sglang"
shopt -s nullglob
patches=("$PATCH_DIR"/0*.patch)
if (( ${#patches[@]} )); then
    # git am records a committer; servers often have no git identity configured.
    git -C "$TARGET/sglang" -c user.name=sglang-patch-apply -c user.email=sglang-patch-apply@localhost \
        am -q --3way "${patches[@]}"
fi
echo "SGLang $VERSION ($COMMIT) with ${#patches[@]} patch(es) at $TARGET/sglang"
