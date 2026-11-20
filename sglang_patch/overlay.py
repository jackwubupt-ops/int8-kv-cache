"""Overlay the patched SGLang .py files onto the installed sglang wheel.

The v0.5.12 source tree needs a Rust toolchain to build, so instead of an editable
install we copy only the files our patches touch into the installed package and
record them in a manifest. Re-applying first restores files that are no longer
patched from the pristine checkout. The manifest also stores a digest of
sglang_patch/0*.patch, so `verify` notices patches pulled after the last setup.sh.

Usage (run with the stage 4 venv python):
  python sglang_patch/overlay.py apply  [--checkout third_party]
  python sglang_patch/overlay.py verify [--checkout third_party]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

MANIFEST = ".int8kv_overlay.json"
PKG_PREFIX = "python/sglang/"
PATCH_DIR = Path(__file__).resolve().parent


def package_relative(names: Sequence[str]) -> List[str]:
    return [n[len(PKG_PREFIX):] for n in names if n.startswith(PKG_PREFIX)]


def patches_digest(patch_dir: Path = PATCH_DIR) -> str:
    """sha256 over the names and contents of patch_dir/0*.patch, in order."""
    h = hashlib.sha256()
    for path in sorted(Path(patch_dir).glob("0*.patch")):
        h.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def _read_manifest(site_pkg: Path) -> dict:
    """{"files": [...], "patches": digest or None}; manifests from before the digest were a bare file list."""
    path = site_pkg / MANIFEST
    if not path.exists():
        return {"files": [], "patches": None}
    data = json.loads(path.read_text())
    return {"files": data, "patches": None} if isinstance(data, list) else data


def apply_overlay(*, patched_root: Path, pristine_root: Path, site_pkg: Path, changed: Sequence[str],
                  patches_digest: Optional[str] = None) -> None:
    for rel in set(_read_manifest(site_pkg)["files"]) - set(changed):
        original = pristine_root / rel
        if original.exists():
            shutil.copyfile(original, site_pkg / rel)
        else:
            (site_pkg / rel).unlink(missing_ok=True)
    for rel in changed:
        (site_pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(patched_root / rel, site_pkg / rel)
    (site_pkg / MANIFEST).write_text(json.dumps({"files": sorted(changed), "patches": patches_digest}))


def verify_overlay(*, patched_root: Path, site_pkg: Path, patches_digest: Optional[str] = None) -> List[str]:
    """Problems with the installed overlay; with patches_digest, also whether the patches changed since apply."""
    if not (site_pkg / MANIFEST).exists():
        return ["no overlay manifest"]
    manifest = _read_manifest(site_pkg)
    problems = []
    if patches_digest is not None and manifest["patches"] != patches_digest:
        problems.append("sglang_patch/0*.patch changed since the overlay was applied")
    return problems + [
        rel for rel in manifest["files"]
        if not (site_pkg / rel).exists() or (site_pkg / rel).read_bytes() != (patched_root / rel).read_bytes()
    ]


def _installed_package() -> Path:
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sglang is not installed in this python")
    return Path(list(spec.submodule_search_locations)[0])


def _changed_files(checkout: Path, base_commit: str) -> List[str]:
    out = subprocess.run(
        ["git", "-C", str(checkout / "sglang"), "diff", "--name-only", base_commit, "HEAD", "--", "python/sglang"],
        check=True, capture_output=True, text=True,
    ).stdout.split()
    return package_relative(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["apply", "verify"])
    parser.add_argument("--checkout", type=Path, default=Path("third_party"))
    args = parser.parse_args(argv)

    site_pkg = _installed_package()
    patched_root = args.checkout / "sglang" / PKG_PREFIX
    if args.command == "apply":
        base = (PATCH_DIR / "BASE_VERSION").read_text().split("commit=")[1].split()[0]
        changed = _changed_files(args.checkout, base)
        apply_overlay(patched_root=patched_root, pristine_root=args.checkout / "sglang-pristine" / PKG_PREFIX,
                      site_pkg=site_pkg, changed=changed, patches_digest=patches_digest())
        print(f"overlaid {len(changed)} file(s) onto {site_pkg}: {', '.join(changed)}")
        return 0
    problems = verify_overlay(patched_root=patched_root, site_pkg=site_pkg, patches_digest=patches_digest())
    for p in problems:
        print(f"overlay mismatch: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
