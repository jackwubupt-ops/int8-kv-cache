"""Overlay the patched SGLang .py files onto the installed sglang wheel.

The v0.5.12 source tree needs a Rust toolchain to build, so instead of an editable
install we copy only the files our patches touch into the installed package and
record them in a manifest. Re-applying first restores files that are no longer
patched from the pristine checkout.

Usage (run with the stage 4 venv python):
  python sglang_patch/overlay.py apply  [--checkout third_party]
  python sglang_patch/overlay.py verify [--checkout third_party]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

MANIFEST = ".int8kv_overlay.json"
PKG_PREFIX = "python/sglang/"


def package_relative(names: Sequence[str]) -> List[str]:
    return [n[len(PKG_PREFIX):] for n in names if n.startswith(PKG_PREFIX)]


def _read_manifest(site_pkg: Path) -> List[str]:
    path = site_pkg / MANIFEST
    return json.loads(path.read_text()) if path.exists() else []


def apply_overlay(*, patched_root: Path, pristine_root: Path, site_pkg: Path, changed: Sequence[str]) -> None:
    for rel in set(_read_manifest(site_pkg)) - set(changed):
        original = pristine_root / rel
        if original.exists():
            shutil.copyfile(original, site_pkg / rel)
        else:
            (site_pkg / rel).unlink(missing_ok=True)
    for rel in changed:
        (site_pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(patched_root / rel, site_pkg / rel)
    (site_pkg / MANIFEST).write_text(json.dumps(sorted(changed)))


def verify_overlay(*, patched_root: Path, site_pkg: Path) -> List[str]:
    if not (site_pkg / MANIFEST).exists():
        return ["no overlay manifest"]
    return [
        rel for rel in _read_manifest(site_pkg)
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
        base = (Path(__file__).parent / "BASE_VERSION").read_text().split("commit=")[1].split()[0]
        changed = _changed_files(args.checkout, base)
        apply_overlay(patched_root=patched_root, pristine_root=args.checkout / "sglang-pristine" / PKG_PREFIX,
                      site_pkg=site_pkg, changed=changed)
        print(f"overlaid {len(changed)} file(s) onto {site_pkg}: {', '.join(changed)}")
        return 0
    problems = verify_overlay(patched_root=patched_root, site_pkg=site_pkg)
    for p in problems:
        print(f"overlay mismatch: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
