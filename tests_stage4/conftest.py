"""Shared fixtures for stage 4 tests.

Pure-PyTorch tests load SGLang files by path so they also run on a Mac without
importing the sglang package; GPU tests are marked needs_gpu and skip elsewhere.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SGLANG_PKG = ROOT / "third_party" / "sglang" / "python" / "sglang"
PRISTINE_PKG = ROOT / "third_party" / "sglang-pristine" / "python" / "sglang"
sys.path.insert(0, str(ROOT))


def load_sglang_file(relpath: str, pristine: bool = False):
    """Import one file of the (patched or pristine) SGLang checkout as a standalone module."""
    pkg = PRISTINE_PKG if pristine else SGLANG_PKG
    if not pkg.exists():
        pytest.skip(f"{pkg} not found; run sglang_patch/apply.sh", allow_module_level=True)
    path = pkg / relpath
    if not path.exists():
        raise FileNotFoundError(path)
    name = ("pristine_" if pristine else "patched_") + relpath.replace("/", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gpu_ready() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import sglang  # noqa: F401
    except Exception:
        return False
    return True


needs_gpu = pytest.mark.skipif(not _gpu_ready(), reason="needs CUDA and an importable sglang")
