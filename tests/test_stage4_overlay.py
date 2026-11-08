"""Overlay of patched SGLang .py files onto the installed wheel (no source build needed)."""
import importlib.util
import json
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "sglang_patch" / "overlay.py"
_spec = importlib.util.spec_from_file_location("overlay", _PATH)
O = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(O)


def _tree(root: Path, files: dict):
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)


def _setup(tmp_path):
    pristine, patched, site = tmp_path / "pristine", tmp_path / "patched", tmp_path / "site" / "sglang"
    _tree(pristine, {"srt/a.py": "a0", "srt/b.py": "b0"})
    _tree(patched, {"srt/a.py": "a1", "srt/b.py": "b0", "srt/new.py": "n1"})
    _tree(site, {"srt/a.py": "a0", "srt/b.py": "b0", "srt/ext.so": "binary"})
    return pristine, patched, site


def test_apply_copies_changed_files_and_writes_manifest(tmp_path):
    pristine, patched, site = _setup(tmp_path)
    O.apply_overlay(patched_root=patched, pristine_root=pristine, site_pkg=site, changed=["srt/a.py", "srt/new.py"])
    assert (site / "srt/a.py").read_text() == "a1"
    assert (site / "srt/new.py").read_text() == "n1"
    assert (site / "srt/ext.so").read_text() == "binary"
    assert json.loads((site / O.MANIFEST).read_text()) == ["srt/a.py", "srt/new.py"]


def test_reapply_restores_files_no_longer_patched(tmp_path):
    pristine, patched, site = _setup(tmp_path)
    O.apply_overlay(patched_root=patched, pristine_root=pristine, site_pkg=site, changed=["srt/a.py", "srt/new.py"])
    O.apply_overlay(patched_root=patched, pristine_root=pristine, site_pkg=site, changed=["srt/b.py"])
    assert (site / "srt/a.py").read_text() == "a0"
    assert not (site / "srt/new.py").exists()
    assert json.loads((site / O.MANIFEST).read_text()) == ["srt/b.py"]


def test_verify_reports_mismatches(tmp_path):
    pristine, patched, site = _setup(tmp_path)
    assert O.verify_overlay(patched_root=patched, site_pkg=site) == ["no overlay manifest"]
    O.apply_overlay(patched_root=patched, pristine_root=pristine, site_pkg=site, changed=["srt/a.py"])
    assert O.verify_overlay(patched_root=patched, site_pkg=site) == []
    (site / "srt/a.py").write_text("edited")
    assert O.verify_overlay(patched_root=patched, site_pkg=site) == ["srt/a.py"]


def test_changed_files_strip_package_prefix():
    names = ["python/sglang/srt/a.py", "python/pyproject.toml", "docs/x.md", "python/sglang/srt/n.py"]
    assert O.package_relative(names) == ["srt/a.py", "srt/n.py"]
