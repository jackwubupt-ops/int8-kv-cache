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
    assert json.loads((site / O.MANIFEST).read_text())["files"] == ["srt/a.py", "srt/new.py"]


def test_reapply_restores_files_no_longer_patched(tmp_path):
    pristine, patched, site = _setup(tmp_path)
    O.apply_overlay(patched_root=patched, pristine_root=pristine, site_pkg=site, changed=["srt/a.py", "srt/new.py"])
    O.apply_overlay(patched_root=patched, pristine_root=pristine, site_pkg=site, changed=["srt/b.py"])
    assert (site / "srt/a.py").read_text() == "a0"
    assert not (site / "srt/new.py").exists()
    assert json.loads((site / O.MANIFEST).read_text())["files"] == ["srt/b.py"]


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


def _patches(root: Path, texts):
    root.mkdir(exist_ok=True)
    for old in root.glob("0*.patch"):
        old.unlink()
    for i, text in enumerate(texts, 1):
        (root / f"000{i}-x.patch").write_text(text)
    return root


def test_verify_reports_patches_changed_since_apply(tmp_path):
    pristine, patched, site = _setup(tmp_path)
    patch_dir = _patches(tmp_path / "patches", ["one", "two"])
    O.apply_overlay(patched_root=patched, pristine_root=pristine, site_pkg=site, changed=["srt/a.py"],
                    patches_digest=O.patches_digest(patch_dir))
    assert O.verify_overlay(patched_root=patched, site_pkg=site, patches_digest=O.patches_digest(patch_dir)) == []
    _patches(patch_dir, ["one", "two", "three"])  # a pulled commit adds a patch; setup.sh was not rerun
    problems = O.verify_overlay(patched_root=patched, site_pkg=site, patches_digest=O.patches_digest(patch_dir))
    assert len(problems) == 1 and "sglang_patch" in problems[0]


def test_verify_rejects_a_manifest_from_before_patch_digests(tmp_path):
    pristine, patched, site = _setup(tmp_path)
    (site / O.MANIFEST).write_text(json.dumps(["srt/a.py"]))
    (site / "srt/a.py").write_text("a1")
    problems = O.verify_overlay(patched_root=patched, site_pkg=site, patches_digest="abc")
    assert len(problems) == 1 and "sglang_patch" in problems[0]


def test_patches_digest_depends_on_patch_contents(tmp_path):
    d = _patches(tmp_path / "p", ["one"])
    first = O.patches_digest(d)
    assert O.patches_digest(_patches(d, ["one"])) == first
    assert O.patches_digest(_patches(d, ["changed"])) != first
