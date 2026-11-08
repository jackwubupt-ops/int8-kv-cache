"""The SGLang patch flow: export commits from third_party/sglang, re-apply them on a fresh clone."""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPLY = ROOT / "sglang_patch" / "apply.sh"
EXPORT = ROOT / "sglang_patch" / "export.sh"
GIT_ENV = {
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
}


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
                          env={**os.environ, **GIT_ENV}).stdout.strip()


def _fake_upstream(tmp_path):
    repo = tmp_path / "upstream"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("base\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    _git(repo, "tag", "v9")
    return repo, _git(repo, "rev-parse", "HEAD")


def _run(script, tmp_path, upstream, patch_dir, *args):
    env = {**os.environ, **GIT_ENV, "SGLANG_REPO_URL": f"file://{upstream}", "SGLANG_PATCH_DIR": str(patch_dir)}
    return subprocess.run(["bash", str(script), *args], env=env, capture_output=True, text=True, cwd=tmp_path)


def test_apply_then_export_round_trip(tmp_path):
    upstream, sha = _fake_upstream(tmp_path)
    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    (patch_dir / "BASE_VERSION").write_text(f"version=v9\ncommit={sha}\n")

    first = _run(APPLY, tmp_path, upstream, patch_dir, str(tmp_path / "tp"))
    assert first.returncode == 0, first.stderr
    work = tmp_path / "tp" / "sglang"
    (work / "a.txt").write_text("patched\n")
    (work / "b.txt").write_text("new\n")
    _git(work, "add", ".")
    _git(work, "commit", "-qm", "int8 change")

    exported = _run(EXPORT, tmp_path, upstream, patch_dir, str(tmp_path / "tp"))
    assert exported.returncode == 0, exported.stderr
    assert len(list(patch_dir.glob("0*.patch"))) == 1

    second = _run(APPLY, tmp_path, upstream, patch_dir, str(tmp_path / "tp2"))
    assert second.returncode == 0, second.stderr
    assert (tmp_path / "tp2" / "sglang" / "a.txt").read_text() == "patched\n"
    assert (tmp_path / "tp2" / "sglang" / "b.txt").read_text() == "new\n"
    assert (tmp_path / "tp2" / "sglang-pristine" / "a.txt").read_text() == "base\n"


def test_apply_rejects_wrong_commit(tmp_path):
    upstream, _ = _fake_upstream(tmp_path)
    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    (patch_dir / "BASE_VERSION").write_text("version=v9\ncommit=0000000000000000000000000000000000000000\n")
    result = _run(APPLY, tmp_path, upstream, patch_dir, str(tmp_path / "tp"))
    assert result.returncode != 0
    assert "commit" in result.stderr


COMMON = ROOT / "scripts" / "stage4" / "common.sh"


def _fake_venv(tmp_path, sglang_file):
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    py = bin_dir / "python"
    py.write_text(f"#!/bin/sh\necho '{sglang_file}'\n")
    py.chmod(0o755)
    return tmp_path / "venv"


def _require_env(venv):
    env = {**os.environ, "STAGE4_VENV": str(venv)}
    return subprocess.run(["bash", "-c", f"source '{COMMON}' && require_env"], env=env, capture_output=True, text=True)


def test_require_env_accepts_patched_checkout(tmp_path):
    venv = _fake_venv(tmp_path, f"{ROOT}/third_party/sglang/python/sglang/__init__.py")
    assert _require_env(venv).returncode == 0


def test_require_env_rejects_other_sglang(tmp_path):
    venv = _fake_venv(tmp_path, "/usr/lib/python3/site-packages/sglang/__init__.py")
    result = _require_env(venv)
    assert result.returncode == 1 and "third_party/sglang" in result.stderr


def test_require_env_rejects_missing_venv(tmp_path):
    result = _require_env(tmp_path / "nope")
    assert result.returncode == 1 and "setup.sh" in result.stderr


def test_apply_switches_existing_checkout_to_new_base(tmp_path):
    upstream, sha9 = _fake_upstream(tmp_path)
    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    (patch_dir / "BASE_VERSION").write_text(f"version=v9\ncommit={sha9}\n")
    assert _run(APPLY, tmp_path, upstream, patch_dir, str(tmp_path / "tp")).returncode == 0

    (upstream / "a.txt").write_text("v10\n")
    _git(upstream, "commit", "-qam", "v10")
    _git(upstream, "tag", "v10")
    sha10 = _git(upstream, "rev-parse", "HEAD")
    (patch_dir / "BASE_VERSION").write_text(f"version=v10\ncommit={sha10}\n")

    result = _run(APPLY, tmp_path, upstream, patch_dir, str(tmp_path / "tp"))
    assert result.returncode == 0, result.stderr
    for name in ("sglang", "sglang-pristine"):
        assert _git(tmp_path / "tp" / name, "rev-parse", "HEAD") == sha10
        assert (tmp_path / "tp" / name / "a.txt").read_text() == "v10\n"
