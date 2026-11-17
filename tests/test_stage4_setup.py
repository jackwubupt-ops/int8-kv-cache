"""scripts/stage4/setup.sh against a fake venv python, fake nvidia-smi and fake curl.

The fake pip fails on the `kernels` pin, so setup.sh stops right after the torch and sglang-kernel
steps: these tests check which index torch comes from and that a wheel copied in by hand is used
without touching the network.
"""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "scripts" / "stage4" / "setup.sh"
KERNEL_WHL = "sglang_kernel-0.4.2.post2+cu129-cp310-abi3-manylinux2014_x86_64.whl"
CN_ARGS = "-i https://mirrors.aliyun.com/pypi/simple --extra-index-url https://mirrors.nju.edu.cn/pytorch/whl/cu128"

FAKE_PY = """#!/bin/bash
echo "python $*" >> "$FAKE_LOG"
[[ $1 == -c ]] && exit 1                       # nothing is installed yet
[[ "$*" == *"kernels>=0.12"* ]] && exit 7      # stop setup.sh here
exit 0
"""


def _exe(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


def _run_setup(tmp_path, with_wheel=True, **env_extra):
    venv, wheels, log = tmp_path / "venv", tmp_path / "wheels", tmp_path / "calls.log"
    _exe(venv / "bin" / "python", FAKE_PY)
    _exe(tmp_path / "fakebin" / "nvidia-smi", "#!/bin/sh\necho 'Fake GPU, 570.124.04'\n")
    _exe(tmp_path / "fakebin" / "curl", '#!/bin/sh\necho "curl $*" >> "$FAKE_LOG"\nexit 22\n')
    if with_wheel:
        wheels.mkdir()
        (wheels / KERNEL_WHL).write_bytes(b"wheel")
    env = {k: v for k, v in os.environ.items() if k not in ("TORCH_INDEX_ARGS", "PIP_INDEX_URL")}
    env.update(STAGE4_VENV=str(venv), WHEELS=str(wheels), FAKE_LOG=str(log), CUDA_HOME=str(tmp_path / "cuda"),
               PATH=f"{tmp_path / 'fakebin'}:/usr/bin:/bin", **env_extra)
    result = subprocess.run(["bash", str(SETUP)], env=env, capture_output=True, text=True, timeout=60)
    calls = log.read_text().splitlines() if log.exists() else []
    return result, calls


def _torch_installs(calls):
    return [c for c in calls if "pip install" in c and "torch==2.11.0+cu128" in c]


def test_setup_installs_torch_from_pytorch_org_by_default(tmp_path):
    result, calls = _run_setup(tmp_path)
    assert result.returncode == 7, result.stderr
    installs = _torch_installs(calls)
    assert installs and all("--index-url https://download.pytorch.org/whl/cu128" in c for c in installs)


def test_setup_torch_index_is_configurable(tmp_path):
    result, calls = _run_setup(tmp_path, TORCH_INDEX_ARGS=CN_ARGS)
    assert result.returncode == 7, result.stderr
    installs = _torch_installs(calls)
    assert len(installs) == 2  # before sglang, and again in the pinned-dependency check
    for call in installs:
        assert CN_ARGS in call and "download.pytorch.org" not in call
        assert "torchvision==0.26.0+cu128" in call and "torchaudio==2.11.0+cu128" in call


def test_setup_uses_a_wheel_copied_in_by_hand_without_downloading(tmp_path):
    result, calls = _run_setup(tmp_path, TORCH_INDEX_ARGS=CN_ARGS)
    assert result.returncode == 7, result.stderr
    assert not [c for c in calls if c.startswith("curl")]
    assert any(c.endswith(f"pip install --force-reinstall --no-deps {tmp_path / 'wheels' / KERNEL_WHL}")
               for c in calls)


def test_setup_names_the_wheel_path_when_the_download_fails(tmp_path):
    result, calls = _run_setup(tmp_path, with_wheel=False)
    assert result.returncode == 1
    assert [c for c in calls if c.startswith("curl")]
    assert KERNEL_WHL in result.stderr and str(tmp_path / "wheels") in result.stderr
