"""Run the stage 4 server scripts (bench.sh, eval.sh, fp8_probe.sh, run_all.sh) end to end against a fake venv.

The fake `python` stands in for the GPU work: `-m sglang.launch_server` prints the log lines the real server
prints and stays up, `-m sglang.bench_serving` appends a result line, the experiment scripts write small outputs.
Everything else (parse_logs.py, eval_accuracy.py --summarize) runs for real. This checks the shell logic: the
result layout, resuming, and that one failure does not stop the rest.
"""
import json
import os
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen2.5-7B-Instruct"

FAKE_PYTHON = r'''import json, os, sys, time
from pathlib import Path

REAL, STATE, ROOT = {real!r}, Path({state!r}), {root!r}
argv = sys.argv[1:]
joined = " ".join(argv)


def opt(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def note(kind, text):
    with open(STATE / kind, "a") as f:
        f.write(text + "\n")


if "overlay.py" in joined:
    sys.exit(0)
if argv[:3] == ["-m", "pip", "freeze"]:
    print("fake==1.0")
    sys.exit(0)
if argv[:2] == ["-m", "pytest"]:
    note("pytest", joined)
    print("3 passed in 0.01s")
    sys.exit(0)
if argv[:2] == ["-m", "sglang.launch_server"]:
    kv = opt("--kv-cache-dtype")
    note("launches", f"{{kv}}:{{os.environ.get('SGLANG_INT8_KV_IMPL')}}:{{opt('--attention-backend')}}:{{opt('--mem-fraction-static')}}")
    if (STATE / f"fail_server_{{kv}}").exists():
        print("RuntimeError: CUDA out of memory", flush=True)
        sys.exit(1)
    tokens = 77056 if kv == "auto" else 151872
    print(f"KV Cache is allocated. #tokens: {{tokens}}, K size: 2.05 GB, V size: 2.05 GB")
    print(f"max_total_num_tokens={{tokens}}, chunked_prefill_size=2048, max_prefill_tokens=16384, "
          f"max_running_requests=2048, context_len=32768, available_gpu_mem=2.90 GB")
    print("The server is fired up and ready to roll!", flush=True)
    n = 0
    while True:
        n += 1
        print(f"Decode batch, #running-req: {{n % 7 + 3}}, #token: 1, token usage: 0.50, cuda graph: True, "
              f"gen throughput (token/s): 400.00, #queue-req: 0", flush=True)
        time.sleep(0.05)
if argv[:2] == ["-m", "sglang.bench_serving"]:
    out = opt("--output-file")
    note("benches", joined)
    assert opt("--dataset-name") == "random-ids" and "--tokenize-prompt" in argv and "--flush-cache" in argv
    time.sleep(0.25)
    with open(out, "a") as f:
        f.write(json.dumps({{"completed": int(opt("--num-prompts")), "duration": 1.0, "output_throughput": 1000.0,
                             "median_ttft_ms": 50.0, "median_tpot_ms": 5.0, "request_throughput": 2.0,
                             "total_input_tokens": 1, "total_output_tokens": 1,
                             "max_concurrent_requests": int(opt("--num-prompts"))}}) + "\n")
    sys.exit(0)
script = Path(argv[0]).name if argv else ""
if script == "bench_decode_kernel.py":
    contexts = [int(c) for c in opt("--contexts").split(",")]
    Path(opt("--out")).write_text(json.dumps({{"dtype": opt("--dtype"), "batch": 32, "results": [
        {{"context": c, "median_ms": c / 1000}} for c in contexts]}}))
    print(f"KERNEL {{opt('--dtype')}} ok")
    sys.exit(0)
if script == "fp8_probe.py":
    Path(opt("--out")).write_text(json.dumps({{"layers": []}}))
    Path(opt("--calib-out")).write_text(json.dumps({{"model_type": "qwen2", "kv_cache": {{
        "dtype": "float8_e4m3fn", "scaling_factor": {{"0": {{"0": 1.5}}}}}}}}))
    print("PROBE fake ok")
    sys.exit(0)
if script == "kv_format_sim.py":
    note("sim", joined)
    assert Path(opt("--calib")).exists()
    Path(opt("--out")).write_text(json.dumps({{"formats": {{"int8_per_token_head": {{}}}}}}))
    print("SIM fake ok")
    sys.exit(0)
if script == "eval_accuracy.py" and "--summarize" not in argv:
    sys.path.insert(0, str(Path(ROOT) / "experiments" / "stage4"))
    kv, impl, backend, task = opt("--kv-dtype"), opt("--impl"), opt("--attention-backend"), opt("--task")
    note("evals", f"{{task}}:{{kv}}:{{impl}}:{{backend}}:{{opt('--n')}}")
    if (STATE / f"fail_eval_{{kv}}").exists():
        print("Traceback (most recent call last):\nRuntimeError: boom")
        sys.exit(1)
    name = {{"auto": "bf16", "int8_per_token_head": f"int8_{{impl}}"}}.get(kv, kv)
    name = name if backend == "triton" else f"{{name}}_{{backend}}"
    preds = [{{"id": i, "pred": "1", "gold": "1", "correct": True}} for i in range(2)]
    out = Path(opt("--out"))
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{{task}}_{{name}}.json").write_text(json.dumps({{
        "task": task, "config": name, "accuracy": 1.0, "correct": 2, "total": 2, "seconds": 1.0,
        "chunked_prefill_size": 2048, "predictions": preds}}))
    print(f"EVAL {{task}} {{name}}: accuracy=1.0000 (2/2) in 1s")
    sys.exit(0)
os.execv(REAL, [REAL] + argv)
'''


def _executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def server(tmp_path):
    """A fake GPU server: venv with the fake python, stub system tools, results and state directories."""
    venv, stubs, state = tmp_path / "venv", tmp_path / "stubs", tmp_path / "state"
    for d in (venv / "bin", stubs, state, tmp_path / "models" / MODEL):
        d.mkdir(parents=True)
    (venv / "fake_python.py").write_text(FAKE_PYTHON.format(real=sys.executable, state=str(state), root=str(ROOT)))
    # A shell wrapper, not a shebang: the real interpreter's path may contain spaces.
    _executable(venv / "bin" / "python", f'#!/bin/sh\nexec "{sys.executable}" "{venv / "fake_python.py"}" "$@"\n')
    for tool in ("ninja", "nvcc"):
        _executable(venv / "bin" / tool, "#!/bin/sh\nexit 0\n")
    _executable(stubs / "flock", "#!/bin/sh\nexit 0\n")
    _executable(stubs / "setsid", '#!/bin/sh\nexec "$@"\n')
    _executable(stubs / "nvidia-smi", "#!/bin/sh\necho 0\n")
    env = {"HOME": os.environ.get("HOME", str(tmp_path)), "PATH": f"{stubs}:/usr/bin:/bin:/usr/sbin:/sbin",
           "STAGE4_VENV": str(venv), "CUDA_HOME": str(tmp_path / "no-cuda"), "MODELS": str(tmp_path / "models"),
           "STAGE4_RESULTS": str(tmp_path / "results" / "stage4")}

    class Server:
        results = tmp_path / "results" / "stage4"
        model_dir = results / MODEL

        def run(self, script, *args, **extra_env):
            return subprocess.run(["bash", str(ROOT / "scripts" / "stage4" / script), *args], env={**env, **extra_env},
                                  capture_output=True, text=True, timeout=300)

        def lines(self, kind):
            path = state / kind
            return path.read_text().splitlines() if path.exists() else []

        def fail(self, what):
            (state / f"fail_{what}").write_text("")

    return Server()


def test_bench_writes_the_layout_parse_logs_expects(server):
    r = server.run("bench.sh", RUNS="2")
    assert r.returncode == 0, r.stdout + r.stderr
    bench = server.model_dir / "bench"
    assert server.lines("launches") == [
        "auto:plain:triton:0.85", "fp8_e4m3:plain:triton:0.85", "int8_per_token_head:fused:triton:0.85",
        "int8_per_token_head:plain:triton:0.85", "auto:plain:flashinfer:0.85"]
    summary = json.loads((bench / "bench_summary.json").read_text())
    assert sorted(summary["configs"]) == ["bf16", "bf16_flashinfer", "fp8_e4m3", "int8_fused", "int8_plain"]
    full = summary["configs"]["int8_fused"]
    assert sorted(full["serving"]) == ["1024x256", "4096x512", "8192x512"]
    assert all(shape["runs"] == 2 for shape in full["serving"].values())
    assert full["capacity_vs_bf16"] == pytest.approx(151872 / 77056)
    assert full["long"]["8192x512"]["completed"] == 128 and full["long"]["8192x512"]["peak_running_reqs"] >= 3
    light = summary["configs"]["int8_plain"]
    assert sorted(light["serving"]) == ["1024x256"] and light["long"] == {}
    assert sorted(summary["decode_kernel"]) == ["bf16", "fp8", "int8", "int8_plain"]
    assert "BENCH int8_fused" in r.stdout
    # every run used its own seed and fixed lengths
    first = server.lines("benches")[0]
    assert "--random-range-ratio 1.0" in first and "--request-rate inf" in first and "--seed 1" in first


def test_bench_resumes_without_restarting_finished_configs(server):
    assert server.run("bench.sh", BENCH_SMOKE="1").returncode == 0
    launches = len(server.lines("launches"))
    again = server.run("bench.sh", BENCH_SMOKE="1")
    assert again.returncode == 0 and len(server.lines("launches")) == launches
    assert again.stdout.count("skip ") >= 9  # 4 kernel dtypes + 5 configurations
    missing = server.model_dir / "bench_smoke" / "int8_fused" / "long_4096x64_run1.jsonl"
    missing.unlink()
    benches = len(server.lines("benches"))
    third = server.run("bench.sh", BENCH_SMOKE="1")
    assert third.returncode == 0, third.stdout + third.stderr
    assert server.lines("launches")[launches:] == ["int8_per_token_head:fused:triton:0.85"]
    assert len(server.lines("benches")) == benches + 1 and missing.exists()


def test_bench_reports_a_server_that_never_starts_and_finishes_the_rest(server):
    server.fail("server_fp8_e4m3")
    r = server.run("bench.sh", BENCH_SMOKE="1")
    assert r.returncode == 1
    assert "FAILED fp8_e4m3 (server)" in r.stdout and "CUDA out of memory" in r.stdout
    summary = json.loads((server.model_dir / "bench_smoke" / "bench_summary.json").read_text())
    assert "error" in summary["configs"]["fp8_e4m3"]
    assert summary["configs"]["int8_fused"]["serving"]["1024x64"]["runs"] == 1


def test_eval_runs_every_config_on_every_task_and_summarizes(server):
    r = server.run("eval.sh")
    assert r.returncode == 0, r.stdout + r.stderr
    evals = server.lines("evals")
    assert len(evals) == 15 and evals[0] == "gsm8k:auto:plain:triton:None"
    assert "needle600:int8_per_token_head:fused:triton:None" in evals
    summary = json.loads((server.model_dir / "accuracy" / "accuracy_summary.json").read_text())
    assert sorted(summary["gsm8k"]) == ["fp8_e4m3", "fp8_e5m2", "int8_fused", "int8_plain"]
    assert "ACC gsm8k" in r.stdout and "RUN needle150" in r.stdout


def test_run_all_smoke_runs_every_step_and_packs_the_results(server):
    r = server.run("run_all.sh", RUN_ALL_SMOKE="1")
    assert r.returncode == 0, r.stdout + r.stderr
    for step in ("l0", "fp8", "eval", "bench", "pack"):
        assert f"{step}: OK" in r.stdout
    assert "--gsm8k 3" in server.lines("sim")[0] and "fp8_calib_smoke.json" in server.lines("sim")[0]
    assert all(line.endswith(":20") for line in server.lines("evals"))
    tarballs = list(server.results.parent.glob("stage4_results_smoke_*.tar.gz"))
    assert len(tarballs) == 1
    with tarfile.open(tarballs[0]) as tar:
        names = tar.getnames()
    for expected in (f"stage4/{MODEL}/bench_smoke/bench_summary.json",
                     f"stage4/{MODEL}/accuracy_smoke20/accuracy_summary.json",
                     f"stage4/{MODEL}/fp8_probe_smoke.json", f"stage4/{MODEL}/format_sim_smoke.json",
                     "stage4/COMMIT"):
        assert expected in names


def test_run_all_full_mode_uses_full_sizes(server):
    r = server.run("run_all.sh", "fp8,eval")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--gsm8k 100" in server.lines("sim")[0]
    assert all(line.endswith(":None") for line in server.lines("evals"))
    assert (server.model_dir / "fp8_calib.json").exists() and (server.model_dir / "format_sim.json").exists()
    assert "bench" not in r.stdout.split("RUN_ALL SUMMARY")[1]


def test_run_all_reports_a_failed_step_and_still_runs_the_rest(server):
    server.fail("eval_fp8_e5m2")
    r = server.run("run_all.sh", RUN_ALL_SMOKE="1")
    assert r.returncode == 1
    tail = r.stdout.split("RUN_ALL SUMMARY")[1]
    assert "eval: FAILED" in tail and "bench: OK" in tail and "pack: OK" in tail
    assert "!! FAILED gsm8k fp8_e5m2:plain" in r.stdout
