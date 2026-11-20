"""Benchmark log and result parsing (experiments/stage4/parse_logs.py) on fixed samples.

The server log sample follows the format strings in SGLang v0.5.12 (scheduler.py, memory_pool.py,
scheduler_metrics_mixin.py); the bench_serving sample has the fields bench_serving.py writes per run."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "stage4"
_spec = importlib.util.spec_from_file_location("parse_logs", ROOT / "experiments" / "stage4" / "parse_logs.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

LOG = (FIX / "server.log").read_text()


def test_parse_max_total_tokens():
    assert P.parse_max_total_tokens(LOG) == 151872


def test_parse_max_total_tokens_missing_is_an_error():
    with pytest.raises(ValueError, match="max_total_num_tokens"):
        P.parse_max_total_tokens("server crashed before the pool was built")


def test_parse_peak_running_reqs_covers_prefill_and_decode_lines():
    assert P.parse_peak_running_reqs(LOG) == 17
    assert P.parse_peak_running_reqs("no batches yet") == 0


def test_parse_server_log_collects_capacity_facts():
    info = P.parse_server_log(LOG)
    assert info == {"max_total_num_tokens": 151872, "kv_cache_gb": pytest.approx(4.12),
                    "chunked_prefill_size": 2048, "max_running_requests": 2048,
                    "peak_running_reqs": 17, "peak_gen_throughput": pytest.approx(498.10)}


def test_parse_bench_serving_reads_the_last_run_in_the_file(tmp_path):
    out = P.parse_bench_serving(FIX / "bench_serving.jsonl")
    assert out == {"output_throughput": 1579.2, "median_ttft_ms": 9120.5, "median_tpot_ms": 81.3,
                   "request_throughput": 6.17, "completed": 256, "duration": 41.5,
                   "total_input_tokens": 262144, "total_output_tokens": 65536, "max_concurrent_requests": 256}
    two = tmp_path / "two.jsonl"
    first = json.loads((FIX / "bench_serving.jsonl").read_text())
    second = dict(first, output_throughput=1700.0)
    two.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    assert P.parse_bench_serving(two)["output_throughput"] == 1700.0


def _bench_dir(tmp_path):
    """bench/<config>/ as bench.sh lays it out, for bf16 (two shapes) and int8_fused (one shape + long test)."""
    base = json.loads((FIX / "bench_serving.jsonl").read_text())

    def run(d, name, **kw):
        (d / name).write_text(json.dumps(dict(base, **kw)) + "\n")

    bf16 = tmp_path / "bf16"
    bf16.mkdir()
    (bf16 / "server.log").write_text(LOG.replace("151872", "77056").replace("2.06 GB", "2.05 GB"))
    for i, tput in enumerate([1500.0, 1700.0, 1600.0], 1):
        run(bf16, f"serving_1024x256_run{i}.jsonl", output_throughput=tput, median_ttft_ms=100.0 * i, median_tpot_ms=10.0 * i)
    run(bf16, "serving_4096x512_run1.jsonl", output_throughput=900.0)
    int8 = tmp_path / "int8_fused"
    int8.mkdir()
    (int8 / "server.log").write_text(LOG)
    for i, tput in enumerate([1400.0, 1450.0], 1):
        run(int8, f"serving_1024x256_run{i}.jsonl", output_throughput=tput)
    run(int8, "long_8192x512_run1.jsonl", output_throughput=700.0, completed=256)
    (int8 / "long_8192x512_server.log").write_text(LOG)
    return tmp_path


def test_summarize_bench_takes_medians_over_runs(tmp_path):
    s = P.summarize_bench(_bench_dir(tmp_path))
    shape = s["configs"]["bf16"]["serving"]["1024x256"]
    assert shape["runs"] == 3 and shape["output_throughput"] == 1600.0
    assert shape["median_ttft_ms"] == 200.0 and shape["median_tpot_ms"] == 20.0
    assert s["configs"]["bf16"]["serving"]["4096x512"] == pytest.approx(
        {"runs": 1, "output_throughput": 900.0, "median_ttft_ms": 9120.5, "median_tpot_ms": 81.3, "completed": 256})
    assert s["configs"]["int8_fused"]["serving"]["1024x256"]["output_throughput"] == 1425.0  # median of two = mean


def test_summarize_bench_reports_the_fewest_completed_requests_over_runs(tmp_path):
    d = _bench_dir(tmp_path)
    partial = json.loads((d / "bf16" / "serving_1024x256_run2.jsonl").read_text())
    (d / "bf16" / "serving_1024x256_run2.jsonl").write_text(json.dumps(dict(partial, completed=200)) + "\n")
    assert P.summarize_bench(d)["configs"]["bf16"]["serving"]["1024x256"]["completed"] == 200


def test_summarize_bench_reports_capacity_ratio_against_bf16(tmp_path):
    s = P.summarize_bench(_bench_dir(tmp_path))
    assert s["configs"]["bf16"]["max_total_num_tokens"] == 77056 and s["configs"]["bf16"]["capacity_vs_bf16"] == 1.0
    assert s["configs"]["int8_fused"]["capacity_vs_bf16"] == pytest.approx(151872 / 77056)


def test_summarize_bench_long_test_uses_its_own_server_log_segment(tmp_path):
    s = P.summarize_bench(_bench_dir(tmp_path))
    long = s["configs"]["int8_fused"]["long"]["8192x512"]
    assert long["peak_running_reqs"] == 17 and long["output_throughput"] == 700.0 and long["completed"] == 256
    assert s["configs"]["bf16"]["long"] == {}


def test_summarize_bench_writes_the_summary_file(tmp_path):
    s = P.summarize_bench(_bench_dir(tmp_path))
    assert json.loads((tmp_path / "bench_summary.json").read_text()) == s


def test_summarize_bench_includes_kernel_microbench_results(tmp_path):
    d = _bench_dir(tmp_path)
    (d / "decode_kernel_int8.json").write_text(json.dumps({"dtype": "int8", "batch": 32, "results": [
        {"context": 1024, "median_ms": 0.41}, {"context": 4096, "median_ms": 1.52}]}))
    s = P.summarize_bench(d)
    assert s["decode_kernel"]["int8"] == {"1024": 0.41, "4096": 1.52}


def test_summarize_bench_keeps_going_when_one_server_never_started(tmp_path):
    d = _bench_dir(tmp_path)
    broken = d / "fp8_e4m3"
    broken.mkdir()
    (broken / "server.log").write_text("Traceback (most recent call last):\nRuntimeError: CUDA out of memory\n")
    s = P.summarize_bench(d)
    assert "max_total_num_tokens" in s["configs"]["fp8_e4m3"]["error"]
    assert s["configs"]["int8_fused"]["capacity_vs_bf16"] == pytest.approx(151872 / 77056)
    assert P.main([str(d)]) == 0
