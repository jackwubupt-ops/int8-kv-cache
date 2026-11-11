"""Report tables generated from stage 4 result files (experiments/stage4/make_report.py), on synthetic results."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_report", ROOT / "experiments" / "stage4" / "make_report.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def _acc(d, task, config, correct, chunk=2048):
    preds = [{"id": i, "pred": "x", "gold": "y", "correct": c} for i, c in enumerate(correct)]
    (d / f"{task}_{config}.json").write_text(json.dumps(
        {"task": task, "config": config, "accuracy": sum(correct) / len(correct), "correct": sum(correct),
         "total": len(correct), "seconds": 12.0, "chunked_prefill_size": chunk, "predictions": preds}))


def _server_log(tokens):
    return (f"KV Cache is allocated. #tokens: {tokens}, K size: 2.00 GB, V size: 2.00 GB\n"
            f"max_total_num_tokens={tokens}, chunked_prefill_size=2048, max_prefill_tokens=16384, "
            f"max_running_requests=2048, context_len=32768, available_gpu_mem=2.90 GB\n"
            "Decode batch, #running-req: 9, #token: 1, token usage: 0.9, cuda graph: True, "
            "gen throughput (token/s): 400.00, #queue-req: 0\n")


def _bench(d, config, tokens, tput):
    c = d / config
    c.mkdir(parents=True)
    (c / "server.log").write_text(_server_log(tokens))
    run = {"completed": 256, "duration": 1.0, "output_throughput": tput, "median_ttft_ms": 50.0,
           "median_tpot_ms": 5.0, "request_throughput": 2.0, "total_input_tokens": 1, "total_output_tokens": 1,
           "max_concurrent_requests": 256}
    (c / "serving_1024x256_run1.jsonl").write_text(json.dumps(run) + "\n")
    (c / "long_8192x512_run1.jsonl").write_text(json.dumps(dict(run, completed=128)) + "\n")
    (c / "long_8192x512_server.log").write_text(_server_log(tokens))


@pytest.fixture
def model_dir(tmp_path):
    d = tmp_path / "Qwen2.5-7B-Instruct"
    acc = d / "accuracy"
    acc.mkdir(parents=True)
    _acc(acc, "gsm8k", "bf16", [True, True, True, False])
    _acc(acc, "gsm8k", "int8_fused", [True, True, False, False])
    _acc(acc, "gsm8k", "fp8_e4m3", [False, False, False, False])
    _acc(acc, "needle150", "bf16", [True, True])
    _acc(acc, "needle150", "int8_fused", [True, True])
    bench = d / "bench"
    _bench(bench, "bf16", 77056, 1500.0)
    _bench(bench, "int8_fused", 151872, 1450.0)
    (bench / "decode_kernel_bf16.json").write_text(json.dumps(
        {"dtype": "bf16", "batch": 32, "results": [{"context": 1024, "median_ms": 0.40}]}))
    (bench / "decode_kernel_int8.json").write_text(json.dumps(
        {"dtype": "int8", "batch": 32, "results": [{"context": 1024, "median_ms": 0.50}]}))
    fmt = {"nan_frac": 0.0, "rel_l2": 0.02, "centered_rel_l2": 0.05}
    layer = {"absmax": 700.0, "channel_absmax_max_over_median": 80.0, "e4m3_overflow_frac": 0.001,
             "centered_norm_ratio": 0.6, "fp8_e4m3": dict(fmt, nan_frac=0.001), "fp8_e5m2": fmt,
             "int8_per_token_head": dict(fmt, centered_rel_l2=0.02), "fp8_e4m3_calib": fmt}
    (d / "fp8_probe.json").write_text(json.dumps({"probe_tokens": 3353, "margin": 2.0, "layers": [
        {"layer": 0, "calib_scale": 3.1, "k": layer, "v": dict(layer, absmax=20.0, e4m3_overflow_frac=0.0)}]}))
    entry = {"prompts": 22, "top1_per_prompt_mean": 0.98, "top1_short": 0.98, "top1_needle": 0.99,
             "kl_mean_finite": 0.003, "nan_logit_prompts": 0, "exact_generation_rate": 0.6, "needle_correct": 9,
             "needle_total": 10, "gsm8k": {"accuracy": 0.9, "baseline_accuracy": 0.92, "only_baseline": 3,
                                            "only_quantized": 1, "mcnemar_p": 0.625, "total": 100}}
    (d / "format_sim.json").write_text(json.dumps({
        "baseline": {"needle_correct": 10, "needle_total": 10},
        "gsm8k_baseline": {"accuracy": 0.92, "total": 100},
        "formats": {"int8_per_token_head": entry,
                    "fp8_e4m3": dict(entry, top1_per_prompt_mean=0.1, nan_logit_prompts=22, needle_correct=0)}}))
    return d


def test_accuracy_table_pairs_each_config_with_bf16(model_dir):
    text = R.build_report(model_dir)
    assert "| int8_fused | 0.5000 (2/4) | -0.2500 | 1 | 0 | 1 |" in text
    assert "| fp8_e4m3 | 0.0000 (0/4) | -0.7500 | 3 | 0 | 0.25 |" in text
    assert "| bf16 | 0.7500 (3/4) |" in text


def test_capacity_table_shows_ratio_against_theory(model_dir):
    text = R.build_report(model_dir)
    assert "| int8_fused | 151872 | 1.971 | 4.00 |" in text
    assert "1.969" in text  # theoretical ratio for head_dim 128


def test_serving_long_and_kernel_tables(model_dir):
    text = R.build_report(model_dir)
    assert "| int8_fused | 1024x256 | 1450.0 | 50 | 5.0 | 1 |" in text
    assert "| int8_fused | 8192x512 | 9 | 1450.0 | 128 |" in text
    assert "| int8 | 0.500 |" in text and "| bf16 | 0.400 |" in text


def test_fp8_section_reports_overflow_and_simulation(model_dir):
    text = R.build_report(model_dir)
    assert "700.0" in text and "1/1" in text  # K absmax and layers with e4m3 overflow
    assert "| fp8_e4m3 | 0.1000 |" in text and "22/22" in text
    assert "| int8_per_token_head | 0.9800 |" in text and "0.9000" in text


def test_checks_flag_a_capacity_ratio_far_from_theory(model_dir):
    ok = R.checks(model_dir)
    assert not [c for c in ok if c.startswith("⚠")]
    (model_dir / "bench" / "int8_fused" / "server.log").write_text(_server_log(100000))
    flagged = R.checks(model_dir)
    assert any(c.startswith("⚠") and "int8_fused" in c and "1.298" in c for c in flagged)


def test_checks_list_missing_results(model_dir):
    lines = R.checks(model_dir)
    assert any("needle600" in c for c in lines)            # a task with no results at all
    assert any("gsm8k" in c and "int8_plain" in c for c in lines)  # an expected config missing for a task


def test_report_survives_missing_sections(tmp_path):
    d = tmp_path / "Empty"
    d.mkdir()
    text = R.build_report(d)
    assert "没有精度结果" in text and "没有性能结果" in text and "没有 FP8 探针结果" in text


def test_main_writes_report_tables_file(model_dir):
    assert R.main([str(model_dir)]) == 0
    assert (model_dir / "report_tables.md").read_text() == R.build_report(model_dir)
