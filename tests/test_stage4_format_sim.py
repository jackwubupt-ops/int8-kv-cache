"""Pure-PyTorch KV format simulation (experiments/stage4/kv_format_sim.py)."""
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "stage4" / "kv_format_sim.py"
_spec = importlib.util.spec_from_file_location("kv_format_sim", _PATH)
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def _kv(tokens, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, 2, tokens, 8, generator=g) * scale, torch.randn(1, 2, tokens, 8, generator=g) * scale)


def test_roundtrip_int8_matches_the_stage3_quantizer():
    from int8kv.quant import compute_scale, dequantize, quantize, round_scale_up

    k, _ = _kv(5, scale=30.0)
    s = round_scale_up(compute_scale(k, "per_token_head"), torch.bfloat16)
    assert torch.equal(S.roundtrip(k, "int8_per_token_head"), dequantize(quantize(k, s), s))


def test_roundtrip_fp8_formats_and_calibrated_scale():
    x = torch.tensor([[[[1.0, 300.0, 500.0, -900.0]]]])
    assert torch.isnan(S.roundtrip(x, "fp8_e4m3")).sum() == 2
    assert torch.isfinite(S.roundtrip(x, "fp8_e5m2")).all()
    calibrated = S.roundtrip(x, "fp8_e4m3_calib", scale=4.0)
    assert torch.isfinite(calibrated).all()
    assert ((calibrated - x).abs() <= x.abs() / 16 + 1e-6).all()


def test_prefill_chunk_attends_to_its_own_float_kv_but_stores_the_roundtrip():
    cache = S.RoundtripKVCache("fp8_e5m2")
    k, v = _kv(5)
    k_used, v_used = cache.update(k, v, 0)
    assert torch.equal(k_used, k) and torch.equal(v_used, v)
    k1, v1 = _kv(1, seed=1)
    k_all, v_all = cache.update(k1, v1, 0)
    assert k_all.shape[-2] == 6
    assert torch.equal(k_all[..., :5, :], S.roundtrip(k, "fp8_e5m2"))
    assert torch.equal(k_all[..., 5:, :], S.roundtrip(k1, "fp8_e5m2"))  # decode reads its own token from the cache
    assert torch.equal(v_all[..., :5, :], S.roundtrip(v, "fp8_e5m2"))


def test_second_chunk_reads_the_first_from_the_cache():
    cache = S.RoundtripKVCache("int8_per_token_head")
    k, v = _kv(4)
    cache.update(k, v, 0)
    k2, v2 = _kv(3, seed=2)
    k_all, _ = cache.update(k2, v2, 0)
    assert torch.equal(k_all[..., :4, :], S.roundtrip(k, "int8_per_token_head"))
    assert torch.equal(k_all[..., 4:, :], k2)


def test_float_prefill_off_reads_everything_from_the_cache():
    cache = S.RoundtripKVCache("fp8_e5m2", float_prefill=False)
    k, v = _kv(5)
    k_used, _ = cache.update(k, v, 0)
    assert torch.equal(k_used, S.roundtrip(k, "fp8_e5m2"))


def test_calibrated_cache_uses_one_scale_per_layer():
    cache = S.RoundtripKVCache("fp8_e4m3_calib", scales=[1.0, 4.0])
    k, v = _kv(3, scale=300.0)
    for layer in (0, 1):
        cache.update(k, v, layer)
    k1, v1 = _kv(1, seed=3)
    layer0, _ = cache.update(k1, v1, 0)
    layer1, _ = cache.update(k1, v1, 1)
    assert torch.isnan(layer0).any()       # scale 1.0 overflows e4m3
    assert torch.isfinite(layer1).all()    # scale 4.0 does not


def test_calibrated_format_requires_scales():
    with pytest.raises(ValueError, match="scales"):
        S.RoundtripKVCache("fp8_e4m3_calib")


def test_load_calib_scales_reads_the_sglang_quant_param_file(tmp_path):
    path = tmp_path / "calib.json"
    path.write_text(json.dumps({"model_type": "qwen2", "kv_cache": {
        "dtype": "float8_e4m3fn", "scaling_factor": {"0": {"1": 2.5, "0": 1.0, "2": 3.0}}}}))
    assert S.load_calib_scales(path) == [1.0, 2.5, 3.0]


def test_summarize_reports_nan_logits_instead_of_crashing():
    rows = [{"kind": "short", "top1": 1.0, "kl": 0.01, "nan_logits": False, "exact": True, "needle_correct": None},
            {"kind": "needle", "top1": 0.0, "kl": float("nan"), "nan_logits": True, "exact": False,
             "needle_correct": False}]
    s = S.summarize(rows)
    assert s["top1_per_prompt_mean"] == 0.5 and s["nan_logit_prompts"] == 1
    assert s["kl_mean_finite"] == pytest.approx(0.01) and s["needle_correct"] == 0 and s["needle_total"] == 1
    assert s["exact_generation_rate"] == 0.5
    json.dumps(s, allow_nan=False)  # must stay strict JSON


def test_paired_gsm8k_counts():
    base = [True, True, False, True]
    other = [True, False, False, False]
    r = S.paired(base, other)
    assert r == {"accuracy": 0.25, "baseline_accuracy": 0.75, "only_baseline": 2, "only_quantized": 0,
                 "mcnemar_p": pytest.approx(0.5), "total": 4}
    assert not math.isnan(r["mcnemar_p"])


def test_update_accepts_both_transformers_calling_conventions():
    """transformers 4.57 passes cache_kwargs as a 4th positional argument; 5.x passes only three."""
    k, v = _kv(2)
    for extra in ((), ({"cache_position": None},)):
        cache = S.RoundtripKVCache("fp8_e5m2")
        k_used, _ = cache.update(k, v, 0, *extra)
        assert torch.equal(k_used, k)
