"""Harness logic of experiments/stage4/bench_decode_kernel.py on CPU with injected fake kernels.

The real kernels need CUDA and Triton; here we check what the harness builds and how it calls them, and that the
keyword names it relies on exist in the patched SGLang source."""
import ast
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("bench_decode_kernel",
                                               ROOT / "experiments" / "stage4" / "bench_decode_kernel.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

SGLANG = ROOT / "third_party" / "sglang" / "python" / "sglang"
SHAPE = dict(batch=3, context=10, q_heads=8, kv_heads=2, head_dim=16, kv_splits=4)


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def _fake_store(cache, loc, q_buf, scale_buf):
    from int8kv.quant import compute_scale, quantize, round_scale_up

    x = cache.float().permute(1, 0, 2).unsqueeze(0)
    s = round_scale_up(compute_scale(x, "per_token_head"), torch.bfloat16)
    q_buf[loc] = quantize(x, s).squeeze(0).permute(1, 0, 2)
    scale_buf[loc] = s.squeeze(0).squeeze(-1).permute(1, 0)


def _fake_dequant(q, scale, dtype=torch.bfloat16):
    return (q.float() * scale.float().unsqueeze(-1)).to(dtype)


def test_parse_contexts():
    assert B.parse_contexts("1024,4096, 16384") == [1024, 4096, 16384]
    with pytest.raises(ValueError):
        B.parse_contexts("1024,-1")


def test_case_shapes_match_the_decode_kernel_contract():
    c = B.build_case(device="cpu", **SHAPE)
    assert c["q"].shape == (3, 8, 16) and c["q"].dtype == torch.bfloat16
    assert c["k"].shape == (30, 2, 16) and c["v"].shape == (30, 2, 16)
    assert c["kv_indptr"].tolist() == [0, 10, 20, 30] and c["kv_indptr"].dtype == torch.int32
    assert c["kv_indices"].tolist() == list(range(30)) and c["kv_indices"].dtype == torch.int64
    assert c["attn_logits"].shape == (3, 8, 4, 16) and c["attn_logits"].dtype == torch.float32
    assert c["attn_lse"].shape == (3, 8, 4) and c["num_kv_splits"].tolist() == [4, 4, 4]
    assert c["o"].shape == (3, 8, 16)


@pytest.mark.parametrize("dtype", ["bf16", "fp8", "int8", "int8_plain"])
def test_runner_calls_the_kernel_with_the_right_buffers(dtype):
    c = B.build_case(device="cpu", **SHAPE)
    decode = Recorder()
    run = B.make_runner(dtype, c, decode_fn=decode, store_fn=_fake_store, dequant_fn=_fake_dequant)
    run()
    (args, kwargs), = decode.calls
    q, k_buf, v_buf, o, kv_indptr, kv_indices, logits, lse, splits, max_splits, sm_scale, k_scale, v_scale = args
    assert q is c["q"] and o is c["o"] and max_splits == 4 and sm_scale == pytest.approx(16 ** -0.5)
    if dtype == "bf16":
        assert k_buf.dtype == torch.bfloat16 and (k_scale, v_scale) == (1.0, 1.0) and kwargs == {}
    elif dtype == "fp8":
        assert k_buf.dtype == torch.float8_e4m3fn and v_buf.dtype == torch.float8_e4m3fn
        assert isinstance(k_scale, float) and k_scale > 0 and kwargs == {}
        assert torch.isfinite(k_buf.float()).all()
    elif dtype == "int8":
        assert k_buf.dtype == torch.int8 and (k_scale, v_scale) == (1.0, 1.0)
        assert set(kwargs) == {"k_scale_buffer", "v_scale_buffer"}
        assert kwargs["k_scale_buffer"].shape == (30, 2) and kwargs["k_scale_buffer"].dtype == torch.bfloat16
    else:  # int8_plain dequantizes the whole pool on every call, then runs the bf16 kernel
        assert k_buf.dtype == torch.bfloat16 and kwargs == {}
        run()
        assert decode.calls[1][0][1] is not k_buf


def test_unknown_dtype_is_rejected():
    c = B.build_case(device="cpu", **SHAPE)
    with pytest.raises(ValueError, match="dtype"):
        B.make_runner("fp4", c, decode_fn=Recorder(), store_fn=_fake_store, dequant_fn=_fake_dequant)


def test_time_kernel_discards_warmup_and_reports_the_median():
    samples = iter([50.0, 60.0, 1.0, 3.0, 2.0, 100.0])
    calls = []
    stats = B.time_kernel(lambda: calls.append(1), warmup=2, iters=4, timer=lambda fn: (fn(), next(samples))[1])
    assert len(calls) == 6
    assert stats == {"median_ms": 2.5, "min_ms": 1.0, "p90_ms": 100.0, "iters": 4}


def _signature(relpath, func):
    tree = ast.parse((SGLANG / relpath).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func)
    return [a.arg for a in node.args.args]


@pytest.mark.skipif(not SGLANG.exists(), reason="patched SGLang checkout not present")
def test_harness_argument_order_matches_the_patched_sglang_signatures():
    params = _signature("srt/layers/attention/triton_ops/decode_attention.py", "decode_attention_fwd")
    assert params[:13] == ["q", "k_buffer", "v_buffer", "o", "kv_indptr", "kv_indices", "attn_logits", "attn_lse",
                           "num_kv_splits", "max_kv_splits", "sm_scale", "k_scale", "v_scale"]
    assert {"k_scale_buffer", "v_scale_buffer"} <= set(params)
    assert _signature("srt/layers/attention/triton_ops/int8_kv_store.py", "int8_quantize_and_store") == [
        "cache", "loc", "q_buf", "scale_buf"]
    assert _signature("srt/layers/quantization/int8_kv.py", "dequantize_per_token_head")[:2] == ["q", "scale"]
