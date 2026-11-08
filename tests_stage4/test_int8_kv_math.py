import pytest
import torch

from int8kv.quant import compute_scale, quantize, round_scale_up
from tests_stage4.conftest import load_sglang_file

M = load_sglang_file("srt/layers/quantization/int8_kv.py")


def _wide_range_kv(n=37, h=4, d=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, h, d, generator=g)
    return (x * torch.tensor([1.0, 50.0, 300.0, 0.01]).view(1, h, 1)).to(torch.bfloat16)


def _reference(x):
    """int8kv works on [batch, heads, tokens, dim]; SGLang stores [tokens, heads, dim]."""
    xt = x.permute(1, 0, 2).unsqueeze(0)
    scale = round_scale_up(compute_scale(xt, "per_token_head"), torch.bfloat16)
    q = quantize(xt, scale)
    return q.squeeze(0).permute(1, 0, 2), scale.squeeze(0).squeeze(-1).permute(1, 0)


def test_matches_int8kv_reference():
    x = _wide_range_kv()
    q, scale = M.quantize_per_token_head(x)
    ref_q, ref_scale = _reference(x)
    assert q.dtype == torch.int8 and scale.dtype == torch.bfloat16
    assert q.shape == (37, 4, 128) and scale.shape == (37, 4)
    assert torch.equal(scale, ref_scale)
    assert torch.equal(q, ref_q)


def test_zero_rows():
    x = _wide_range_kv()
    x[5, 2] = 0
    q, scale = M.quantize_per_token_head(x)
    assert scale[5, 2].item() == 1.0
    assert torch.count_nonzero(q[5, 2]) == 0
    assert torch.isfinite(M.dequantize_per_token_head(q, scale)).all()


def test_round_trip_error_bound():
    x = _wide_range_kv(seed=1)
    q, scale = M.quantize_per_token_head(x)
    back = M.dequantize_per_token_head(q, scale, dtype=torch.float32)
    bound = scale.float().unsqueeze(-1) / 2 + 1e-6
    assert ((back - x.float()).abs() <= bound).all()


def test_dequantize_default_dtype_is_bf16():
    q, scale = M.quantize_per_token_head(_wide_range_kv())
    assert M.dequantize_per_token_head(q, scale).dtype == torch.bfloat16


def test_bf16_round_up_never_rounds_down():
    g = torch.Generator().manual_seed(2)
    x = torch.rand(10_000, generator=g) * torch.logspace(-6, 3, 10_000)
    up = M.bf16_round_up(x)
    assert up.dtype == torch.bfloat16
    assert (up.float() >= x).all()
    assert torch.equal(up, round_scale_up(x, torch.bfloat16))


def test_qmax_constant():
    assert M.INT8_KV_QMAX == 127
