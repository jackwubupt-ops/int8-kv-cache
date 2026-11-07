import math

import pytest
import torch
import torch.nn.functional as F

from int8kv.attention import attention_float, attention_int8, integer_dot
from int8kv.quant import compute_scale, quantize


def make(seed=0, tq=3, tk=9, heads=2, d=32):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(1, heads, tq, d, generator=g),
        torch.randn(1, heads, tk, d, generator=g),
        torch.randn(1, heads, tk, d, generator=g),
    )


def quant(k, v, gran="per_token_head"):
    sk, sv = compute_scale(k, gran), compute_scale(v, gran)
    return quantize(k, sk), sk, quantize(v, sv), sv


def test_float_baseline_matches_pytorch_sdpa():
    q, k, v = make()
    assert torch.allclose(attention_float(q, k, v), F.scaled_dot_product_attention(q, k, v), atol=1e-5)


def test_float_baseline_matches_sdpa_with_bool_mask():
    q, k, v = make()
    mask = torch.rand(3, 9) > 0.3
    mask[:, 0] = True
    assert torch.allclose(attention_float(q, k, v, mask), F.scaled_dot_product_attention(q, k, v, attn_mask=mask), atol=1e-5)


def test_float_baseline_matches_sdpa_with_additive_mask():
    q, k, v = make()
    mask = torch.zeros(3, 9)
    mask[:, 4:] = -1e9
    assert torch.allclose(attention_float(q, k, v, mask), F.scaled_dot_product_attention(q, k, v, attn_mask=mask), atol=1e-5)


def test_masked_key_does_not_affect_output():
    q, k, v = make()
    mask = torch.ones(3, 9, dtype=torch.bool)
    mask[:, -1] = False
    ki, sk, vi, sv = quant(k, v)
    base = attention_int8(q, ki, sk, vi, sv, mask, mode="integer")
    vi2 = vi.clone()
    vi2[..., -1, :] = 99
    assert torch.equal(base, attention_int8(q, ki, sk, vi2, sv, mask, mode="integer"))


@pytest.mark.parametrize("gran", ["per_tensor", "per_head", "per_token", "per_token_head"])
@pytest.mark.parametrize("mode", ["dequant", "integer"])
def test_int8_output_close_to_float(gran, mode):
    q, k, v = make(1)
    ki, sk, vi, sv = quant(k, v, gran)
    base = attention_float(q, k, v)
    out = attention_int8(q, ki, sk, vi, sv, mode=mode)
    assert (out - base).norm() / base.norm() < 0.10


def test_integer_dot_is_exact_at_extreme_values():
    d = 128
    q = torch.full((1, 1, 1, d), 127, dtype=torch.int8)
    k = torch.full((1, 1, 2, d), -127, dtype=torch.int8)
    assert integer_dot(q, k).tolist() == [[[[-2064512.0, -2064512.0]]]]


def test_integer_dot_large_dim_stays_exact():
    d = 2048
    q = torch.full((1, 1, 1, d), 127, dtype=torch.int8)
    q[..., 0] = 126
    k = torch.full((1, 1, 1, d), 127, dtype=torch.int8)
    assert integer_dot(q, k).item() == 127 * 127 * (d - 1) + 126 * 127


def test_dequant_mode_equals_float_on_reconstructed_kv():
    q, k, v = make(2)
    ki, sk, vi, sv = quant(k, v)
    out = attention_int8(q, ki, sk, vi, sv, mode="dequant")
    assert torch.allclose(out, attention_float(q, ki.float() * sk, vi.float() * sv), atol=1e-6)


def test_output_keeps_query_dtype():
    q, k, v = make()
    ki, sk, vi, sv = quant(k, v)
    assert attention_int8(q.to(torch.bfloat16), ki, sk, vi, sv, mode="integer").dtype == torch.bfloat16


def test_batch_items_are_independent():
    q, k, v = make(3)
    q2, k2, v2 = make(4)
    a = quant(k, v)
    b = quant(k2, v2)
    both = attention_int8(
        torch.cat([q, q2]), torch.cat([a[0], b[0]]), torch.cat([a[1], b[1]]),
        torch.cat([a[2], b[2]]), torch.cat([a[3], b[3]]), mode="integer",
    )
    assert torch.allclose(both[0:1], attention_int8(q, *a, mode="integer"), atol=1e-5)


def test_unknown_mode_rejected():
    q, k, v = make()
    with pytest.raises(ValueError, match="mode"):
        attention_int8(q, *quant(k, v), mode="fast")


@pytest.mark.parametrize("mode", ["dequant", "integer"])
def test_fully_masked_query_row_gives_zeros_not_nan(mode):
    q, k, v = make()
    mask = torch.ones(3, 9, dtype=torch.bool)
    mask[0] = False
    out = attention_int8(q, *quant(k, v), mask, mode=mode)
    assert torch.isfinite(out).all()
    assert torch.equal(out[..., 0, :], torch.zeros_like(out[..., 0, :]))
