"""Per-channel / per-head INT8 KV: static scales, the K-scale fold into Q, and implementation A vs B."""
import pytest
import torch

from int8kv.attention import attention_float, attention_int8
from int8kv.quant import calibrate_scale, compute_scale, dequantize, quantize


def make(seed=0, tq=1, tk=64, heads=2, d=32, outlier_channel=3):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, heads, tq, d, generator=g)
    k = torch.randn(1, heads, tk, d, generator=g)
    k[..., outlier_channel] *= 40  # a fixed outlier channel, as in real Qwen K
    v = torch.randn(1, heads, tk, d, generator=g)
    return q, k, v


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def quant(x, gran):
    s = compute_scale(x, gran)
    return quantize(x, s), s


def test_per_channel_scale_shape():
    assert compute_scale(torch.randn(2, 3, 5, 8), "per_channel").shape == (2, 3, 1, 8)


def test_calibrate_scale_shapes_and_values():
    absmax = torch.tensor([[1.27, 2.54], [0.0, 12.7]])
    ch = calibrate_scale(absmax, "per_channel")
    assert ch.shape == (1, 2, 1, 2)
    assert torch.allclose(ch[0, :, 0], torch.tensor([[0.01, 0.02], [1.0, 0.1]]))
    head = calibrate_scale(absmax, "per_head")
    assert head.shape == (1, 2, 1, 1)
    assert torch.allclose(head.flatten(), torch.tensor([0.02, 0.1]))


def test_calibrate_scale_uses_quantile_when_given():
    absmax = torch.full((1, 2), 100.0)
    clip = torch.full((1, 2), 12.7)
    assert torch.allclose(calibrate_scale(absmax, "per_channel", clip), torch.full((1, 1, 1, 2), 0.1))


def test_calibrate_scale_rejects_token_dependent_granularity():
    with pytest.raises(ValueError, match="static"):
        calibrate_scale(torch.ones(1, 2), "per_token_head")


def test_per_channel_beats_per_head_on_outlier_channel():
    _, k, _ = make()
    errs = {g: rel(dequantize(*quant(k, g)), k) for g in ("per_head", "per_channel")}
    assert errs["per_channel"] < errs["per_head"] / 3


def test_integer_mode_folds_k_scale_into_q_exactly():
    """With per_channel K, integer mode must equal float attention on (q_hat, k_hat, v_hat),
    where q_hat is the quantize/dequantize of the folded q * s_k, divided back by s_k."""
    q, k, v = make(1)
    ki, sk = quant(k, "per_channel")
    vi, sv = quant(v, "per_channel")
    q_eff = q * sk
    sq = compute_scale(q_eff, "per_tensor")
    q_hat = dequantize(quantize(q_eff, sq), sq) / sk
    expected = attention_float(q_hat, dequantize(ki, sk), dequantize(vi, sv))
    out = attention_int8(q, ki, sk, vi, sv, mode="integer", q_granularity="per_tensor")
    assert torch.allclose(out, expected, atol=1e-5)


def test_naive_factoring_of_per_channel_scale_is_wrong():
    """The reason for the fold: pulling a per-channel s_k out of the head_dim sum is not the same math."""
    q, k, _ = make(2)
    ki, sk = quant(k, "per_channel")
    right = q @ dequantize(ki, sk).transpose(-2, -1)
    naive = (q @ ki.float().transpose(-2, -1)) * sk.mean(-1, keepdim=True)
    assert rel(naive, right) > 0.1


@pytest.mark.parametrize("gran", ["per_head", "per_channel", "per_token_head"])
def test_int8_mode_matches_integer_mode_without_p_rounding(gran):
    """int8 only adds P quantization on top of integer mode: the difference is P rounding error only."""
    q, k, v = make(3, tq=2)
    ki, sk = quant(k, gran)
    vi, sv = quant(v, gran)
    a = attention_int8(q, ki, sk, vi, sv, mode="integer", q_granularity="per_tensor")
    b = attention_int8(q, ki, sk, vi, sv, mode="int8", q_granularity="per_tensor")
    assert 0 < rel(b, a) < 0.05


def _static_errors(gran):
    q, k, v = make(4)
    sk = calibrate_scale(k.abs().amax(dim=(0, 2)), gran)
    sv = calibrate_scale(v.abs().amax(dim=(0, 2)), gran)
    ki, vi = quantize(k, sk), quantize(v, sv)
    base = attention_float(q, k, v)
    return {m: rel(attention_int8(q, ki, sk, vi, sv, mode=m), base) for m in ("dequant", "integer", "int8")}


def test_static_per_channel_all_modes_within_10_percent():
    errs = _static_errors("per_channel")
    assert max(errs.values()) < 0.10
    assert errs["dequant"] <= errs["int8"]


def test_static_per_head_breaks_on_outlier_channel():
    """One 40x channel sets the whole head's scale, so the other channels lose their precision."""
    assert _static_errors("per_head")["dequant"] > 2 * _static_errors("per_channel")["dequant"]


def test_static_scale_broadcasts_over_batch_and_clamps_outliers():
    q, k, v = make(5)
    sk = calibrate_scale(k[..., :32, :].abs().amax(dim=(0, 2)), "per_channel")
    k2 = torch.cat([k, k * 3])  # second batch item exceeds the calibrated range
    ki = quantize(k2, sk)
    assert ki.abs().max() == 127
    vi, sv = quant(torch.cat([v, v]), "per_channel")
    out = attention_int8(torch.cat([q, q]), ki, sk, vi, sv, mode="int8")
    assert out.shape == (2, 2, 1, 32) and torch.isfinite(out).all()


def test_mixed_k_channel_v_token_head():
    q, k, v = make(6)
    ki, sk = quant(k, "per_channel")
    vi, sv = quant(v, "per_token_head")
    base = attention_float(q, k, v)
    assert rel(attention_int8(q, ki, sk, vi, sv, mode="int8"), base) < 0.10


def test_int8_default_q_scale_survives_one_head_with_outlier_channel():
    """A per_tensor Q scale after the fold lets one head's outlier channel (q and k both large)
    swamp every other head; the default per-(token, head) Q scale keeps the other heads accurate."""
    q, k, v = make(5, tq=4)
    q[:, 0, :, 0] *= 100
    k[:, 0, :, 0] *= 100
    ki, sk = quant(k, "per_channel")
    vi, sv = quant(v, "per_channel")
    base = attention_float(q, k, v)[:, 1:]
    default = attention_int8(q, ki, sk, vi, sv, mode="int8")[:, 1:]
    per_tensor = attention_int8(q, ki, sk, vi, sv, mode="int8", q_granularity="per_tensor")[:, 1:]
    assert rel(default, base) < 0.1
    assert rel(per_tensor, base) > 0.5
