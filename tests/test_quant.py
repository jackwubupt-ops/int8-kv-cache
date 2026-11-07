import pytest
import torch

from int8kv.quant import compute_scale, dequantize, kv_bytes_per_token, quantize, round_scale_up

GRANS = ["per_tensor", "per_head", "per_token", "per_token_head"]
SHAPES = {
    "per_tensor": (2, 1, 1, 1),
    "per_head": (2, 3, 1, 1),
    "per_token": (2, 1, 5, 1),
    "per_token_head": (2, 3, 5, 1),
}


@pytest.mark.parametrize("gran", GRANS)
def test_scale_shape_broadcasts_to_input(gran):
    x = torch.randn(2, 3, 5, 8)
    assert compute_scale(x, gran).shape == SHAPES[gran]


def test_unknown_granularity_rejected():
    with pytest.raises(ValueError, match="granularity"):
        compute_scale(torch.randn(1, 1, 1, 4), "per_cat")


@pytest.mark.parametrize("gran", GRANS)
def test_roundtrip_error_is_within_half_step(gran):
    x = torch.randn(2, 3, 5, 8) * 7
    s = compute_scale(x, gran)
    q = quantize(x, s)
    assert q.dtype == torch.int8
    assert (dequantize(q, s) - x).abs().max() <= (s / 2).max() * 1.0001


def test_absmax_maps_to_127():
    x = torch.tensor([[[[-2.0, 1.0, 0.5, 2.0]]]])
    q = quantize(x, compute_scale(x, "per_token_head"))
    assert q.tolist() == [[[[-127, 64, 32, 127]]]]


def test_all_zero_token_stays_finite():
    x = torch.zeros(1, 1, 2, 4)
    x[0, 0, 1] = 1.0
    s = compute_scale(x, "per_token_head")
    assert s[0, 0, 0, 0] == 1.0
    assert torch.equal(dequantize(quantize(x, s), s)[0, 0, 0], torch.zeros(4))


def test_non_finite_input_rejected():
    x = torch.ones(1, 1, 1, 4)
    x[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        compute_scale(x, "per_tensor")


def test_round_scale_up_never_rounds_down():
    s = torch.rand(2000) * torch.logspace(-6, 3, 2000)
    r = round_scale_up(s, torch.bfloat16)
    assert r.dtype == torch.bfloat16
    assert (r.to(torch.float32) >= s).all()


def test_bf16_scale_does_not_clip_values():
    x = torch.randn(1, 2, 200, 16)
    exact = compute_scale(x, "per_token_head")
    s = round_scale_up(exact, torch.bfloat16)
    ratio = x.abs() / s.to(torch.float32)
    assert ratio.max() <= 127.0


def test_kv_bytes_matches_design_numbers():
    assert kv_bytes_per_token(layers=28, kv_heads=2, head_dim=128, value_bytes=2, scale_bytes=0) == 28672
    assert kv_bytes_per_token(layers=28, kv_heads=2, head_dim=128, value_bytes=1, scale_bytes=2) == 14560


def test_round_scale_up_moves_at_most_one_representable_step():
    torch.manual_seed(0)
    s = torch.rand(20000) * torch.logspace(-6, 3, 20000)
    r = round_scale_up(s, torch.bfloat16).to(torch.float32)
    assert ((r - s) / s).max() <= 2.0**-7
