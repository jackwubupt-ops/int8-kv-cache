"""Pure functions of experiments/stage4/fp8_probe.py."""
import importlib.util
from pathlib import Path

import pytest
import torch

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "stage4" / "fp8_probe.py"
_spec = importlib.util.spec_from_file_location("fp8_probe", _PATH)
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)


def test_roundtrip_formats():
    x = torch.tensor([[[1.0, -2.5, 300.0, 0.0]]])
    assert torch.equal(P.roundtrip(x, "fp8_e5m2"), x.to(torch.float8_e5m2).float())
    assert torch.equal(P.roundtrip(x, "fp8_e4m3"), x.to(torch.float8_e4m3fn).float())
    q = P.roundtrip(x, "int8_per_token_head")
    assert (q - x).abs().max() <= 300 / 127 / 2 + 1e-4


def test_e4m3_overflow_is_counted():
    x = torch.tensor([[[100.0, 500.0, -600.0, 1.0]]])
    stats = P.layer_stats(x)
    assert stats["e4m3_overflow_frac"] == pytest.approx(0.5)


def test_centered_error_removes_the_common_channel_offset():
    """A large per-channel offset shared by all tokens (like a K bias) does not enter the centered norm."""
    g = torch.Generator().manual_seed(0)
    signal = torch.randn(64, 2, 16, generator=g)
    offset = torch.zeros(1, 2, 16)
    offset[..., 3] = 300.0
    x = signal + offset
    stats = P.layer_stats(x)
    assert stats["centered_norm_ratio"] < 0.1
    assert stats["fp8_e5m2"]["centered_rel_l2"] > 5 * stats["fp8_e5m2"]["rel_l2"]
    assert stats["int8_per_token_head"]["centered_rel_l2"] < stats["fp8_e5m2"]["centered_rel_l2"]


def test_calibration_scale_keeps_the_layer_absmax_inside_e4m3_with_margin():
    scales = P.calibration_scales(k_absmax=[448.0, 10.0], v_absmax=[4.0, 896.0], margin=2.0)
    assert scales == pytest.approx([2.0, 4.0])  # margin * max(k, v) / 448, one scale per layer for K and V


def test_calibration_scale_never_goes_below_one_for_small_layers():
    """Scaling small tensors up gains nothing in a floating format; keep SGLang's default of 1.0 there."""
    assert P.calibration_scales(k_absmax=[3.0], v_absmax=[1.0], margin=2.0) == [1.0]


def test_quant_param_json_matches_sglang_schema():
    doc = P.quant_param_json([1.5, 2.0], model_type="qwen2")
    assert doc == {"model_type": "qwen2",
                   "kv_cache": {"dtype": "float8_e4m3fn", "scaling_factor": {"0": {"0": 1.5, "1": 2.0}}}}


def test_calibrated_roundtrip_removes_overflow():
    x = torch.tensor([[[100.0, 500.0, -600.0, 1.0]]])
    stats = P.layer_stats(x, calib_scale=P.calibration_scales([600.0], [0.0], margin=2.0)[0])
    assert stats["fp8_e4m3"]["nan_frac"] == pytest.approx(0.5)
    assert stats["fp8_e4m3_calib"]["nan_frac"] == 0.0
    assert stats["fp8_e4m3_calib"]["rel_l2"] < 0.07  # 3 mantissa bits: at most 1/16 relative error


def test_running_absmax_takes_the_max_over_prompts():
    acc = P.AbsmaxTracker(num_layers=2)
    acc.update(0, torch.tensor([1.0, -5.0]), torch.tensor([2.0]))
    acc.update(0, torch.tensor([3.0]), torch.tensor([-7.0]))
    acc.update(1, torch.tensor([0.5]), torch.tensor([0.25]))
    assert acc.k == [5.0, 0.5] and acc.v == [7.0, 0.25]
