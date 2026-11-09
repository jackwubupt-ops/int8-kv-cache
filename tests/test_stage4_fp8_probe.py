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
