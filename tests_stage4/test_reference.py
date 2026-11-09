import math

import torch

from tests_stage4.reference import reference_attention


def test_single_key_returns_its_value():
    q, k, v = torch.randn(1, 4, 8), torch.randn(1, 2, 8), torch.randn(1, 2, 8)
    out = reference_attention(q, k, v)
    assert torch.allclose(out, v.repeat_interleave(2, dim=1))


def test_gqa_matches_per_head_softmax():
    torch.manual_seed(0)
    q, k, v = torch.randn(3, 4, 8), torch.randn(5, 2, 8), torch.randn(5, 2, 8)
    out = reference_attention(q, k, v)
    h, kvh = 3, 1  # query head 3 uses kv head 1
    w = (q[:, h] @ k[:, kvh].T / math.sqrt(8)).softmax(-1)
    assert torch.allclose(out[:, h], w @ v[:, kvh], atol=1e-6)


def test_causal_offset_hides_future_keys():
    torch.manual_seed(1)
    q, k, v = torch.randn(2, 2, 8), torch.randn(4, 2, 8), torch.randn(4, 2, 8)
    out = reference_attention(q, k, v, causal_offset=2)
    assert torch.allclose(out[0], reference_attention(q[:1], k[:3], v[:3])[0], atol=1e-6)
