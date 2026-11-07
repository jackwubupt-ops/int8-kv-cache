import math

import torch

from int8kv.metrics import first_divergence, kl_from_logits, operator_metrics, top1_agreement


def test_kl_is_zero_for_identical_logits():
    x = torch.randn(4, 10)
    assert kl_from_logits(x, x).item() == 0.0


def test_kl_matches_hand_computed_value():
    p, q = torch.tensor([[0.5, 0.5]]).log(), torch.tensor([[0.9, 0.1]]).log()
    expected = 0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1)
    assert abs(kl_from_logits(p, q).item() - expected) < 1e-6


def test_top1_agreement_fraction():
    a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    b = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    assert top1_agreement(a, b) == 0.5


def test_first_divergence_reports_index_or_none():
    assert first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert first_divergence([1, 2], [1, 2]) is None
    assert first_divergence([1, 2], [1, 2, 3]) == 2


def test_operator_metrics_zero_error_for_same_input():
    q, k, v = torch.randn(1, 2, 3, 8), torch.randn(1, 2, 6, 8), torch.randn(1, 2, 6, 8)
    m = operator_metrics(q, k, v, k, v)
    assert m["output_rel_l2"] < 1e-6
    assert m["attn_kl"] < 1e-6
    assert m["logit_abs_err"] < 1e-6
    assert abs(m["output_cosine"] - 1) < 1e-6
    assert m["k_rel_l2"] == 0 and m["v_rel_l2"] == 0


def test_operator_metrics_ignores_masked_keys_in_kl():
    q, k, v = torch.randn(1, 1, 2, 8), torch.randn(1, 1, 4, 8), torch.randn(1, 1, 4, 8)
    mask = torch.ones(2, 4, dtype=torch.bool)
    mask[:, -1] = False
    m = operator_metrics(q, k, v, k, v, mask)
    assert math.isfinite(m["attn_kl"]) and m["attn_kl"] < 1e-6
    assert math.isfinite(m["logit_abs_err"])
