"""Accuracy metrics for comparing INT8 results against a float baseline."""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch

from .attention import _apply_mask, masked_softmax


def kl_from_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """Mean KL(p || q) over the last dim; keys with p=0 contribute nothing."""
    lp = p_logits.to(torch.float32).log_softmax(-1)
    lq = q_logits.to(torch.float32).log_softmax(-1)
    p = lp.exp()
    term = torch.where(p > 0, p * (lp - lq), torch.zeros_like(p))
    return term.sum(-1).mean()


def top1_agreement(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.argmax(-1) == b.argmax(-1)).float().mean().item()


def first_divergence(a: Sequence[int], b: Sequence[int]) -> Optional[int]:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _rel_l2(hat: torch.Tensor, ref: torch.Tensor) -> float:
    return ((hat - ref).norm() / ref.norm().clamp_min(1e-12)).item()


def operator_metrics(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_hat: torch.Tensor,
    v_hat: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    q_hat: Optional[torch.Tensor] = None,
) -> dict:
    """Compare attention on (q, k, v) against (q_hat, k_hat, v_hat)."""
    q, k, v = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
    k_hat, v_hat = k_hat.to(torch.float32), v_hat.to(torch.float32)
    q_hat = q if q_hat is None else q_hat.to(torch.float32)
    scale = 1 / math.sqrt(q.shape[-1])

    raw = q @ k.transpose(-2, -1) * scale
    raw_hat = q_hat @ k_hat.transpose(-2, -1) * scale
    logit_err = (raw_hat - raw).abs()
    if mask is not None:
        allowed = mask if mask.dtype == torch.bool else mask > -1e8
        logit_err = logit_err[allowed.expand_as(logit_err)]

    logits = _apply_mask(raw, mask)
    logits_hat = _apply_mask(raw_hat, mask)
    out = masked_softmax(raw, mask) @ v
    out_hat = masked_softmax(raw_hat, mask) @ v_hat
    return {
        "k_rel_l2": _rel_l2(k_hat, k),
        "v_rel_l2": _rel_l2(v_hat, v),
        "logit_abs_err": logit_err.mean().item(),
        "attn_kl": kl_from_logits(logits, logits_hat).item(),
        "output_rel_l2": _rel_l2(out_hat, out),
        "output_cosine": torch.cosine_similarity(out_hat, out, dim=-1).mean().item(),
    }
