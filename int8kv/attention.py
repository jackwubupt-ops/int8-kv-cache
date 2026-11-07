"""Baseline and INT8 scaled dot-product attention."""
from __future__ import annotations

import math
from typing import Optional

import torch

from .quant import QMAX, compute_scale, dequantize, quantize

MODES = ("dequant", "integer")
_FP32_EXACT_LIMIT = 2**24


def _apply_mask(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return logits
    if mask.dtype == torch.bool:
        return logits.masked_fill(~mask, float("-inf"))
    return logits + mask.to(logits.dtype)


def masked_softmax(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Softmax over allowed keys; a query with no allowed key gets all-zero weights."""
    weights = _apply_mask(logits, mask).softmax(dim=-1)
    return torch.nan_to_num(weights, nan=0.0)


def attention_float(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Reference attention: softmax(QK^T / sqrt(d)) V, computed in FP32."""
    q32, k32, v32 = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
    logits = q32 @ k32.transpose(-2, -1) / math.sqrt(q.shape[-1])
    weights = masked_softmax(logits, mask)
    return (weights @ v32).to(q.dtype)


def integer_dot(q_int: torch.Tensor, k_int: torch.Tensor) -> torch.Tensor:
    """Exact Q_int @ K_int^T.

    GPU matmul has no integer path, so integers are carried in a float type
    that can represent every partial sum exactly.
    """
    bound = q_int.shape[-1] * QMAX * QMAX
    dtype = torch.float32 if bound < _FP32_EXACT_LIMIT else torch.float64
    return q_int.to(dtype) @ k_int.to(dtype).transpose(-2, -1)


def attention_int8(
    q: torch.Tensor,
    k_int: torch.Tensor,
    s_k: torch.Tensor,
    v_int: torch.Tensor,
    s_v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    mode: str = "dequant",
    q_granularity: str = "per_token_head",
) -> torch.Tensor:
    """Attention over an INT8 KV cache.

    dequant: restore K/V to float, Q stays float.
    integer: quantize Q as well and compute QK^T from integers plus scales.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
    if mode == "dequant":
        return attention_float(q, dequantize(k_int, s_k), dequantize(v_int, s_v), mask).to(q.dtype)

    s_q = compute_scale(q, q_granularity)
    dot = integer_dot(quantize(q, s_q), k_int).to(torch.float32)
    logits = dot * s_q.to(torch.float32) * s_k.to(torch.float32).transpose(-2, -1) / math.sqrt(q.shape[-1])
    weights = masked_softmax(logits, mask)
    # Per-token V scales cannot be factored out, so they weight each key.
    weighted = weights * s_v.to(torch.float32).transpose(-2, -1)
    return (weighted @ v_int.to(torch.float32)).to(q.dtype)
