"""Baseline and INT8 scaled dot-product attention."""
from __future__ import annotations

import math
from typing import Optional

import torch

from .quant import QMAX, compute_scale, dequantize, quantize

MODES = ("dequant", "integer", "int8")
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


def _quantize_rows(p: torch.Tensor):
    """Dynamic INT8 for softmax weights: one scale per query row.

    The row is not contracted in P @ V, so its scale factors out of the matmul.
    """
    scale = p.amax(dim=-1, keepdim=True) / QMAX
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return quantize(p, scale), scale


def attention_int8(
    q: torch.Tensor,
    k_int: torch.Tensor,
    s_k: torch.Tensor,
    v_int: torch.Tensor,
    s_v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    mode: str = "dequant",
    q_granularity: Optional[str] = None,
) -> torch.Tensor:
    """Attention over an INT8 KV cache.

    dequant (implementation A): restore K/V to float, the attention itself is unchanged.
    integer: quantize Q as well and compute QK^T from integers; softmax weights stay FP32.
    int8 (implementation B): integer QK^T and integer PV, i.e. what an INT8 tensor-core kernel computes.

    Scales factor out of a matmul only when they do not depend on the contracted dimension:
    - QK^T contracts head_dim. A per_channel K scale [.., 1, head_dim] depends on it, so it is folded
      into Q first (q * s_k), and the folded Q gets its own scale. Token/head K scales factor out.
    - PV contracts tokens. A per_channel V scale [.., 1, head_dim] factors out onto the output;
      a per-token V scale [.., tokens, 1] does not, so it weights P before P is quantized.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
    if mode == "dequant":
        return attention_float(q, dequantize(k_int, s_k), dequantize(v_int, s_v), mask).to(q.dtype)
    if q_granularity is None:
        # One Q scale per (query token, head): a row of QK^T, so it still factors out of the int8 GEMM.
        # A single per_tensor scale is not enough once a per_channel s_k is folded into Q: one head's
        # outlier channel sets the scale for every head and rounds the other heads' Q to ~0, which made
        # Qwen2.5 generate garbage.
        q_granularity = "per_token_head"

    s_k32, s_v32 = s_k.to(torch.float32), s_v.to(torch.float32)
    k_channel = s_k.shape[-1] > 1
    v_channel = s_v.shape[-1] > 1
    q32 = q.to(torch.float32)
    q_eff = q32 * s_k32 if k_channel else q32
    s_q = compute_scale(q_eff, q_granularity)
    dot = integer_dot(quantize(q_eff, s_q), k_int).to(torch.float32)
    logits = dot * s_q if k_channel else dot * s_q * s_k32.transpose(-2, -1)
    weights = masked_softmax(logits / math.sqrt(q.shape[-1]), mask)

    if not v_channel:
        # Per-token V scales cannot be factored out, so they weight each key.
        weights = weights * s_v32.transpose(-2, -1)
    if mode == "int8":
        p_int, s_p = _quantize_rows(weights)
        # FP32 carries the integer sums; a real kernel accumulates them in INT32.
        out = (p_int.to(torch.float32) @ v_int.to(torch.float32)) * s_p
    else:
        out = weights @ v_int.to(torch.float32)
    if v_channel:
        out = out * s_v32
    return out.to(q.dtype)
