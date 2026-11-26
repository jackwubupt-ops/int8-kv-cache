"""INT8 quantization for tensors laid out as [batch, heads, tokens, head_dim]."""
from __future__ import annotations

from typing import Optional

import torch

QMAX = 127
GRANULARITIES = ("per_tensor", "per_head", "per_token", "per_token_head", "per_channel")
_REDUCE_DIMS = {
    "per_tensor": (-3, -2, -1),
    "per_head": (-2, -1),
    "per_token": (-3, -1),
    "per_token_head": (-1,),
    # One scale per (head, head_dim): reduce over tokens only.
    "per_channel": (-2,),
}
# Granularities whose scale does not depend on the token, so it can be calibrated once and fixed.
STATIC_GRANULARITIES = ("per_head", "per_channel")


def compute_scale(x: torch.Tensor, granularity: str = "per_token_head") -> torch.Tensor:
    """Return absmax / 127, keeping dims so the result broadcasts against x."""
    if granularity not in _REDUCE_DIMS:
        raise ValueError(f"unknown granularity {granularity!r}, expected one of {GRANULARITIES}")
    x = x.detach().to(torch.float32)
    if not torch.isfinite(x).all():
        raise ValueError("cannot quantize a tensor with non-finite values")
    absmax = x.abs().amax(dim=_REDUCE_DIMS[granularity], keepdim=True)
    scale = absmax / QMAX
    # An all-zero group quantizes to zeros under any scale; 1 avoids dividing by 0.
    return torch.where(absmax > 0, scale, torch.ones_like(scale))


def calibrate_scale(
    absmax: torch.Tensor, granularity: str, quantile_values: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Static scale from calibration statistics.

    absmax: [heads, head_dim], the max |x| seen per channel over all calibration tokens.
    quantile_values: optional [heads, head_dim] percentile of |x| to clip outliers instead of absmax.
    Returns [1, heads, 1, head_dim] for per_channel or [1, heads, 1, 1] for per_head.
    """
    if granularity not in STATIC_GRANULARITIES:
        raise ValueError(f"static scales need one of {STATIC_GRANULARITIES}, got {granularity!r}")
    bound = (absmax if quantile_values is None else quantile_values).to(torch.float32)
    if granularity == "per_head":
        bound = bound.amax(dim=-1, keepdim=True)
    scale = bound / QMAX
    scale = torch.where(bound > 0, scale, torch.ones_like(scale))
    return scale.unsqueeze(0).unsqueeze(-2)


def round_scale_up(scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast scale to a narrower dtype without ever rounding it down."""
    cast = scale.to(dtype)
    too_small = cast.to(scale.dtype) < scale
    if too_small.any():
        wide = cast.to(scale.dtype)
        # frexp has no MPS kernel; the scale tensors are tiny, so do this step on the CPU there.
        host = wide.cpu() if wide.device.type == "mps" else wide
        _, exponent = torch.frexp(host)
        ulp = torch.ldexp(torch.full_like(host, torch.finfo(dtype).eps), exponent - 1).to(wide.device)
        cast = torch.where(too_small, (wide + ulp).to(dtype), cast)
    return cast


def quantize(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    ratio = x.to(torch.float32) / scale.to(torch.float32)
    return ratio.round().clamp(-QMAX, QMAX).to(torch.int8)


def dequantize(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return (q.to(torch.float32) * scale.to(torch.float32)).to(dtype)


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int, value_bytes: int, scale_bytes: int) -> int:
    """Bytes for one token K+V across all layers; scale_bytes is per (token, head) for each of K and V."""
    return layers * kv_heads * 2 * (head_dim * value_bytes + scale_bytes)
