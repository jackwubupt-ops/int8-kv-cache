"""INT8 quantization for tensors laid out as [batch, heads, tokens, head_dim]."""
from __future__ import annotations

import torch

QMAX = 127
GRANULARITIES = ("per_tensor", "per_head", "per_token", "per_token_head")
_REDUCE_DIMS = {
    "per_tensor": (-3, -2, -1),
    "per_head": (-2, -1),
    "per_token": (-3, -1),
    "per_token_head": (-1,),
}


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


def round_scale_up(scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast scale to a narrower dtype without ever rounding it down."""
    cast = scale.to(dtype)
    too_small = cast.to(scale.dtype) < scale
    if too_small.any():
        wide = cast.to(scale.dtype)
        _, exponent = torch.frexp(wide)
        ulp = torch.ldexp(torch.full_like(wide, torch.finfo(dtype).eps), exponent - 1)
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
