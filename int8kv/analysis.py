"""KV distribution statistics used to choose the quantization granularity."""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import torch

from .quant import GRANULARITIES, compute_scale, dequantize, quantize

ABSMAX_BINS = np.logspace(-4, 3, 71)
SCALE_BYTES = 2


class GranularityStats:
    """Reconstruction error and scale overhead per granularity, over write chunks."""

    def __init__(self):
        self.squared_error = {g: 0.0 for g in GRANULARITIES}
        self.scale_elements = {g: 0 for g in GRANULARITIES}
        self.squared_norm = 0.0
        self.token_heads = 0

    def add_chunk(self, x: torch.Tensor) -> None:
        x = x.to(torch.float32)
        self.squared_norm += x.pow(2).sum().item()
        self.token_heads += x.shape[0] * x.shape[1] * x.shape[2]
        for gran in GRANULARITIES:
            scale = compute_scale(x, gran)
            self.squared_error[gran] += (dequantize(quantize(x, scale), scale) - x).pow(2).sum().item()
            self.scale_elements[gran] += scale.numel()

    def table(self) -> Dict[str, dict]:
        return {
            gran: {
                "rel_l2": (self.squared_error[gran] / max(self.squared_norm, 1e-24)) ** 0.5,
                "scale_bytes_per_token_head": self.scale_elements[gran] * SCALE_BYTES / max(self.token_heads, 1),
            }
            for gran in GRANULARITIES
        }


def write_chunks(x: torch.Tensor, prompt_len: int) -> Iterable[torch.Tensor]:
    """Split a cached tensor as it was written: one prefill chunk, then one token per decode step."""
    yield x[..., :prompt_len, :]
    for i in range(prompt_len, x.shape[-2]):
        yield x[..., i : i + 1, :]


def recommend_granularity(table: Dict[str, dict], tolerance: float = 1.1) -> str:
    """Pick the cheapest granularity whose error is within tolerance of the best."""
    best = min(row["rel_l2"] for row in table.values())
    eligible: List[str] = [g for g in GRANULARITIES if table[g]["rel_l2"] <= best * tolerance]
    return min(eligible, key=lambda g: table[g]["scale_bytes_per_token_head"])


def absmax_histogram(x: torch.Tensor) -> np.ndarray:
    absmax = x.to(torch.float32).abs().amax(dim=-1).cpu().numpy().ravel()
    return np.histogram(np.clip(absmax, ABSMAX_BINS[0], ABSMAX_BINS[-1]), bins=ABSMAX_BINS)[0]


def crest_factor(x: torch.Tensor) -> torch.Tensor:
    """absmax / rms per (token, head): large values mean a few outliers waste INT8 range."""
    x = x.to(torch.float32)
    rms = x.pow(2).mean(dim=-1).sqrt().clamp_min(1e-12)
    return x.abs().amax(dim=-1) / rms
