"""Collect per-channel K/V statistics and turn them into static INT8 scales."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import torch

from .quant import calibrate_scale

TOP_K = 512


class ChannelStats:
    """Running absmax and the TOP_K largest |x| per (layer, K/V, head, channel).

    Keeping the top values instead of all of them gives an exact high quantile (e.g. 99.9%)
    as long as (1 - q) * tokens < TOP_K, at a fixed memory cost.
    """

    def __init__(self, top_k: int = TOP_K):
        self.top_k = top_k
        self.tokens: Dict[int, int] = {}
        self.top: Dict[Tuple[int, str], torch.Tensor] = {}

    def add(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        """keys/values: [batch, kv_heads, tokens, head_dim]."""
        for kind, x in (("k", keys), ("v", values)):
            a = x.detach().abs().to(torch.float32).cpu().permute(0, 2, 1, 3).reshape(-1, x.shape[1], x.shape[3])
            prev = self.top.get((layer_idx, kind))
            joined = a if prev is None else torch.cat([prev, a])
            self.top[(layer_idx, kind)] = joined.topk(min(self.top_k, joined.shape[0]), dim=0).values
        self.tokens[layer_idx] = self.tokens.get(layer_idx, 0) + keys.shape[0] * keys.shape[2]

    def absmax(self, layer_idx: int, kind: str) -> torch.Tensor:
        return self.top[(layer_idx, kind)][0]

    def quantile(self, layer_idx: int, kind: str, q: float) -> torch.Tensor:
        """The q-quantile of |x| per channel: the value with floor((1-q) * n) larger values."""
        rank = math.floor((1 - q) * self.tokens[layer_idx])
        top = self.top[(layer_idx, kind)]
        if rank >= top.shape[0]:
            raise ValueError(f"quantile {q} needs rank {rank} but only {top.shape[0]} top values are kept")
        return top[rank]

    def layers(self):
        return sorted(self.tokens)

    def save(self, path: Path, quantile: float) -> None:
        data = {"tokens": self.tokens, "quantile": quantile, "layers": {}}
        for i in self.layers():
            data["layers"][i] = {
                f"{kind}_{name}": t
                for kind in ("k", "v")
                for name, t in (("absmax", self.absmax(i, kind)), ("quantile", self.quantile(i, kind, quantile)))
            }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, path)

    def channel_spread(self) -> dict:
        """How uneven the channels are: per head, max channel absmax / median channel absmax.

        Large values inside a head mean per_head wastes range and per_channel is needed;
        large differences between heads with flat channels mean per_head is enough.
        """
        out = {}
        for kind in ("k", "v"):
            within, between = [], []
            for i in self.layers():
                m = self.absmax(i, kind)  # [heads, head_dim]
                within.append((m.amax(-1) / m.median(-1).values.clamp_min(1e-12)))
                head_max = m.amax(-1)
                between.append(head_max.max() / head_max.min().clamp_min(1e-12))
            within_all = torch.cat(within)
            out[kind] = {
                "within_head_max_over_median_p50": within_all.median().item(),
                "within_head_max_over_median_max": within_all.max().item(),
                "between_heads_max_over_min_p50": torch.stack(between).median().item(),
                "per_layer_within_head_max": [w.max().item() for w in within],
            }
        return out


def load_static_scales(path: Path, granularity: str, use_quantile: bool) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """{layer: (s_k, s_v)} shaped [1, heads, 1, head_dim] (per_channel) or [1, heads, 1, 1] (per_head)."""
    data = torch.load(path, map_location="cpu")
    name = "quantile" if use_quantile else "absmax"
    return {
        int(i): (calibrate_scale(row[f"k_{name}"], granularity), calibrate_scale(row[f"v_{name}"], granularity))
        for i, row in data["layers"].items()
    }


def build_cache(cfg: dict, scales_file: Path, use_quantile: bool, keep_float: bool = False, float_prefill: bool = True):
    """QuantizedKVCache for a parsed config dict (granularity, mode, scale_source)."""
    from .cache import QuantizedKVCache

    scales = None
    if cfg["scale_source"] == "static":
        scales = load_static_scales(scales_file, cfg["granularity"], use_quantile)
    return QuantizedKVCache(
        granularity=cfg["granularity"], mode=cfg["mode"], scale_source=cfg["scale_source"],
        static_scales=scales, keep_float=keep_float, float_prefill=float_prefill,
    )
