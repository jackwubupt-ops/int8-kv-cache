"""A DynamicCache that stores K/V as INT8 plus quantization scales."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from transformers import DynamicCache

from .quant import STATIC_GRANULARITIES, compute_scale, quantize, round_scale_up

SCALE_SOURCES = ("write", "dynamic", "static")


class QuantizedKVCache(DynamicCache):
    """INT8 K/V plus scales. Layout: [batch, kv_heads, tokens, head_dim].

    scale_source:
      write   - scale from each written chunk (prefill chunk, then one decode token); stored INT8.
      dynamic - the cache stays float; at every read the scale is the absmax over all cached tokens
                and the whole cache is quantized on the fly (accuracy check only, no memory saving).
      static  - fixed per-layer scales from calibration (static_scales[layer] = (s_k, s_v));
                K/V are quantized once on write and stored INT8.
    """

    def __init__(
        self,
        granularity: str = "per_token_head",
        mode: str = "dequant",
        scale_dtype: torch.dtype = torch.bfloat16,
        keep_float: bool = False,
        float_prefill: bool = False,
        scale_source: str = "write",
        static_scales: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        super().__init__()
        if scale_source not in SCALE_SOURCES:
            raise ValueError(f"unknown scale_source {scale_source!r}, expected one of {SCALE_SOURCES}")
        if scale_source == "write" and granularity == "per_channel":
            raise ValueError("per_channel reduces over tokens; use scale_source='dynamic' or 'static'")
        if scale_source in ("dynamic", "static") and granularity not in STATIC_GRANULARITIES:
            raise ValueError(f"scale_source={scale_source!r} needs one of {STATIC_GRANULARITIES}")
        if scale_source == "static" and not static_scales:
            raise ValueError("scale_source='static' needs static_scales")
        self.scale_source = scale_source
        self.static_scales = static_scales
        self.granularity = granularity
        self.mode = mode
        self.scale_dtype = scale_dtype
        self.keep_float = keep_float
        # Like vLLM/SGLang: attention for a multi-token chunk uses the chunk's own float K/V,
        # and only later steps read them back from INT8.
        self.float_prefill = float_prefill
        self._k_scales: List[torch.Tensor] = []
        self._v_scales: List[torch.Tensor] = []
        self._floats: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._stored_scale_elements = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if self.scale_source == "dynamic":
            return super().update(key_states, value_states, layer_idx, cache_kwargs)
        if self.scale_source == "static":
            s_k, s_v = self._static(layer_idx, key_states)
            if layer_idx == len(self._k_scales):
                self._k_scales.append(s_k)
                self._v_scales.append(s_v)
                self._stored_scale_elements += s_k.numel() + s_v.numel()
        else:
            s_k = self._make_scale(key_states)
            s_v = self._make_scale(value_states)
            self._stored_scale_elements += s_k.numel() + s_v.numel()
            self._append_scale(self._k_scales, s_k, key_states, layer_idx)
            self._append_scale(self._v_scales, s_v, value_states, layer_idx)
        k_int, v_int = quantize(key_states, s_k), quantize(value_states, s_v)
        if self.keep_float:
            prev = self._floats.get(layer_idx)
            f_k, f_v = key_states.detach(), value_states.detach()
            self._floats[layer_idx] = (
                (f_k, f_v) if prev is None else (torch.cat([prev[0], f_k], -2), torch.cat([prev[1], f_v], -2))
            )
        return super().update(k_int, v_int, layer_idx, cache_kwargs)

    def _static(self, layer_idx: int, like: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s_k, s_v = self.static_scales[layer_idx]
        return tuple(round_scale_up(s.to(like.device, torch.float32), self.scale_dtype) for s in (s_k, s_v))

    def _make_scale(self, x: torch.Tensor) -> torch.Tensor:
        return round_scale_up(compute_scale(x, self.granularity), self.scale_dtype)

    @staticmethod
    def _append_scale(store: List[torch.Tensor], scale: torch.Tensor, x: torch.Tensor, layer_idx: int) -> None:
        # Coarse scales cover one write; expand so each token slot can find its own.
        expanded = scale.expand(*x.shape[:-1], 1)
        if layer_idx == len(store):
            store.append(expanded)
        else:
            store[layer_idx] = torch.cat([store[layer_idx], expanded], dim=-2)

    def get_kv(self, layer_idx: int):
        layer = self.layers[layer_idx]
        if self.scale_source == "dynamic":
            s_k, s_v = self._make_scale(layer.keys), self._make_scale(layer.values)
            return quantize(layer.keys, s_k), s_k, quantize(layer.values, s_v), s_v
        return layer.keys, self._k_scales[layer_idx], layer.values, self._v_scales[layer_idx]

    def get_float_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.scale_source == "dynamic":
            return self.layers[layer_idx].keys, self.layers[layer_idx].values
        return self._floats[layer_idx]

    def crop(self, max_length: int) -> None:
        super().crop(max_length)
        for store in (self._k_scales, self._v_scales):
            for i, scale in enumerate(store):
                store[i] = scale[..., : self.get_seq_length(i), :]
        for i, (f_k, f_v) in self._floats.items():
            self._floats[i] = (f_k[..., :max_length, :], f_v[..., :max_length, :])

    def batch_repeat_interleave(self, repeats: int) -> None:
        raise NotImplementedError("QuantizedKVCache supports greedy decoding only")

    def memory_bytes(self) -> dict:
        kv = sum(t.numel() * t.element_size() for layer in self.layers for t in (layer.keys, layer.values))
        scale_bytes = self._stored_scale_elements * torch.finfo(self.scale_dtype).bits // 8
        return {"kv_bytes": kv, "scale_bytes": scale_bytes, "total_bytes": kv + scale_bytes}
