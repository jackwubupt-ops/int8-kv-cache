"""A DynamicCache that stores K/V as INT8 plus quantization scales."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from transformers import DynamicCache

from .quant import compute_scale, quantize, round_scale_up


class QuantizedKVCache(DynamicCache):
    """Quantizes new K/V on write. Layout: [batch, kv_heads, tokens, head_dim]."""

    def __init__(
        self,
        granularity: str = "per_token_head",
        mode: str = "dequant",
        scale_dtype: torch.dtype = torch.bfloat16,
        keep_float: bool = False,
        float_prefill: bool = False,
    ):
        super().__init__()
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
        s_k = self._make_scale(key_states)
        s_v = self._make_scale(value_states)
        k_int, v_int = quantize(key_states, s_k), quantize(value_states, s_v)
        self._stored_scale_elements += s_k.numel() + s_v.numel()
        self._append_scale(self._k_scales, s_k, key_states, layer_idx)
        self._append_scale(self._v_scales, s_v, value_states, layer_idx)
        if self.keep_float:
            prev = self._floats.get(layer_idx)
            f_k, f_v = key_states.detach(), value_states.detach()
            self._floats[layer_idx] = (
                (f_k, f_v) if prev is None else (torch.cat([prev[0], f_k], -2), torch.cat([prev[1], f_v], -2))
            )
        return super().update(k_int, v_int, layer_idx, cache_kwargs)

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
        return layer.keys, self._k_scales[layer_idx], layer.values, self._v_scales[layer_idx]

    def get_float_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
        kv = sum(t.numel() for layer in self.layers for t in (layer.keys, layer.values))
        scale_bytes = self._stored_scale_elements * torch.finfo(self.scale_dtype).bits // 8
        return {"kv_bytes": kv, "scale_bytes": scale_bytes, "total_bytes": kv + scale_bytes}
