"""Route Qwen2Attention through INT8 attention when given a QuantizedKVCache."""
from __future__ import annotations

from collections import defaultdict
from types import MethodType
from typing import Optional

import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb

from .attention import attention_int8
from .cache import QuantizedKVCache
from .metrics import operator_metrics
from .quant import compute_scale, dequantize, quantize


class OperatorRecorder:
    """Accumulates operator metrics by layer and phase."""

    def __init__(self):
        self.rows = []

    def record(self, layer_idx: int, phase: str, metrics: dict) -> None:
        self.rows.append({"layer": layer_idx, "phase": phase, **metrics})

    def summary(self) -> dict:
        grouped = defaultdict(list)
        for row in self.rows:
            grouped[row["phase"]].append(row)
        result = {}
        for phase, rows in grouped.items():
            keys = [k for k in rows[0] if k not in ("layer", "phase")]
            result[phase] = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
            result[phase]["samples"] = len(rows)
        return result


def _causal_mask_if_needed(mask: Optional[torch.Tensor], q: torch.Tensor, total_keys: int) -> Optional[torch.Tensor]:
    """Transformers passes mask=None when SDPA is expected to apply causality itself."""
    if mask is not None or q.shape[-2] == 1:
        return mask
    offset = total_keys - q.shape[-2]
    keys = torch.arange(total_keys, device=q.device)
    queries = torch.arange(q.shape[-2], device=q.device).unsqueeze(-1) + offset
    return keys <= queries


def _repeat_heads(x: torch.Tensor, groups: int) -> torch.Tensor:
    return x if groups == 1 else x.repeat_interleave(groups, dim=1)


def patch_qwen2(model, recorder: Optional[OperatorRecorder] = None) -> int:
    """Patch every Qwen2Attention in model; returns how many were patched."""
    count = 0
    for module in model.modules():
        if isinstance(module, Qwen2Attention):
            module.forward = MethodType(_make_forward(module.forward, recorder), module)
            count += 1
    if count == 0:
        raise ValueError("no Qwen2Attention found; only Qwen2-architecture models are supported")
    return count


def _make_forward(original, recorder: Optional[OperatorRecorder]):
    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None,
                cache_position=None, **kwargs):
        if not isinstance(past_key_values, QuantizedKVCache):
            return original(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        if self.sliding_window is not None:
            raise NotImplementedError("sliding-window attention is out of scope for INT8 KV")

        cache = past_key_values
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        phase = "decode" if cache.get_seq_length(self.layer_idx) > 0 else "prefill"
        cache.update(k, v, self.layer_idx)
        k_int, s_k, v_int, s_v = cache.get_kv(self.layer_idx)
        groups = self.num_key_value_groups
        k_int, s_k = _repeat_heads(k_int, groups), _repeat_heads(s_k, groups)
        v_int, s_v = _repeat_heads(v_int, groups), _repeat_heads(s_v, groups)
        mask = _causal_mask_if_needed(attention_mask, q, k_int.shape[-2])

        if recorder is not None:
            _record(recorder, cache, self.layer_idx, phase, q, k_int, s_k, v_int, s_v, mask, groups)
        output = attention_int8(q, k_int, s_k, v_int, s_v, mask, mode=cache.mode)
        output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(output), None

    return forward


def _record(recorder, cache, layer_idx, phase, q, k_int, s_k, v_int, s_v, mask, groups) -> None:
    if not cache.keep_float:
        raise ValueError("operator recording needs QuantizedKVCache(keep_float=True)")
    f_k, f_v = cache.get_float_kv(layer_idx)
    q_hat = None
    if cache.mode == "integer":
        s_q = compute_scale(q)
        q_hat = dequantize(quantize(q, s_q), s_q)
    recorder.record(
        layer_idx,
        phase,
        operator_metrics(
            q, _repeat_heads(f_k, groups), _repeat_heads(f_v, groups),
            dequantize(k_int, s_k), dequantize(v_int, s_v), mask, q_hat,
        ),
    )
