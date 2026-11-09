"""FP32 reference attention shared by the fused decode (Task 6) and extend (Task 7) tests."""
import math
from typing import Optional

import torch


def reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        causal_offset: Optional[int] = None, sm_scale: Optional[float] = None) -> torch.Tensor:
    """q [Tq, Hq, D], k [Tk, Hkv, D], v [Tk, Hkv, Dv] -> [Tq, Hq, Dv] in FP32.

    Query heads are grouped onto KV heads (GQA). With causal_offset, query i sees keys j <= i + causal_offset.
    """
    tq, hq, d = q.shape
    tk, hkv, _ = k.shape
    group = hq // hkv
    qf = q.float().permute(1, 0, 2)
    kf = k.float().repeat_interleave(group, dim=1).permute(1, 0, 2)
    vf = v.float().repeat_interleave(group, dim=1).permute(1, 0, 2)
    scores = qf @ kf.transpose(1, 2) * (sm_scale if sm_scale is not None else 1.0 / math.sqrt(d))
    if causal_offset is not None:
        i = torch.arange(tq, device=q.device)[:, None]
        j = torch.arange(tk, device=q.device)[None, :]
        scores = scores.masked_fill(j > i + causal_offset, float("-inf"))
    return (scores.softmax(dim=-1) @ vf).permute(1, 0, 2)
