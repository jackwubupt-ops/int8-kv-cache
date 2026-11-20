"""KV cache storage formats compared in pure PyTorch, independent of SGLang.

Each format stores K/V after a quantize -> dequantize round trip and the unmodified HF model reads them back:

  fp8_e4m3             cast to float8_e4m3fn at scale 1.0 (SGLang without checkpoint scales); overflows past 448
  fp8_e5m2             cast to float8_e5m2 at scale 1.0
  fp8_e4m3_calib       e4m3 with one calibrated static scale per layer (fp8_probe.py --calib-out): FP8's best case
  int8_per_token_head  this project's format: one dynamic scale per (token, head)

Like SGLang, a multi-token chunk attends to its own float K/V and later steps read it from the cache. Reports, per
format, teacher-forced top-1 agreement and KL against the float model, greedy-generation agreement, needle
retrieval and (optionally) paired GSM8K accuracy. This separates what a format costs from how SGLang implements it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from transformers import DynamicCache

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from int8kv.metrics import kl_from_logits, mcnemar_exact, top1_agreement  # noqa: E402
from int8kv.quant import compute_scale, dequantize, quantize, round_scale_up  # noqa: E402

FORMATS = ("fp8_e4m3", "fp8_e5m2", "fp8_e4m3_calib", "int8_per_token_head")
_FP8 = {"fp8_e4m3": torch.float8_e4m3fn, "fp8_e4m3_calib": torch.float8_e4m3fn, "fp8_e5m2": torch.float8_e5m2}


def roundtrip(x: torch.Tensor, fmt: str, scale: float = 1.0) -> torch.Tensor:
    """x [batch, kv_heads, tokens, head_dim] -> its stored-then-read value, in x's dtype and on x's device."""
    if fmt == "int8_per_token_head":
        s = round_scale_up(compute_scale(x, "per_token_head"), torch.bfloat16)
        return dequantize(quantize(x, s), s).to(x.dtype)
    if fmt not in _FP8:
        raise ValueError(f"unknown format {fmt!r}; expected one of {FORMATS}")
    # float8 casts exist on CPU and CUDA only; elsewhere (MPS) do the cast on CPU.
    work = x if x.device.type in ("cpu", "cuda") else x.cpu()
    stored = (work.float() / scale).to(_FP8[fmt])
    return (stored.float() * scale).to(dtype=x.dtype, device=x.device)


class RoundtripKVCache(DynamicCache):
    """Stores round-tripped K/V. With float_prefill, a multi-token chunk attends to its own float K/V."""

    def __init__(self, fmt: str, scales: Optional[Sequence[float]] = None, float_prefill: bool = True):
        super().__init__()
        if fmt not in FORMATS:
            raise ValueError(f"unknown format {fmt!r}; expected one of {FORMATS}")
        if fmt == "fp8_e4m3_calib" and scales is None:
            raise ValueError("fp8_e4m3_calib needs per-layer scales (fp8_probe.py --calib-out)")
        self.fmt = fmt
        self.scales = list(scales) if scales is not None else None
        self.float_prefill = float_prefill

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        # *args: transformers 4.57 passes cache_kwargs as a 4th positional argument, 5.x passes nothing more.
        scale = self.scales[layer_idx] if self.fmt == "fp8_e4m3_calib" else 1.0
        k_all, v_all = super().update(roundtrip(key_states, self.fmt, scale),
                                      roundtrip(value_states, self.fmt, scale), layer_idx, *args, **kwargs)
        n = key_states.shape[-2]
        if self.float_prefill and n > 1:
            k_all = torch.cat([k_all[..., :-n, :], key_states], dim=-2)
            v_all = torch.cat([v_all[..., :-n, :], value_states], dim=-2)
        return k_all, v_all


def load_calib_scales(path: Path) -> List[float]:
    """Per-layer scales from a --quantization-param-path JSON (TP rank 0)."""
    by_layer = json.loads(Path(path).read_text())["kv_cache"]["scaling_factor"]["0"]
    return [float(by_layer[str(i)]) for i in range(len(by_layer))]


def summarize(rows: List[dict]) -> Dict[str, object]:
    def mean(values):
        values = list(values)
        return sum(values) / len(values) if values else None

    needles = [r["needle_correct"] for r in rows if r["needle_correct"] is not None]
    return {
        "prompts": len(rows),
        "top1_per_prompt_mean": mean(r["top1"] for r in rows),
        "top1_short": mean(r["top1"] for r in rows if r["kind"] == "short"),
        "top1_needle": mean(r["top1"] for r in rows if r["kind"] == "needle"),
        "kl_mean_finite": mean(r["kl"] for r in rows if math.isfinite(r["kl"])),
        "nan_logit_prompts": sum(r["nan_logits"] for r in rows),
        "exact_generation_rate": mean(r["exact"] for r in rows),
        "needle_correct": sum(needles),
        "needle_total": len(needles),
    }


def paired(base: Sequence[bool], other: Sequence[bool]) -> Dict[str, object]:
    only_b = sum(b and not o for b, o in zip(base, other))
    only_q = sum(o and not b for b, o in zip(base, other))
    return {"accuracy": sum(other) / len(other), "baseline_accuracy": sum(base) / len(base),
            "only_baseline": only_b, "only_quantized": only_q, "mcnemar_p": mcnemar_exact(only_b, only_q),
            "total": len(other)}


def _strict(obj):
    """Replace non-finite floats with None so the output is strict JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _strict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strict(v) for v in obj]
    return obj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--formats", default=",".join(FORMATS))
    ap.add_argument("--calib", type=Path, default=None, help="quant param JSON for fp8_e4m3_calib")
    ap.add_argument("--needles", type=int, default=10)
    ap.add_argument("--needle-facts", type=int, default=150)
    ap.add_argument("--max-prompts", type=int, default=None, help="limit the short prompts (quick local runs)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--gsm8k", type=int, default=0, help="also run the first N GSM8K questions per format")
    ap.add_argument("--gsm8k-max-new-tokens", type=int, default=512)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    from eval_accuracy import load_items, score
    from int8kv.prompts import SHORT_PROMPTS, make_needle_prompt, needle_correct
    from int8kv.runtime import (encode_chat, environment_info, greedy_generate, load_model, pick_device,
                                teacher_forced_logits)

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    scales = load_calib_scales(args.calib) if args.calib else None
    if "fp8_e4m3_calib" in formats and scales is None:
        ap.error("fp8_e4m3_calib needs --calib")
    device = pick_device(args.device)
    model, tok = load_model(args.model, device, args.dtype)

    prompts = [{"kind": "short", "prompt": p, "answer": None} for p in SHORT_PROMPTS[: args.max_prompts]]
    prompts += [{"kind": "needle", **make_needle_prompt(args.needle_facts, seed)} for seed in range(args.needles)]
    gsm = load_items("gsm8k", args.gsm8k) if args.gsm8k else []

    def gsm_correct(make_cache) -> List[bool]:
        out = []
        for item in gsm:
            ids = encode_chat(tok, item["prompt"], device)
            text = tok.decode(greedy_generate(model, ids, args.gsm8k_max_new_tokens, make_cache()),
                              skip_special_tokens=True)
            out.append(score("gsm8k", text, item["gold"])["correct"])
        return out

    start = time.time()
    baseline = []
    for item in prompts:
        ids = encode_chat(tok, item["prompt"], device)
        tokens = greedy_generate(model, ids, args.max_new_tokens, DynamicCache())
        baseline.append({"ids": ids, "tokens": tokens,
                         "logits": teacher_forced_logits(model, ids, tokens, DynamicCache()),
                         "needle_correct": needle_correct(tok.decode(tokens), item["answer"]) if item["answer"] else None})
    base_gsm = gsm_correct(DynamicCache)
    base_needles = [b["needle_correct"] for b in baseline if b["needle_correct"] is not None]
    print(f"SIM {Path(args.model).name}: baseline on {len(prompts)} prompts, "
          f"needle={sum(base_needles)}/{len(base_needles)}"
          + (f", GSM8K {sum(base_gsm)}/{len(base_gsm)}" if gsm else "") + f" ({time.time() - start:.0f}s)", flush=True)

    result = {"environment": environment_info(device, args.model, args.dtype), "needle_facts": args.needle_facts,
              "max_new_tokens": args.max_new_tokens, "calib": str(args.calib) if args.calib else None,
              "baseline": {"needle_correct": sum(base_needles), "needle_total": len(base_needles)},
              "gsm8k_baseline": {"accuracy": sum(base_gsm) / len(base_gsm), "total": len(base_gsm)} if gsm else None,
              "formats": {}}
    for fmt in formats:
        start = time.time()

        def make_cache(fmt=fmt):
            return RoundtripKVCache(fmt, scales)

        rows = []
        for item, run in zip(prompts, baseline):
            logits = teacher_forced_logits(model, run["ids"], run["tokens"], make_cache())
            tokens = greedy_generate(model, run["ids"], args.max_new_tokens, make_cache())
            finite = bool(torch.isfinite(logits).all())
            rows.append({
                "kind": item["kind"], "top1": top1_agreement(run["logits"], logits),
                "kl": kl_from_logits(run["logits"], logits).item() if finite else float("nan"),
                "nan_logits": not finite, "exact": tokens == run["tokens"],
                "needle_correct": needle_correct(tok.decode(tokens), item["answer"]) if item["answer"] else None,
            })
        entry = {**summarize(rows), "per_prompt": rows}
        line = (f"SIM {fmt:20s} top1/prompt={entry['top1_per_prompt_mean']:.4f} short={entry['top1_short']} "
                f"needle_top1={entry['top1_needle']} kl={entry['kl_mean_finite']} "
                f"nan_prompts={entry['nan_logit_prompts']}/{len(rows)} gen_exact={entry['exact_generation_rate']:.2f} "
                f"needle={entry['needle_correct']}/{entry['needle_total']}")
        if gsm:
            entry["gsm8k"] = paired(base_gsm, gsm_correct(make_cache))
            g = entry["gsm8k"]
            line += (f" gsm8k={g['accuracy']:.4f} vs {g['baseline_accuracy']:.4f} "
                     f"(only_base={g['only_baseline']} only_fmt={g['only_quantized']} p={g['mcnemar_p']:.3g})")
        print(line + f" ({time.time() - start:.0f}s)", flush=True)
        result["formats"][fmt] = entry
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(_strict(result), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
