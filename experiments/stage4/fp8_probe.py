"""Why does FP8 KV (scale 1.0) break Qwen2.5-7B while per-token-head INT8 does not?

Runs one prompt through the HF model, takes each layer's K and V (post-RoPE, [tokens, kv_heads, head_dim]) and
round-trips them through fp8_e4m3, fp8_e5m2 (as SGLang stores them without checkpoint scales) and INT8
per-token-head. Softmax ignores a component shared by all tokens, so besides the plain relative L2 the probe
reports the error relative to the *centered* tensor (per-channel mean over tokens removed): the part of K that
decides which token attention picks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

FORMATS = ("fp8_e4m3", "fp8_e5m2", "int8_per_token_head")
E4M3_MAX = 448.0


def roundtrip(x: torch.Tensor, fmt: str) -> torch.Tensor:
    """x [T, H, D] -> its stored-then-read value in FP32 (NaN where e4m3 overflows, as torch casts)."""
    xf = x.float()
    if fmt == "fp8_e4m3":
        return xf.to(torch.float8_e4m3fn).float()
    if fmt == "fp8_e5m2":
        return xf.to(torch.float8_e5m2).float()
    if fmt == "int8_per_token_head":
        from int8kv.quant import compute_scale, dequantize, quantize, round_scale_up

        xt = xf.permute(1, 0, 2).unsqueeze(0)  # int8kv layout [batch, heads, tokens, dim]
        scale = round_scale_up(compute_scale(xt, "per_token_head"), torch.bfloat16)
        return dequantize(quantize(xt, scale), scale).squeeze(0).permute(1, 0, 2)
    raise ValueError(fmt)


def layer_stats(x: torch.Tensor) -> Dict[str, object]:
    xf = x.float()
    centered = xf - xf.mean(dim=0, keepdim=True)
    chan_absmax = xf.abs().amax(dim=0).flatten()
    out: Dict[str, object] = {
        "absmax": xf.abs().max().item(),
        "channel_absmax_max_over_median": (chan_absmax.max() / chan_absmax.median().clamp_min(1e-6)).item(),
        "e4m3_overflow_frac": (xf.abs() > E4M3_MAX).float().mean().item(),
        "centered_norm_ratio": (centered.norm() / xf.norm().clamp_min(1e-12)).item(),
    }
    for fmt in FORMATS:
        err = roundtrip(xf, fmt) - xf
        finite = torch.isfinite(err)
        e = torch.where(finite, err, torch.zeros_like(err))
        out[fmt] = {
            "nan_frac": (~finite).float().mean().item(),
            "rel_l2": (e.norm() / xf.norm().clamp_min(1e-12)).item(),
            "centered_rel_l2": (e.norm() / centered.norm().clamp_min(1e-12)).item(),
        }
    return out


def _layer_kv(cache, i):
    if hasattr(cache, "layers"):
        return cache.layers[i].keys, cache.layers[i].values
    return cache.key_cache[i], cache.value_cache[i]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--needle-facts", type=int, default=150)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from int8kv.prompts import make_needle_prompt

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    text = make_needle_prompt(args.needle_facts, 0)["prompt"]
    chat = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
    ids = tok(chat, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    with torch.no_grad():
        cache = model(ids, use_cache=True).past_key_values
    layers = []
    for i in range(model.config.num_hidden_layers):
        k, v = _layer_kv(cache, i)  # [1, H, T, D]
        k, v = k[0].transpose(0, 1).cpu(), v[0].transpose(0, 1).cpu()
        layers.append({"layer": i, "k": layer_stats(k), "v": layer_stats(v)})
    print(f"PROBE {Path(args.model).name}: {ids.shape[1]} tokens, {len(layers)} layers", flush=True)
    for kind in ("k", "v"):
        rows = [r[kind] for r in layers]
        worst = max(rows, key=lambda r: r["absmax"])
        print(f"PROBE {kind.upper()} absmax max={max(r['absmax'] for r in rows):.1f} "
              f"(layer {rows.index(worst)}); layers with e4m3 overflow: "
              f"{sum(r['e4m3_overflow_frac'] > 0 for r in rows)}/{len(rows)}; "
              f"max channel-absmax max/median={max(r['channel_absmax_max_over_median'] for r in rows):.1f}; "
              f"mean centered/raw norm={sum(r['centered_norm_ratio'] for r in rows) / len(rows):.3f}", flush=True)
        for fmt in FORMATS:
            mean = lambda key: sum(r[fmt][key] for r in rows) / len(rows)  # noqa: E731
            print(f"PROBE {kind.upper()} {fmt:20s} nan_frac={mean('nan_frac'):.2e} rel_l2={mean('rel_l2'):.4f} "
                  f"centered_rel_l2={mean('centered_rel_l2'):.4f} "
                  f"(worst layer {max(r[fmt]['centered_rel_l2'] for r in rows):.4f})", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": args.model, "tokens": ids.shape[1], "layers": layers}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
