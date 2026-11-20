"""Why does FP8 KV (scale 1.0) break Qwen2.5-7B while per-token-head INT8 does not? And calibrate FP8 scales.

Runs prompts through the HF model and takes each layer's K and V (post-RoPE, [tokens, kv_heads, head_dim]).

1. Calibration: the per-layer absmax of K and V over the calibration prompts gives one static scale per layer,
   scale = margin * max(|K|, |V|) / 448, written in the JSON format of SGLang's --quantization-param-path.
2. Probe: on a held-out needle prompt, round-trips K and V through fp8_e4m3 and fp8_e5m2 at scale 1.0 (what SGLang
   does without checkpoint scales), fp8_e4m3 with the calibrated scale, and INT8 per-token-head. Softmax ignores
   a component shared by all tokens, so besides the plain relative L2 the probe reports the error relative to
   the *centered* tensor (per-channel mean over tokens removed): the part of K that decides which token
   attention picks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

FORMATS = ("fp8_e4m3", "fp8_e5m2", "int8_per_token_head")
CALIB_FORMAT = "fp8_e4m3_calib"
E4M3_MAX = 448.0
CALIB_NEEDLE_SEED = 10_000  # evaluation needles use seeds 0..99


def roundtrip(x: torch.Tensor, fmt: str, scale: float = 1.0) -> torch.Tensor:
    """x [T, H, D] -> its stored-then-read value in FP32 (NaN where e4m3 overflows, as torch casts)."""
    xf = x.float()
    if fmt == "fp8_e4m3":
        return xf.to(torch.float8_e4m3fn).float()
    if fmt == CALIB_FORMAT:  # SGLang stores k / k_scale and multiplies back on read
        return (xf / scale).to(torch.float8_e4m3fn).float() * scale
    if fmt == "fp8_e5m2":
        return xf.to(torch.float8_e5m2).float()
    if fmt == "int8_per_token_head":
        from int8kv.quant import compute_scale, dequantize, quantize, round_scale_up

        xt = xf.permute(1, 0, 2).unsqueeze(0)  # int8kv layout [batch, heads, tokens, dim]
        s = round_scale_up(compute_scale(xt, "per_token_head"), torch.bfloat16)
        return dequantize(quantize(xt, s), s).squeeze(0).permute(1, 0, 2)
    raise ValueError(fmt)


def layer_stats(x: torch.Tensor, calib_scale: Optional[float] = None) -> Dict[str, object]:
    xf = x.float()
    centered = xf - xf.mean(dim=0, keepdim=True)
    chan_absmax = xf.abs().amax(dim=0).flatten()
    out: Dict[str, object] = {
        "absmax": xf.abs().max().item(),
        "channel_absmax_max_over_median": (chan_absmax.max() / chan_absmax.median().clamp_min(1e-6)).item(),
        "e4m3_overflow_frac": (xf.abs() > E4M3_MAX).float().mean().item(),
        "centered_norm_ratio": (centered.norm() / xf.norm().clamp_min(1e-12)).item(),
    }
    formats = FORMATS if calib_scale is None else (*FORMATS, CALIB_FORMAT)
    for fmt in formats:
        err = roundtrip(xf, fmt, calib_scale if calib_scale is not None else 1.0) - xf
        finite = torch.isfinite(err)
        e = torch.where(finite, err, torch.zeros_like(err))
        out[fmt] = {
            "nan_frac": (~finite).float().mean().item(),
            "rel_l2": (e.norm() / xf.norm().clamp_min(1e-12)).item(),
            "centered_rel_l2": (e.norm() / centered.norm().clamp_min(1e-12)).item(),
        }
    return out


class AbsmaxTracker:
    """Running per-layer absmax of K and V over calibration prompts."""

    def __init__(self, num_layers: int):
        self.k = [0.0] * num_layers
        self.v = [0.0] * num_layers

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self.k[layer] = max(self.k[layer], k.float().abs().max().item())
        self.v[layer] = max(self.v[layer], v.float().abs().max().item())


def calibration_scales(k_absmax: List[float], v_absmax: List[float], margin: float = 2.0) -> List[float]:
    """One scale per layer, shared by K and V (SGLang's Qwen2 loader sets k_scale = v_scale).

    Stored values are x / scale; margin keeps prompts not seen in calibration below e4m3's 448. Scales below 1.0
    are clamped to 1.0: e4m3 is a floating format, so scaling small tensors up gains no precision."""
    return [max(1.0, margin * max(k, v) / E4M3_MAX) for k, v in zip(k_absmax, v_absmax)]


def quant_param_json(scales: List[float], model_type: str) -> dict:
    """The --quantization-param-path document for TP size 1 (sglang.srt.model_loader.weight_utils)."""
    return {"model_type": model_type,
            "kv_cache": {"dtype": "float8_e4m3fn", "scaling_factor": {"0": {str(i): s for i, s in enumerate(scales)}}}}


def _layer_kv(cache, i):
    """K, V of layer i as [tokens, kv_heads, head_dim] on CPU, across transformers cache layouts."""
    if hasattr(cache, "layers"):
        k, v = cache.layers[i].keys, cache.layers[i].values
    else:
        k, v = cache.key_cache[i], cache.value_cache[i]
    return k[0].transpose(0, 1).float().cpu(), v[0].transpose(0, 1).float().cpu()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--needle-facts", type=int, default=150)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--max-calib-prompts", type=int, default=None, help="limit calibration prompts (quick local runs)")
    ap.add_argument("--out", type=Path, required=True, help="per-layer statistics JSON")
    ap.add_argument("--calib-out", type=Path, default=None, help="write SGLang --quantization-param-path JSON here")
    args = ap.parse_args(argv)

    from int8kv.prompts import SHORT_PROMPTS, make_needle_prompt
    from int8kv.runtime import encode_chat, environment_info, load_model, pick_device

    device = pick_device(args.device)
    model, tok = load_model(args.model, device, args.dtype)
    num_layers = model.config.num_hidden_layers

    def kv_cache(text: str):
        ids = encode_chat(tok, text, device)
        with torch.no_grad():
            return ids.shape[1], model(ids, use_cache=True).past_key_values

    calib_prompts = (SHORT_PROMPTS + [make_needle_prompt(args.needle_facts, CALIB_NEEDLE_SEED)["prompt"]])
    calib_prompts = calib_prompts[-args.max_calib_prompts:] if args.max_calib_prompts else calib_prompts
    tracker = AbsmaxTracker(num_layers)
    for text in calib_prompts:
        _, cache = kv_cache(text)
        for i in range(num_layers):
            tracker.update(i, *_layer_kv(cache, i))
        del cache
    scales = calibration_scales(tracker.k, tracker.v, args.margin)

    n_tokens, cache = kv_cache(make_needle_prompt(args.needle_facts, 0)["prompt"])
    layers = []
    for i in range(num_layers):
        k, v = _layer_kv(cache, i)
        layers.append({"layer": i, "calib_scale": scales[i], "calib_k_absmax": tracker.k[i],
                       "calib_v_absmax": tracker.v[i], "k": layer_stats(k, scales[i]), "v": layer_stats(v, scales[i])})

    name = Path(args.model).name
    print(f"PROBE {name}: {n_tokens} probe tokens, {num_layers} layers, {len(calib_prompts)} calibration prompts, "
          f"device {device}", flush=True)
    print(f"PROBE calibration: K absmax max={max(tracker.k):.1f}, V absmax max={max(tracker.v):.1f}, "
          f"scales min={min(scales):.3f} max={max(scales):.3f} (margin {args.margin})", flush=True)
    for kind in ("k", "v"):
        rows = [r[kind] for r in layers]
        worst = max(range(num_layers), key=lambda i: rows[i]["absmax"])
        print(f"PROBE {kind.upper()} absmax max={rows[worst]['absmax']:.1f} (layer {worst}); "
              f"layers with e4m3 overflow: {sum(r['e4m3_overflow_frac'] > 0 for r in rows)}/{num_layers}; "
              f"max channel-absmax max/median={max(r['channel_absmax_max_over_median'] for r in rows):.1f}; "
              f"mean centered/raw norm={sum(r['centered_norm_ratio'] for r in rows) / num_layers:.3f}", flush=True)
        for fmt in (*FORMATS, CALIB_FORMAT):
            def mean(key, fmt=fmt):
                return sum(r[fmt][key] for r in rows) / num_layers

            print(f"PROBE {kind.upper()} {fmt:20s} nan_frac={mean('nan_frac'):.2e} rel_l2={mean('rel_l2'):.4f} "
                  f"centered_rel_l2={mean('centered_rel_l2'):.4f} "
                  f"(worst layer {max(r[fmt]['centered_rel_l2'] for r in rows):.4f})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"environment": environment_info(device, args.model, args.dtype),
                                    "model": args.model, "probe_tokens": n_tokens, "margin": args.margin,
                                    "calibration_prompts": len(calib_prompts), "layers": layers}, indent=2))
    if args.calib_out:
        args.calib_out.parent.mkdir(parents=True, exist_ok=True)
        args.calib_out.write_text(json.dumps(quant_param_json(scales, model.config.model_type), indent=2))
        print(f"PROBE wrote calibrated FP8 scales to {args.calib_out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
