"""Stage 5b: operator error of implementation A (dequant) vs B (int8 compute) on real Q/K/V.

For each config the decode-step attention output is compared with float attention on the
unquantized K/V, layer by layer. A's error is the quantization error alone; B - A is the
extra error from computing in INT8 (Q folded/quantized, P quantized).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from int8kv.calibration import build_cache
from int8kv.prompts import SHORT_PROMPTS, make_needle_prompt
from int8kv.qwen2_patch import OperatorRecorder, patch_qwen2
from int8kv.runtime import (
    encode_chat, environment_info, greedy_generate, load_model, parse_cache_configs, pick_device, save_json,
)

DEFAULT_CONFIGS = ",".join(
    f"{g}:{m}:{s}"
    for g, sources in (("per_token_head", ("write",)), ("per_head", ("dynamic", "static")),
                       ("per_channel", ("dynamic", "static")))
    for s in sources
    for m in ("dequant", "int8")
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--needle-facts", type=int, default=100)
    parser.add_argument("--max-prompts", type=int, default=6)
    parser.add_argument("--configs", default=DEFAULT_CONFIGS)
    parser.add_argument("--scales", type=Path, default=Path("results/stage5/scales.pt"))
    parser.add_argument("--quantile-scales", action="store_true", help="use the percentile instead of absmax")
    parser.add_argument("--out", type=Path, default=Path("results/stage5"))
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    model, tokenizer = load_model(args.model, device, args.dtype)
    recorder = OperatorRecorder()
    patch_qwen2(model, recorder)
    prompts = SHORT_PROMPTS[: args.max_prompts] + [make_needle_prompt(args.needle_facts, 0)["prompt"]]

    results = {}
    for cfg in parse_cache_configs(args.configs):
        recorder.rows.clear()
        for text in prompts:
            cache = build_cache(cfg, args.scales, args.quantile_scales, keep_float=True)
            greedy_generate(model, encode_chat(tokenizer, text, device), args.max_new_tokens, cache)
        decode = recorder.summary()["decode"]
        results[cfg["label"]] = decode
        print(f"{cfg['label']:32s} output rel L2 {decode['actual_output_rel_l2'] * 100:.2f}%", flush=True)

    # B over A: the extra error that comes from computing in INT8 rather than from quantizing the cache.
    split = {}
    for label, row in results.items():
        gran, mode, source = label.split(":")
        a = results.get(f"{gran}:dequant:{source}")
        if mode == "int8" and a:
            split[f"{gran}:{source}"] = {
                "A_dequant": a["actual_output_rel_l2"],
                "B_int8": row["actual_output_rel_l2"],
                "B_minus_A": row["actual_output_rel_l2"] - a["actual_output_rel_l2"],
                "B_over_A_increase": row["actual_output_rel_l2"] / a["actual_output_rel_l2"] - 1,
            }
    output = {
        "environment": environment_info(device, args.model, args.dtype),
        "scales": str(args.scales), "quantile_scales": args.quantile_scales,
        "results": results, "a_vs_b": split,
    }
    suffix = "_q" if args.quantile_scales else ""
    save_json(args.out / f"operator_ab{suffix}.json", output)
    for key, row in split.items():
        print(f"{key:28s} A {row['A_dequant'] * 100:.2f}%  B {row['B_int8'] * 100:.2f}%  "
              f"B-A {row['B_minus_A'] * 100:+.2f} pts", flush=True)
    return output


if __name__ == "__main__":
    main()
