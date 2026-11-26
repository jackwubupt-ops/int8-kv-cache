"""Stage 5a: calibrate static per-channel K/V scales and measure how uneven the channels are.

Calibration prompts default to GSM8K *train* questions so that no evaluation set (GSM8K test,
AIME, HumanEval, needle) is seen during calibration. Use --data-file to calibrate on AIME-style
problems instead (JSONL with a "question" or "problem" field).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import DynamicCache

from int8kv.calibration import ChannelStats
from int8kv.prompts import GSM8K_INSTRUCTION
from int8kv.tasks import read_jsonl
from int8kv.runtime import encode_chat, environment_info, greedy_generate, load_model, pick_device, save_json


def calibration_prompts(data_file, count):
    if data_file:
        rows = read_jsonl(data_file)
    else:
        from datasets import load_dataset

        rows = load_dataset("openai/gsm8k", "main", split="train")
    texts = []
    for i in range(min(count, len(rows))):
        row = rows[i]
        texts.append((row.get("question") or row.get("problem")) + GSM8K_INSTRUCTION)
    return texts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--quantile", type=float, default=0.999)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--out", type=Path, default=Path("results/stage5"))
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    model, tokenizer = load_model(args.model, device, args.dtype)
    stats = ChannelStats()
    prompts = calibration_prompts(args.data_file, args.samples)
    for index, text in enumerate(prompts):
        cache = DynamicCache()
        greedy_generate(model, encode_chat(tokenizer, text, device), args.max_new_tokens, cache)
        for layer_idx, layer in enumerate(cache.layers):
            stats.add(layer_idx, layer.keys, layer.values)
        if (index + 1) % 8 == 0:
            print(f"calibration {index + 1}/{len(prompts)}", flush=True)

    stats.save(args.out / "scales.pt", args.quantile)
    spread = stats.channel_spread()
    summary = {
        "environment": environment_info(device, args.model, args.dtype),
        "samples": len(prompts),
        "tokens_per_layer": stats.tokens[0],
        "quantile": args.quantile,
        "channel_spread": spread,
        "absmax_vs_quantile": {
            kind: [
                (stats.absmax(i, kind) / stats.quantile(i, kind, args.quantile).clamp_min(1e-12)).max().item()
                for i in stats.layers()
            ]
            for kind in ("k", "v")
        },
    }
    save_json(args.out / "calibration.json", summary)
    for kind in ("k", "v"):
        s = spread[kind]
        print(f"{kind.upper()}: within-head max/median channel p50={s['within_head_max_over_median_p50']:.1f} "
              f"max={s['within_head_max_over_median_max']:.1f}; between heads p50={s['between_heads_max_over_min_p50']:.1f}")
    return summary


if __name__ == "__main__":
    main()
