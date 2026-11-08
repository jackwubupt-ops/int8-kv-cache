"""Stage 3: compare model behaviour with an INT8 KV cache against the float baseline."""
from __future__ import annotations

import argparse
from pathlib import Path

from transformers import DynamicCache

from int8kv.cache import QuantizedKVCache
from int8kv.metrics import first_divergence, kl_from_logits, top1_agreement
from int8kv.prompts import SHORT_PROMPTS, make_needle_prompt, needle_correct
from int8kv.qwen2_patch import patch_qwen2
from int8kv.runtime import (
    encode_chat, environment_info, greedy_generate, load_model, parse_configs, pick_device, plain_cache_bytes,
    save_json, teacher_forced_logits,
)


def _mean(values):
    return sum(values) / len(values) if values else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--needle-facts", type=int, default=150)
    parser.add_argument("--needles", type=int, default=20)
    parser.add_argument("--configs", default="per_token_head:dequant,per_token_head:integer,per_head:dequant,per_tensor:dequant")
    parser.add_argument("--out", type=Path, default=Path("results/stage3"))
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    model, tokenizer = load_model(args.model, device, args.dtype)
    patch_qwen2(model)
    needles = [make_needle_prompt(args.needle_facts, seed) for seed in range(args.needles)]
    prompts = [{"prompt": p, "answer": None} for p in SHORT_PROMPTS[: args.max_prompts]] + needles

    baseline = []
    for item in prompts:
        ids = encode_chat(tokenizer, item["prompt"], device)
        cache = DynamicCache()
        tokens = greedy_generate(model, ids, args.max_new_tokens, cache)
        logits = teacher_forced_logits(model, ids, tokens, DynamicCache())
        baseline.append({"ids": ids, "tokens": tokens, "logits": logits, "item": item, "bytes": plain_cache_bytes(cache)})
    print(f"baseline done on {len(prompts)} prompts", flush=True)

    def evaluate(make_cache, bytes_of):
        kl, top1, exact, needle_hits = [], [], [], []
        divergence, ratios = [], []
        for run in baseline:
            logits = teacher_forced_logits(model, run["ids"], run["tokens"], make_cache())
            kl.append(kl_from_logits(run["logits"], logits).item())
            top1.append(top1_agreement(run["logits"], logits))
            cache = make_cache()
            tokens = greedy_generate(model, run["ids"], args.max_new_tokens, cache)
            first = first_divergence(run["tokens"], tokens)
            exact.append(first is None)
            if first is not None:
                divergence.append(first)
            ratios.append(bytes_of(cache) / run["bytes"])
            if run["item"]["answer"]:
                needle_hits.append(needle_correct(tokenizer.decode(tokens), run["item"]["answer"]))
        return {
            "logits_kl": _mean(kl),
            "top1_agreement": _mean(top1),
            "exact_generation_rate": _mean(exact),
            "mean_first_divergence": _mean(divergence),
            "kv_memory_ratio_vs_baseline": _mean(ratios),
            "needle_accuracy": _mean(needle_hits),
        }

    results = {"baseline_repeat_noise_floor": evaluate(DynamicCache, plain_cache_bytes)}
    print("noise floor", results["baseline_repeat_noise_floor"], flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    env = environment_info(device, args.model, args.dtype)
    for granularity, mode in parse_configs(args.configs):
        key = f"{granularity}:{mode}"
        results[key] = evaluate(
            lambda: QuantizedKVCache(granularity=granularity, mode=mode),
            lambda cache: cache.memory_bytes()["total_bytes"],
        )
        print(key, results[key], flush=True)
        save_json(args.out / "model_metrics.json", {"environment": env, "results": results})
    return {"environment": env, "results": results}


if __name__ == "__main__":
    main()
