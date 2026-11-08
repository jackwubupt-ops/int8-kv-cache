"""Stage 3: GSM8K accuracy for the float baseline and INT8 KV caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import DynamicCache

from int8kv.cache import QuantizedKVCache
from int8kv.prompts import extract_number
from int8kv.qwen2_patch import patch_qwen2
from int8kv.runtime import encode_chat, environment_info, greedy_generate, load_model, parse_configs, pick_device, save_json

INSTRUCTION = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def load_samples(data_file, count):
    if data_file:
        lines = Path(data_file).read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
    else:
        from datasets import load_dataset

        rows = load_dataset("openai/gsm8k", "main", split="test")
    return [rows[i] for i in range(min(count, len(rows)))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--data-file", default=None, help="Optional JSONL with question and answer fields")
    parser.add_argument("--configs", default="per_token_head:dequant,per_token_head:integer")
    parser.add_argument("--out", type=Path, default=Path("results/stage3"))
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    model, tokenizer = load_model(args.model, device, args.dtype)
    patch_qwen2(model)
    samples = load_samples(args.data_file, args.samples)
    golds = [extract_number(s["answer"]) for s in samples]

    def run(make_cache, label):
        predictions = []
        for index, sample in enumerate(samples):
            ids = encode_chat(tokenizer, sample["question"] + INSTRUCTION, device)
            tokens = greedy_generate(model, ids, args.max_new_tokens, make_cache())
            predictions.append(extract_number(tokenizer.decode(tokens, skip_special_tokens=True)))
            if (index + 1) % 20 == 0:
                print(f"{label}: {index + 1}/{len(samples)}", flush=True)
        return predictions

    env = environment_info(device, args.model, args.dtype)
    base = run(DynamicCache, "baseline")
    results = {"baseline": {"accuracy": sum(p == g for p, g in zip(base, golds)) / len(golds)}}
    for granularity, mode in parse_configs(args.configs):
        key = f"{granularity}:{mode}"
        preds = run(lambda: QuantizedKVCache(granularity=granularity, mode=mode), key)
        results[key] = {
            "accuracy": sum(p == g for p, g in zip(preds, golds)) / len(golds),
            "answers_changed_vs_baseline": sum(p != b for p, b in zip(preds, base)),
        }
        print(key, results[key], flush=True)
        save_json(args.out / "gsm8k.json", {"environment": env, "samples": len(golds), "results": results})
    output = {"environment": env, "samples": len(golds), "results": results}
    save_json(args.out / "gsm8k.json", output)
    return output


if __name__ == "__main__":
    main()
