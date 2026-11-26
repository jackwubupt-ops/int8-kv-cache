"""Stage 5c: benchmark accuracy of native BF16 vs INT8 KV configs (GSM8K, AIME, HumanEval).

Configs are granularity:mode:scale_source (see QuantizedKVCache); "baseline" is the plain BF16 cache.
Prefill attention uses the float K/V of the chunk itself, as in vLLM/SGLang; decode reads INT8.
Results are saved after every config, and a rerun skips configs already in the output file.

Note: HumanEval executes model-generated code in a subprocess (python -I, temp dir, timeout).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import DynamicCache

from int8kv.calibration import build_cache
from int8kv.metrics import mcnemar_exact
from int8kv.qwen2_patch import patch_qwen2
from int8kv.runtime import encode_chat, environment_info, greedy_generate, load_model, parse_cache_configs, pick_device, save_json
from int8kv.tasks import DEFAULT_MAX_NEW_TOKENS, make_task

DEFAULT_CONFIGS = (
    "per_head:dequant:dynamic,per_channel:dequant:dynamic,per_channel:int8:dynamic,"
    "per_channel:int8:static,per_token_head:dequant:write"
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["gsm8k", "aime", "humaneval"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--samples", type=int, default=None, help="first N problems (default: all)")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--data-file", default=None, help="local JSONL instead of the Hugging Face dataset")
    parser.add_argument("--configs", default=DEFAULT_CONFIGS)
    parser.add_argument("--scales", type=Path, default=Path("results/stage5/scales.pt"))
    parser.add_argument("--quantile-scales", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/stage5"))
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    model, tokenizer = load_model(args.model, device, args.dtype)
    patch_qwen2(model)
    task = make_task(args.task, args.data_file, args.samples)
    max_new = args.max_new_tokens or DEFAULT_MAX_NEW_TOKENS[args.task]
    path = args.out / f"{args.task}.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    if saved and saved.get("samples") != len(task.samples):
        raise SystemExit(f"{path} has {saved.get('samples')} samples, this run has {len(task.samples)}; use another --out")
    texts = saved.get("texts", {})
    results = saved.get("results", {})

    def run(make, label):
        outs, ok = [], []
        for i, sample in enumerate(task.samples):
            tokens = greedy_generate(model, encode_chat(tokenizer, task.prompt(sample), device), max_new, make())
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            outs.append(text)
            ok.append(bool(task.score(sample, text)))
            if (i + 1) % 10 == 0:
                print(f"{label}: {i + 1}/{len(task.samples)}  acc so far {sum(ok) / len(ok):.3f}", flush=True)
        return outs, ok

    def output():
        return {"environment": environment_info(device, args.model, args.dtype), "task": args.task,
                "samples": len(task.samples), "max_new_tokens": max_new, "results": results, "texts": texts}

    labels = ["baseline"] + [c["label"] for c in parse_cache_configs(args.configs)]
    configs = {c["label"]: c for c in parse_cache_configs(args.configs)}
    for label in labels:
        if label in results:
            print(f"{label}: already done, accuracy {results[label]['accuracy']:.4f}")
            continue
        make = DynamicCache if label == "baseline" else (lambda c=configs[label]: build_cache(c, args.scales, args.quantile_scales))
        outs, ok = run(make, label)
        texts[label] = outs
        row = {"accuracy": sum(ok) / len(ok), "correct": ok}
        if label != "baseline":
            base = results["baseline"]["correct"]
            only_base = sum(b and not q for b, q in zip(base, ok))
            only_quant = sum(q and not b for b, q in zip(base, ok))
            row.update(only_baseline=only_base, only_quantized=only_quant, mcnemar_p=mcnemar_exact(only_base, only_quant))
        results[label] = row
        save_json(path, output())
        print(f"{label}: accuracy {row['accuracy']:.4f}" + (f"  McNemar p={row['mcnemar_p']:.3f}" if "mcnemar_p" in row else ""), flush=True)

    print(f"\n{args.task} ({len(task.samples)} problems, {args.model})")
    for label in labels:
        r = results[label]
        extra = f"  -{r['only_baseline']} / +{r['only_quantized']}  p={r['mcnemar_p']:.3f}" if "mcnemar_p" in r else ""
        print(f"  {label:32s} {r['accuracy'] * 100:6.2f}%{extra}")
    return output()


if __name__ == "__main__":
    main()
