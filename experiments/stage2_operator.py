"""Stage 2: compare INT8 and float attention on real Q/K/V, layer by layer."""
from __future__ import annotations

import argparse
from pathlib import Path

from int8kv.cache import QuantizedKVCache
from int8kv.prompts import SHORT_PROMPTS, make_needle_prompt
from int8kv.qwen2_patch import OperatorRecorder, patch_qwen2
from int8kv.runtime import encode_chat, environment_info, greedy_generate, load_model, parse_configs, pick_device, save_json

REFERENCE_REL_L2 = 0.10
DEFAULT_CONFIGS = (
    "per_tensor:dequant,per_head:dequant,per_token:dequant,per_token_head:dequant,"
    "per_tensor:integer,per_head:integer,per_token:integer,per_token_head:integer"
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--needle-facts", type=int, default=100)
    parser.add_argument("--max-prompts", type=int, default=6)
    parser.add_argument("--configs", default=DEFAULT_CONFIGS)
    parser.add_argument("--out", type=Path, default=Path("results/stage2"))
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    model, tokenizer = load_model(args.model, device, args.dtype)
    recorder = OperatorRecorder()
    patch_qwen2(model, recorder)
    prompts = SHORT_PROMPTS[: args.max_prompts] + [make_needle_prompt(args.needle_facts, 0)["prompt"]]

    results = {}
    for granularity, mode in parse_configs(args.configs):
        recorder.rows.clear()
        for text in prompts:
            ids = encode_chat(tokenizer, text, device)
            cache = QuantizedKVCache(granularity=granularity, mode=mode, keep_float=True)
            greedy_generate(model, ids, args.max_new_tokens, cache)
        summary = recorder.summary()
        for phase in summary.values():
            phase["within_reference"] = phase["output_rel_l2"] <= REFERENCE_REL_L2
        results[f"{granularity}:{mode}"] = summary
        print(granularity, mode, {p: round(s["output_rel_l2"], 5) for p, s in summary.items()}, flush=True)

    output = {
        "environment": environment_info(device, args.model, args.dtype),
        "reference_output_rel_l2": REFERENCE_REL_L2,
        "results": results,
    }
    save_json(args.out / "operator_metrics.json", output)
    return output


if __name__ == "__main__":
    main()
