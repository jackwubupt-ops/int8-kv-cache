"""L1 model-level check in SGLang: teacher-forced token logprobs of a KV config vs BF16 KV.

BF16 KV (triton backend) greedily generates a continuation for each prompt. Every
config then scores prompt + continuation, and per-position top-1 tokens and the
logprob of the reference token are compared with BF16 scoring itself.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TokenLogprobs = List[Tuple[float, int]]  # per position: (logprob of the reference token, top-1 token id)


def compare_token_logprobs(ref: Sequence[TokenLogprobs], other: Sequence[TokenLogprobs]) -> Dict[str, float]:
    if len(ref) != len(other) or any(len(a) != len(b) for a, b in zip(ref, other)):
        raise ValueError("ref and other must have the same prompts and length per prompt")
    pairs = [(a, b) for seq_a, seq_b in zip(ref, other) for a, b in zip(seq_a, seq_b)]
    n = len(pairs)
    return {
        "top1_agreement": sum(a[1] == b[1] for a, b in pairs) / n,
        "mean_abs_logprob_diff": sum(abs(a[0] - b[0]) for a, b in pairs) / n,
        "n_tokens": n,
    }


def token_logprobs_from_meta(meta: dict) -> TokenLogprobs:
    return [(chosen[0], top[0][1]) for chosen, top in zip(meta["input_token_logprobs"], meta["input_top_logprobs"])]


def verdict(summary: Dict[str, float], min_top1: float) -> str:
    return "PASS" if summary["top1_agreement"] >= min_top1 else "FAIL"


def build_prompts(tokenizer, needles: int, needle_facts: int) -> List[Tuple[str, List[int]]]:
    from int8kv.prompts import SHORT_PROMPTS, make_needle_prompt

    texts = [("short", p) for p in SHORT_PROMPTS]
    texts += [("needle", make_needle_prompt(needle_facts, seed)["prompt"]) for seed in range(needles)]
    def encode(text: str) -> List[int]:
        chat = tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
        return tokenizer(chat, add_special_tokens=False).input_ids

    return [(kind, encode(t)) for kind, t in texts]


def make_engine(model: str, kv_dtype: str, impl: str, cuda_graph: bool):
    import sglang

    os.environ["SGLANG_INT8_KV_IMPL"] = impl
    return sglang.Engine(
        model_path=model, attention_backend="triton", kv_cache_dtype=kv_dtype,
        disable_cuda_graph=not cuda_graph, mem_fraction_static=0.7, log_level="error",
    )


def generate_reference(engine, prompts, max_new_tokens: int) -> List[List[int]]:
    outs = engine.generate(
        input_ids=[ids for _, ids in prompts],
        sampling_params={"max_new_tokens": max_new_tokens, "temperature": 0},
        return_logprob=True,
    )
    return [[tok for _, tok, _ in o["meta_info"]["output_token_logprobs"]] for o in outs]


def teacher_forced(engine, prompts, continuations) -> List[TokenLogprobs]:
    results = []
    for (_, ids), cont in zip(prompts, continuations):
        out = engine.generate(
            input_ids=ids + cont,
            sampling_params={"max_new_tokens": 1, "temperature": 0},
            return_logprob=True, logprob_start_len=len(ids), top_logprobs_num=1,
        )
        results.append(token_logprobs_from_meta(out["meta_info"]))
    return results


def summarize(prompts, ref, other) -> Dict[str, Dict[str, float]]:
    out = {"all": compare_token_logprobs(ref, other)}
    for kind in ("short", "needle"):
        idx = [i for i, (k, _) in enumerate(prompts) if k == kind]
        if idx:
            out[kind] = compare_token_logprobs([ref[i] for i in idx], [other[i] for i in idx])
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--configs", default="int8_per_token_head:plain:on,int8_per_token_head:plain:off",
                        help="comma list of kv_dtype:impl:cuda_graph(on|off)")
    parser.add_argument("--needles", type=int, default=10)
    parser.add_argument("--needle-facts", type=int, default=150)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--min-top1", type=float, default=0.99)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    prompts = build_prompts(AutoTokenizer.from_pretrained(args.model), args.needles, args.needle_facts)
    engine = make_engine(args.model, "auto", "plain", cuda_graph=True)
    continuations = generate_reference(engine, prompts, args.max_new_tokens)
    ref = teacher_forced(engine, prompts, continuations)
    engine.shutdown()

    results, failed = {}, False
    for item in args.configs.split(","):
        kv_dtype, impl, graph = item.split(":")
        engine = make_engine(args.model, kv_dtype, impl, cuda_graph=graph == "on")
        summary = summarize(prompts, ref, teacher_forced(engine, prompts, continuations))
        engine.shutdown()
        summary["verdict"] = verdict(summary["all"], args.min_top1)
        failed |= summary["verdict"] == "FAIL"
        results[item] = summary
        print(f"L1 {item}: {summary['verdict']} top1={summary['all']['top1_agreement']:.4f} "
              f"|dlogp|={summary['all']['mean_abs_logprob_diff']:.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": args.model, "min_top1": args.min_top1, "results": results}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
