"""L1 model-level check in SGLang: forced-decode token logprobs of each KV config vs BF16 KV.

Same method as stage 3 (int8kv.runtime.teacher_forced_logits): BF16 KV greedily writes
a continuation for each prompt, then every config is fed that continuation one token per
request. logprob_start_len = len - 1 makes SGLang take every earlier token from the KV
cache, so each prediction reads the whole prompt and continuation back from the
(quantized) cache, exactly as decoding does. The prompt itself is prefilled in float.

Controls: a fresh BF16 engine ("auto") gives the noise floor; it must agree with the
reference almost perfectly or the comparison is not trustworthy (verdict INVALID).
FP8 is reported for scale. Greedy generations exercise the real decode path and
CUDA graphs; INT8 with graphs on and off must generate the same tokens.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TokenLogprobs = List[Tuple[float, int]]  # per position: (logprob of the reference token, top-1 token id)
NOISE_FLOOR_MIN = 0.999


def compare_token_logprobs(ref: Sequence[TokenLogprobs], other: Sequence[TokenLogprobs]) -> Dict[str, float]:
    if len(ref) != len(other) or any(len(a) != len(b) for a, b in zip(ref, other)):
        raise ValueError("ref and other must have the same prompts and length per prompt")
    pairs = [(a, b) for seq_a, seq_b in zip(ref, other) for a, b in zip(seq_a, seq_b)]
    per_prompt = [sum(a[1] == b[1] for a, b in zip(sa, sb)) / len(sa) for sa, sb in zip(ref, other) if sa]
    n = len(pairs)
    return {
        "top1_agreement": sum(a[1] == b[1] for a, b in pairs) / n,
        "top1_per_prompt_mean": sum(per_prompt) / len(per_prompt),
        "mean_abs_logprob_diff": sum(abs(a[0] - b[0]) for a, b in pairs) / n,
        "n_tokens": n,
    }


def step_score(meta: dict, ref_token: int) -> Tuple[float, int]:
    (ref_lp, tok, _), = meta["output_token_ids_logprobs"][0]
    if tok != ref_token:
        raise ValueError(f"SGLang scored token {tok}, expected reference token {ref_token}")
    return ref_lp, meta["output_top_logprobs"][0][0][1]


def forced_decode_scores(engine, prompts, continuations) -> List[TokenLogprobs]:
    results: List[TokenLogprobs] = [[] for _ in prompts]
    for step in range(max(len(c) for c in continuations)):
        live = [i for i, c in enumerate(continuations) if step < len(c)]
        inputs = [prompts[i][1] + continuations[i][:step] for i in live]
        outs = engine.generate(
            input_ids=inputs,
            sampling_params={"max_new_tokens": 1, "temperature": 0},
            return_logprob=True, logprob_start_len=[len(x) - 1 for x in inputs], top_logprobs_num=1,
            token_ids_logprob=[[continuations[i][step]] for i in live],
        )
        for i, out in zip(live, outs):
            results[i].append(step_score(out["meta_info"], continuations[i][step]))
    return results


def generation_agreement(ref: Sequence[List[int]], other: Sequence[List[int]]) -> Dict[str, Optional[float]]:
    divergence = [next((j for j, (a, b) in enumerate(zip(r, o)) if a != b), None) for r, o in zip(ref, other)]
    divergence = [d if d is not None or len(r) == len(o) else min(len(r), len(o))
                  for d, r, o in zip(divergence, ref, other)]
    diverged = [d for d in divergence if d is not None]
    return {
        "exact_rate": sum(d is None for d in divergence) / len(divergence),
        "mean_first_divergence": sum(diverged) / len(diverged) if diverged else None,
    }


def verdict(top1: float, min_top1: float, noise_floor: float) -> str:
    if noise_floor < NOISE_FLOOR_MIN:
        return "INVALID"
    return "PASS" if top1 >= min_top1 else "FAIL"


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


def greedy(engine, prompts, max_new_tokens: int) -> List[List[int]]:
    engine.flush_cache()  # prompts must be prefilled fresh, not read back from the cache
    outs = engine.generate(
        input_ids=[ids for _, ids in prompts],
        sampling_params={"max_new_tokens": max_new_tokens, "temperature": 0},
        return_logprob=True,
    )
    return [[tok for _, tok, _ in o["meta_info"]["output_token_logprobs"]] for o in outs]


def run_config(model, kv_dtype, impl, cuda_graph, prompts, continuations=None, max_new_tokens=64):
    engine = make_engine(model, kv_dtype, impl, cuda_graph)
    try:
        generated = greedy(engine, prompts, max_new_tokens)
        engine.flush_cache()
        scores = forced_decode_scores(engine, prompts, continuations or generated)
    finally:
        engine.shutdown()
    return generated, scores


def summarize(prompts, ref, other) -> Dict[str, Dict[str, float]]:
    out = {"all": compare_token_logprobs(ref, other)}
    for kind in ("short", "needle"):
        idx = [i for i, (k, _) in enumerate(prompts) if k == kind]
        if idx:
            out[kind] = compare_token_logprobs([ref[i] for i in idx], [other[i] for i in idx])
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--configs",
                        default="auto:plain:on,fp8_e4m3:plain:on,int8_per_token_head:plain:on,int8_per_token_head:plain:off",
                        help="comma list of kv_dtype:impl:cuda_graph(on|off); must include auto (noise floor)")
    parser.add_argument("--needles", type=int, default=10)
    parser.add_argument("--needle-facts", type=int, default=150)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--min-top1", type=float, default=0.99, help="per-prompt mean top-1, as in stage 3 (99.4%%)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    configs = [tuple(item.split(":")) for item in args.configs.split(",")]
    if not any(kv == "auto" for kv, _, _ in configs):
        parser.error("--configs must include an auto (BF16) entry: it is the noise floor")

    from transformers import AutoTokenizer

    prompts = build_prompts(AutoTokenizer.from_pretrained(args.model), args.needles, args.needle_facts)
    continuations, ref = run_config(args.model, "auto", "plain", True, prompts, max_new_tokens=args.max_new_tokens)
    print(f"reference: {len(prompts)} prompts, {sum(map(len, continuations))} continuation tokens", flush=True)

    results, generations = {}, {}
    for kv_dtype, impl, graph in configs:
        item = f"{kv_dtype}:{impl}:{graph}"
        generated, scores = run_config(args.model, kv_dtype, impl, graph == "on", prompts, continuations,
                                       args.max_new_tokens)
        generations[item] = generated
        summary = summarize(prompts, ref, scores)
        summary["generation_vs_bf16"] = generation_agreement(continuations, generated)
        results[item] = summary

    noise_floor = min(r["all"]["top1_per_prompt_mean"] for item, r in results.items() if item.startswith("auto:"))
    failed = False
    for item, summary in results.items():
        top1 = summary["all"]["top1_per_prompt_mean"]
        summary["verdict"] = (verdict(top1, args.min_top1, noise_floor) if item.startswith("int8_")
                              else ("NOISE-FLOOR" if item.startswith("auto:") else "REFERENCE-ONLY"))
        failed |= summary["verdict"] in ("FAIL", "INVALID")
        gen = summary["generation_vs_bf16"]
        print(f"L1 {item}: {summary['verdict']} top1/prompt={top1:.4f} top1/token={summary['all']['top1_agreement']:.4f} "
              f"|dlogp|={summary['all']['mean_abs_logprob_diff']:.4f} "
              f"short={summary['short']['top1_per_prompt_mean']:.4f} needle={summary['needle']['top1_per_prompt_mean']:.4f} "
              f"gen_exact={gen['exact_rate']:.2f} first_div={gen['mean_first_divergence']}", flush=True)

    on, off = "int8_per_token_head:plain:on", "int8_per_token_head:plain:off"
    graph_check = None
    if on in generations and off in generations:
        graph_check = generation_agreement(generations[off], generations[on])
        same = graph_check["exact_rate"] == 1.0
        failed |= not same
        print(f"L1 cuda-graph on vs off (int8 generations): {'PASS' if same else 'FAIL'} exact={graph_check['exact_rate']:.2f}",
              flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": args.model, "min_top1": args.min_top1, "noise_floor": noise_floor,
                                    "cuda_graph_on_vs_off": graph_check, "results": results}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
