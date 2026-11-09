"""L1 model-level check in SGLang: forced-decode token logprobs of each KV config vs BF16 KV.

Same method as stage 3 (int8kv.runtime.teacher_forced_logits): BF16 KV greedily writes
a continuation for each prompt, then every config is fed that continuation one token per
request. logprob_start_len = len - 1 makes SGLang take every earlier token from the KV
cache, so each prediction reads the whole prompt and continuation back from the
(quantized) cache, exactly as decoding does. The prompt itself is prefilled in float:
chunked prefill is off by default, because with SGLang's 2048-token chunks the second
chunk of a ~3.3k-token needle prompt would read the first chunk back from INT8, which
stage 3 never did (its "int8 prefill" run shows that costs needle top-1 1.0 -> 0.86).

Pass criterion: INT8 must reproduce stage 3's PyTorch result on the same prompts
(results/<model>/stage3/model_metrics.json, per_token_head:dequant), within --tol.

Controls: a fresh BF16 engine ("auto") gives the noise floor; it must agree with the
reference almost perfectly or the comparison is not trustworthy (verdict INVALID).
FP8 is reported for scale. Greedy generations exercise the real decode path and
CUDA graphs. Graphs on and off are not bit-identical in SGLang (decode KV splits follow
the padded graph batch; prefill runs piecewise graphs), so greedy text can flip on
near-ties even in BF16. BF16 with graphs off is the control: INT8 may diverge between
graphs on and off on at most --graph-slack more prompts than BF16 does.
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


def verdict(top1: float, expected: float, tol: float, noise_floor: float) -> str:
    if noise_floor < NOISE_FLOOR_MIN:
        return "INVALID"
    return "PASS" if top1 >= expected - tol else "FAIL"


def stage3_expected(metrics_path: Path, n_short: int, n_needle: int,
                    config: str = "per_token_head:dequant") -> Dict[str, float]:
    """Stage 3's per-prompt top-1, averaged over the same prompts L1 runs (first n of each kind)."""
    rows = json.loads(Path(metrics_path).read_text())["results"][config]["per_prompt"]
    picked = {}
    for kind, n in (("short", n_short), ("needle", n_needle)):
        of_kind = [r["top1"] for r in rows if r["kind"] == kind]
        if len(of_kind) < n:
            raise ValueError(f"stage 3 ran {len(of_kind)} {kind} prompts, L1 asks for {n}")
        picked[kind] = of_kind[:n]
    every = picked["short"] + picked["needle"]
    return {kind: sum(v) / len(v) for kind, v in (*picked.items(), ("all", every)) if v}


def build_prompts(tokenizer, needles: int, needle_facts: int) -> List[Tuple[str, List[int]]]:
    from int8kv.prompts import SHORT_PROMPTS, make_needle_prompt

    texts = [("short", p) for p in SHORT_PROMPTS]
    texts += [("needle", make_needle_prompt(needle_facts, seed)["prompt"]) for seed in range(needles)]

    def encode(text: str) -> List[int]:
        chat = tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
        return tokenizer(chat, add_special_tokens=False).input_ids

    return [(kind, encode(t)) for kind, t in texts]


def noise_floor(top1_by_item: Dict[str, float]) -> float:
    """BF16 with graphs on runs exactly as the reference does; graph-off BF16 is a control, not the floor."""
    return min(v for item, v in top1_by_item.items() if item.startswith("auto:") and item.endswith(":on"))


def trace_from_meta(meta: dict) -> dict:
    """Greedy output with each chosen token's logprob and its top-1 minus top-2 logprob gap."""
    return {
        "tokens": [tok for _, tok, _ in meta["output_token_logprobs"]],
        "lps": [lp for lp, _, _ in meta["output_token_logprobs"]],
        "margins": [top[0][0] - top[1][0] if len(top) > 1 else float("inf") for top in meta["output_top_logprobs"]],
    }


def decode_path_compare(a: Sequence[dict], b: Sequence[dict]) -> dict:
    """Compare two greedy runs of the same prompts: logprob drift while they agree, and how close a tie each flip was."""
    drift, margins = [], []
    for ta, tb in zip(a, b):
        n = min(len(ta["tokens"]), len(tb["tokens"]))
        d = next((j for j in range(n) if ta["tokens"][j] != tb["tokens"][j]), None)
        drift += [abs(ta["lps"][j] - tb["lps"][j]) for j in range(n if d is None else d)]
        if d is not None:
            margins.append(max(ta["margins"][d], tb["margins"][d]))
    return {"n": len(a), "n_diverged": len(margins),
            "prefix_mean_abs_dlogp": sum(drift) / len(drift) if drift else 0.0, "divergence_margins": margins}


def graph_verdict(int8: dict, bf16: Optional[dict], drift_factor: float = 3.0, drift_floor: float = 0.01,
                  tie_margin: float = 0.1) -> dict:
    """Graphs on vs off are not bit-identical, so INT8 may drift like BF16 does and flip only on near-ties.

    A flip is a near-tie when, in at least one run, the top-2 logprob gap at that step was below tie_margin.
    A real graph-path bug shows as drift well beyond BF16's or flips where the model was confident."""
    limit = max(drift_factor * bf16["prefix_mean_abs_dlogp"] if bf16 else 0.0, drift_floor)
    drift_ok = int8["prefix_mean_abs_dlogp"] <= limit
    ties_ok = all(m < tie_margin for m in int8["divergence_margins"])
    return {"pass": drift_ok and ties_ok, "drift_limit": limit, "tie_margin": tie_margin,
            "drift_ok": drift_ok, "ties_ok": ties_ok}


def engine_kwargs(model: str, kv_dtype: str, cuda_graph: bool, chunked_prefill_size: int = -1) -> dict:
    return dict(model_path=model, attention_backend="triton", kv_cache_dtype=kv_dtype,
                disable_cuda_graph=not cuda_graph, chunked_prefill_size=chunked_prefill_size,
                mem_fraction_static=0.7, log_level="error")


def make_engine(model: str, kv_dtype: str, impl: str, cuda_graph: bool, chunked_prefill_size: int = -1):
    import sglang

    os.environ["SGLANG_INT8_KV_IMPL"] = impl
    return sglang.Engine(**engine_kwargs(model, kv_dtype, cuda_graph, chunked_prefill_size))


def greedy(engine, prompts, max_new_tokens: int) -> List[dict]:
    engine.flush_cache()  # prompts must be prefilled fresh, not read back from the cache
    outs = engine.generate(
        input_ids=[ids for _, ids in prompts],
        sampling_params={"max_new_tokens": max_new_tokens, "temperature": 0},
        return_logprob=True, top_logprobs_num=2,
    )
    return [trace_from_meta(o["meta_info"]) for o in outs]


def run_config(model, kv_dtype, impl, cuda_graph, prompts, continuations=None, max_new_tokens=64,
               chunked_prefill_size=-1):
    engine = make_engine(model, kv_dtype, impl, cuda_graph, chunked_prefill_size)
    try:
        generated = greedy(engine, prompts, max_new_tokens)
        engine.flush_cache()
        scores = forced_decode_scores(engine, prompts, continuations or [t["tokens"] for t in generated])
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
                        default="auto:plain:on,auto:plain:off,fp8_e4m3:plain:on,int8_per_token_head:plain:on,int8_per_token_head:plain:off",
                        help="comma list of kv_dtype:impl:cuda_graph(on|off); must include auto (noise floor)")
    parser.add_argument("--needles", type=int, default=10)
    parser.add_argument("--needle-facts", type=int, default=150)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--stage3-metrics", type=Path, default=None,
                        help="stage 3 model_metrics.json (default: results/<model name>/stage3/model_metrics.json)")
    parser.add_argument("--tol", type=float, default=0.005,
                        help="allowed drop of per-prompt mean top-1 below stage 3 on the same prompts")
    parser.add_argument("--chunked-prefill-size", type=int, default=-1,
                        help="-1 (default) prefills each prompt in one pass, as stage 3 did; 2048 is SGLang's default here")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    configs = [tuple(item.split(":")) for item in args.configs.split(",")]
    if ("auto", "plain", "on") not in configs:
        parser.error("--configs must include auto:plain:on (BF16 with graphs): it is the noise floor")

    from transformers import AutoTokenizer
    from int8kv.prompts import SHORT_PROMPTS

    stage3_path = args.stage3_metrics or ROOT / "results" / Path(args.model).name / "stage3" / "model_metrics.json"
    expected = stage3_expected(stage3_path, len(SHORT_PROMPTS), args.needles)
    print(f"stage 3 on the same prompts: top1/prompt={expected['all']:.4f} short={expected['short']:.4f} "
          f"needle={expected.get('needle', float('nan')):.4f} (pass if INT8 >= {expected['all'] - args.tol:.4f}); "
          f"chunked_prefill_size={args.chunked_prefill_size}", flush=True)
    prompts = build_prompts(AutoTokenizer.from_pretrained(args.model), args.needles, args.needle_facts)
    traces, ref = run_config(args.model, "auto", "plain", True, prompts, max_new_tokens=args.max_new_tokens,
                                    chunked_prefill_size=args.chunked_prefill_size)
    continuations = [t["tokens"] for t in traces]
    print(f"reference: {len(prompts)} prompts, {sum(map(len, continuations))} continuation tokens", flush=True)

    results, generations = {}, {}
    for kv_dtype, impl, graph in configs:
        item = f"{kv_dtype}:{impl}:{graph}"
        generated, scores = run_config(args.model, kv_dtype, impl, graph == "on", prompts, continuations,
                                       args.max_new_tokens, args.chunked_prefill_size)
        generations[item] = generated
        summary = summarize(prompts, ref, scores)
        summary["generation_vs_bf16"] = generation_agreement(continuations, [t["tokens"] for t in generated])
        results[item] = summary

    floor = noise_floor({item: r["all"]["top1_per_prompt_mean"] for item, r in results.items()})
    failed = False
    for item, summary in results.items():
        top1 = summary["all"]["top1_per_prompt_mean"]
        if item.startswith("int8_"):
            summary["verdict"] = verdict(top1, expected["all"], args.tol, floor)
        elif item.startswith("auto:"):
            summary["verdict"] = "NOISE-FLOOR" if item.endswith(":on") else "GRAPH-CONTROL"
        else:
            summary["verdict"] = "REFERENCE-ONLY"
        failed |= summary["verdict"] in ("FAIL", "INVALID")
        gen = summary["generation_vs_bf16"]
        print(f"L1 {item}: {summary['verdict']} top1/prompt={top1:.4f} top1/token={summary['all']['top1_agreement']:.4f} "
              f"|dlogp|={summary['all']['mean_abs_logprob_diff']:.4f} "
              f"short={summary['short']['top1_per_prompt_mean']:.4f} needle={summary['needle']['top1_per_prompt_mean']:.4f} "
              f"gen_exact={gen['exact_rate']:.2f} first_div={gen['mean_first_divergence']}", flush=True)

    on, off = "int8_per_token_head:plain:on", "int8_per_token_head:plain:off"
    graphs = None
    if on in generations and off in generations:
        int8 = decode_path_compare(generations[on], generations[off])
        bf16 = (decode_path_compare(generations["auto:plain:on"], generations["auto:plain:off"])
                if "auto:plain:on" in generations and "auto:plain:off" in generations else None)
        graphs = {"int8": int8, "bf16": bf16, **graph_verdict(int8, bf16)}
        failed |= not graphs["pass"]
        bf16_txt = (f"bf16 diverged {bf16['n_diverged']}/{bf16['n']} drift={bf16['prefix_mean_abs_dlogp']:.4f}"
                    if bf16 else "no bf16 control")
        print(f"L1 cuda-graph on vs off (decode path): {'PASS' if graphs['pass'] else 'FAIL'} "
              f"int8 diverged {int8['n_diverged']}/{int8['n']} drift={int8['prefix_mean_abs_dlogp']:.4f} "
              f"(limit {graphs['drift_limit']:.4f}) flip margins={[round(m, 3) for m in int8['divergence_margins']]} "
              f"(tie < {graphs['tie_margin']}) | {bf16_txt}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": args.model, "stage3_expected": expected, "tol": args.tol,
                                    "chunked_prefill_size": args.chunked_prefill_size, "noise_floor": floor,
                                    "cuda_graph_on_vs_off": graphs, "results": results}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
