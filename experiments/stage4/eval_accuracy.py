"""Stage 4 accuracy on SGLang: GSM8K and needle retrieval for one KV-cache configuration.

  python experiments/stage4/eval_accuracy.py --model M --kv-dtype int8_per_token_head --impl fused --task gsm8k
  python experiments/stage4/eval_accuracy.py --summarize results/stage4/Qwen2.5-7B-Instruct/accuracy

Each run writes <out>/<task>_<config>.json with per-question predictions. --summarize pairs every quantized
configuration with bf16 on the same questions (exact McNemar test) and writes accuracy_summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from int8kv.metrics import mcnemar_exact  # noqa: E402
from int8kv.prompts import GSM8K_INSTRUCTION, extract_number, make_needle_prompt, needle_correct  # noqa: E402

GSM8K_FILE = ROOT / "data" / "gsm8k" / "test.jsonl"
NEEDLE_FACTS = {"needle150": 150, "needle600": 600}
TASKS = ("gsm8k", *NEEDLE_FACTS)
BASELINE = "bf16"


def config_name(kv_dtype: str, impl: str) -> str:
    if kv_dtype == "auto":
        return BASELINE
    if kv_dtype == "int8_per_token_head":
        return f"int8_{impl}"
    return kv_dtype


def build_prompt(task: str, item: dict) -> str:
    return item["question"] + GSM8K_INSTRUCTION if task == "gsm8k" else item["prompt"]


def load_items(task: str, n: int, data_file: Optional[Path] = None) -> List[dict]:
    """Up to n items, each with "prompt" and "gold"."""
    if task == "gsm8k":
        lines = Path(data_file or GSM8K_FILE).read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()][:n]
        return [{"prompt": build_prompt(task, r), "gold": extract_number(r["answer"])} for r in rows]
    if task in NEEDLE_FACTS:
        needles = [make_needle_prompt(NEEDLE_FACTS[task], seed) for seed in range(n)]
        return [{"prompt": build_prompt(task, x), "gold": x["answer"]} for x in needles]
    raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")


def score(task: str, text: str, gold: str) -> dict:
    if task == "gsm8k":
        pred = extract_number(text)
        return {"pred": pred, "correct": pred is not None and pred == gold}
    return {"pred": text.strip()[:80], "correct": needle_correct(text, gold)}


def run_eval(engine, tokenizer, task: str, n: int, max_new_tokens: int, data_file: Optional[Path] = None) -> dict:
    items = load_items(task, n, data_file)

    def encode(text: str) -> List[int]:
        chat = tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                             tokenize=False)
        return tokenizer(chat, add_special_tokens=False).input_ids

    outs = engine.generate(input_ids=[encode(x["prompt"]) for x in items],
                           sampling_params={"max_new_tokens": max_new_tokens, "temperature": 0})
    predictions = []
    for i, (item, out) in enumerate(zip(items, outs)):
        s = score(task, out["text"], item["gold"])
        predictions.append({"id": i, "pred": s["pred"], "gold": item["gold"], "correct": s["correct"]})
    correct = sum(p["correct"] for p in predictions)
    return {"accuracy": correct / len(predictions) if predictions else 0.0, "correct": correct,
            "total": len(predictions), "predictions": predictions}


def summarize_accuracy(directory: Path) -> Dict[str, Dict[str, dict]]:
    """{task: {config: paired comparison with bf16}} over <task>_<config>.json files in directory."""
    directory = Path(directory)
    runs: Dict[str, Dict[str, dict]] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name == "accuracy_summary.json":
            continue
        data = json.loads(path.read_text())
        runs.setdefault(data["task"], {})[data["config"]] = data
    summary: Dict[str, Dict[str, dict]] = {}
    for task, configs in runs.items():
        summary[task] = {}
        base = configs.get(BASELINE)
        if base is None:
            continue
        base_correct = {p["id"]: p["correct"] for p in base["predictions"]}
        for config, data in configs.items():
            if config == BASELINE:
                continue
            other = {p["id"]: p["correct"] for p in data["predictions"]}
            if other.keys() != base_correct.keys():
                raise ValueError(f"{task}: {config} and {BASELINE} must be paired on the same questions "
                                 f"({len(other)} vs {len(base_correct)})")
            only_b = sum(base_correct[i] and not other[i] for i in other)
            only_q = sum(other[i] and not base_correct[i] for i in other)
            summary[task][config] = {
                "accuracy": sum(other.values()) / len(other), "baseline_accuracy": base["accuracy"],
                "delta": sum(other.values()) / len(other) - base["accuracy"], "total": len(other),
                "only_baseline": only_b, "only_quantized": only_q, "mcnemar_p": mcnemar_exact(only_b, only_q),
            }
    (directory / "accuracy_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--kv-dtype", default="auto", choices=["auto", "fp8_e4m3", "int8_per_token_head"])
    ap.add_argument("--impl", default="fused", choices=["fused", "plain"])
    ap.add_argument("--task", choices=TASKS)
    ap.add_argument("--n", type=int, default=None, help="questions (default: 1319 for gsm8k, 100 for needles)")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="default: 1024 for gsm8k, 64 for needles")
    ap.add_argument("--data-file", type=Path, default=GSM8K_FILE)
    ap.add_argument("--chunked-prefill-size", type=int, default=None,
                    help="default: SGLang's own choice for this GPU (2048 on a 24 GB card)")
    ap.add_argument("--out", type=Path, default=None, help="default: results/stage4/<model name>/accuracy")
    ap.add_argument("--summarize", type=Path, default=None, help="only summarize an accuracy directory")
    args = ap.parse_args(argv)

    if args.summarize:
        summary = summarize_accuracy(args.summarize)
        for task, rows in summary.items():
            for config, r in rows.items():
                print(f"ACC {task:9s} {config:11s} {r['accuracy']:.4f} vs bf16 {r['baseline_accuracy']:.4f} "
                      f"(delta {r['delta']:+.4f}) only_bf16={r['only_baseline']} only_{config}={r['only_quantized']} "
                      f"McNemar p={r['mcnemar_p']:.3g}", flush=True)
        return 0
    if not (args.model and args.task):
        ap.error("--model and --task are required unless --summarize is given")

    import sglang
    from transformers import AutoTokenizer

    n = args.n if args.n is not None else (1319 if args.task == "gsm8k" else 100)
    max_new = args.max_new_tokens or (1024 if args.task == "gsm8k" else 64)
    config = config_name(args.kv_dtype, args.impl)
    out_dir = args.out or ROOT / "results" / "stage4" / Path(args.model).name / "accuracy"
    os.environ["SGLANG_INT8_KV_IMPL"] = args.impl
    kwargs = dict(model_path=args.model, attention_backend="triton", kv_cache_dtype=args.kv_dtype, log_level="error")
    if args.chunked_prefill_size is not None:
        kwargs["chunked_prefill_size"] = args.chunked_prefill_size
    engine = sglang.Engine(**kwargs)
    try:
        start = time.time()
        result = run_eval(engine, AutoTokenizer.from_pretrained(args.model), args.task, n, max_new, args.data_file)
        elapsed = time.time() - start
        try:  # the effective chunked_prefill_size SGLang chose, for the record
            server = engine.get_server_info()
        except Exception:
            server = {}
    finally:
        engine.shutdown()
    result.update({"task": args.task, "config": config, "model": args.model, "kv_dtype": args.kv_dtype,
                   "impl": args.impl, "max_new_tokens": max_new, "seconds": elapsed,
                   "chunked_prefill_size": server.get("chunked_prefill_size", args.chunked_prefill_size)})
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.task}_{config}.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"EVAL {args.task} {config}: accuracy={result['accuracy']:.4f} ({result['correct']}/{result['total']}) "
          f"in {elapsed:.0f}s -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
