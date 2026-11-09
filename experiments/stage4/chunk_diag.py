"""Diagnose chunked prefill on long needle prompts: is a chunked/whole difference INT8 semantics or a fused-kernel bug?

For each needle seed, runs (kv dtype, impl, chunk size) combinations with greedy decoding and prints the answer,
each output, and whether it contains the code. If int8 fused-chunked == int8 plain-chunked, the fused kernels
match the reference implementation and any chunked/whole gap is the cost of reading earlier chunks from INT8.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CONFIGS = [("auto", "plain", -1), ("auto", "plain", 512),
           ("int8_per_token_head", "plain", -1), ("int8_per_token_head", "plain", 512),
           ("int8_per_token_head", "fused", -1), ("int8_per_token_head", "fused", 512)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--facts", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    import sglang
    import torch
    from transformers import AutoTokenizer

    from int8kv.prompts import make_needle_prompt

    tok = AutoTokenizer.from_pretrained(args.model)
    needles = [make_needle_prompt(args.facts, s) for s in range(args.seeds)]

    def encode(text):
        chat = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
        return tok(chat, add_special_tokens=False).input_ids

    ids = [encode(n["prompt"]) for n in needles]
    print(f"{args.seeds} needles x {args.facts} facts, prompt tokens {[len(x) for x in ids]}", flush=True)
    rows = []
    for kv, impl, chunk in CONFIGS:
        os.environ["SGLANG_INT8_KV_IMPL"] = impl
        engine = sglang.Engine(model_path=args.model, attention_backend="triton", kv_cache_dtype=kv,
                               disable_cuda_graph=True, chunked_prefill_size=chunk, mem_fraction_static=0.6,
                               log_level="error")
        try:
            outs = engine.generate(input_ids=ids, sampling_params={"max_new_tokens": args.max_new_tokens,
                                                                    "temperature": 0})
        finally:
            engine.shutdown()
            gc.collect()
            torch.cuda.empty_cache()
        for seed, (n, o) in enumerate(zip(needles, outs)):
            rows.append({"kv": kv, "impl": impl, "chunk": chunk, "seed": seed, "answer": n["answer"],
                         "tokens": o["output_ids"], "text": o["text"], "correct": n["answer"] in o["text"]})
    by_key = {(r["kv"], r["impl"], r["chunk"], r["seed"]): r for r in rows}
    for seed in range(args.seeds):
        print(f"--- seed {seed}: answer {needles[seed]['answer']}", flush=True)
        for kv, impl, chunk in CONFIGS:
            r = by_key[(kv, impl, chunk, seed)]
            print(f"DIAG {kv:20s} {impl:5s} chunk={chunk:4d} {'OK ' if r['correct'] else 'BAD'} {r['text']!r}",
                  flush=True)
    same = [by_key[("int8_per_token_head", "fused", 512, s)]["tokens"]
            == by_key[("int8_per_token_head", "plain", 512, s)]["tokens"] for s in range(args.seeds)]
    print(f"DIAG int8 fused-chunked == plain-chunked: {sum(same)}/{args.seeds}", flush=True)
    for kv, impl, chunk in CONFIGS:
        ok = sum(by_key[(kv, impl, chunk, s)]["correct"] for s in range(args.seeds))
        print(f"DIAG correct {kv}:{impl}:chunk={chunk}: {ok}/{args.seeds}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
