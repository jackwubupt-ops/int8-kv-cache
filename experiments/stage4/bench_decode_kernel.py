"""Micro-benchmark of SGLang's grouped Triton decode attention kernel for one KV-cache dtype.

  python experiments/stage4/bench_decode_kernel.py --dtype int8 --batch 32 --contexts 1024,4096,16384,32768 --out X

dtypes:
  bf16        bf16 K/V, the unpatched code path
  fp8         fp8_e4m3 K/V with a static scale (SGLang's FP8 path; values kept inside e4m3's range)
  int8        int8 K/V + per-(token, head) scales, dequantized inside the kernel (this project, fused)
  int8_plain  int8 K/V dequantized to a bf16 buffer on every call, then the bf16 kernel (this project, plain).
              The buffer here holds only the tokens in use; a server dequantizes its whole pool, which is larger.

Shapes default to Qwen2.5-7B (28 query heads, 4 KV heads, head_dim 128). Each sequence owns a contiguous run of
slots. Timing uses CUDA events: `--warmup` untimed calls, then the median of `--iters` calls.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, List

import torch

DTYPES = ("bf16", "fp8", "int8", "int8_plain")
FP8_HEADROOM = 200.0  # stored |x| / scale stays below this, well inside e4m3's 448


def parse_contexts(text: str) -> List[int]:
    values = [int(x) for x in text.split(",") if x.strip()]
    if not values or any(v <= 0 for v in values):
        raise ValueError(f"contexts must be positive integers, got {text!r}")
    return values


def build_case(*, batch: int, context: int, q_heads: int, kv_heads: int, head_dim: int, kv_splits: int,
               device: str = "cuda", seed: int = 0) -> Dict[str, torch.Tensor]:
    """Decode-step inputs: `batch` sequences of `context` cached tokens each, slots laid out contiguously."""
    g = torch.Generator(device=device).manual_seed(seed)
    slots = batch * context

    def rand(*shape):
        return torch.randn(*shape, device=device, generator=g).to(torch.bfloat16)

    return {
        "q": rand(batch, q_heads, head_dim),
        "k": rand(slots, kv_heads, head_dim),
        "v": rand(slots, kv_heads, head_dim),
        "o": torch.empty(batch, q_heads, head_dim, dtype=torch.bfloat16, device=device),
        "kv_indptr": (torch.arange(batch + 1, device=device) * context).to(torch.int32),
        "kv_indices": torch.arange(slots, dtype=torch.int64, device=device),
        "attn_logits": torch.empty(batch, q_heads, kv_splits, head_dim, dtype=torch.float32, device=device),
        "attn_lse": torch.empty(batch, q_heads, kv_splits, dtype=torch.float32, device=device),
        "num_kv_splits": torch.full((batch,), kv_splits, dtype=torch.int32, device=device),
        "max_kv_splits": kv_splits,
        "sm_scale": head_dim ** -0.5,
    }


def make_runner(dtype: str, case: Dict[str, torch.Tensor], *, decode_fn: Callable, store_fn: Callable,
                dequant_fn: Callable) -> Callable[[], None]:
    """A zero-argument callable running one decode-attention step for `dtype` on `case`."""
    if dtype not in DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}; expected one of {DTYPES}")
    c = case

    def call(k_buf, v_buf, k_scale=1.0, v_scale=1.0, **scales):
        decode_fn(c["q"], k_buf, v_buf, c["o"], c["kv_indptr"], c["kv_indices"], c["attn_logits"], c["attn_lse"],
                  c["num_kv_splits"], c["max_kv_splits"], c["sm_scale"], k_scale, v_scale, **scales)

    if dtype == "bf16":
        return lambda: call(c["k"], c["v"])
    if dtype == "fp8":
        k_scale = c["k"].float().abs().max().item() / FP8_HEADROOM
        v_scale = c["v"].float().abs().max().item() / FP8_HEADROOM
        k8 = (c["k"].float() / k_scale).to(torch.float8_e4m3fn)
        v8 = (c["v"].float() / v_scale).to(torch.float8_e4m3fn)
        return lambda: call(k8, v8, k_scale, v_scale)

    def quantize(x):
        q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
        s = torch.empty(x.shape[:2], dtype=torch.bfloat16, device=x.device)
        store_fn(x, torch.arange(x.shape[0], device=x.device), q, s)
        return q, s

    kq, ks = quantize(c["k"])
    vq, vs = quantize(c["v"])
    if dtype == "int8":
        return lambda: call(kq, vq, k_scale_buffer=ks, v_scale_buffer=vs)
    return lambda: call(dequant_fn(kq, ks, dtype=torch.bfloat16), dequant_fn(vq, vs, dtype=torch.bfloat16))


def cuda_timer(fn: Callable[[], None]) -> float:
    """Milliseconds of GPU time for one call, measured with CUDA events."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def time_kernel(fn: Callable[[], None], *, warmup: int, iters: int, timer: Callable = cuda_timer) -> Dict[str, float]:
    for _ in range(warmup):
        timer(fn)
    samples = sorted(timer(fn) for _ in range(iters))
    return {"median_ms": statistics.median(samples), "min_ms": samples[0],
            "p90_ms": samples[max(0, math.ceil(0.9 * len(samples)) - 1)], "iters": iters}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dtype", required=True, choices=DTYPES)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--contexts", default="1024,4096,16384,32768")
    ap.add_argument("--q-heads", type=int, default=28)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--kv-splits", type=int, default=8, help="SGLang's default triton_attention_num_kv_splits")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    from sglang.srt.layers.attention.triton_ops.decode_attention import decode_attention_fwd
    from sglang.srt.layers.attention.triton_ops.int8_kv_store import int8_quantize_and_store
    from sglang.srt.layers.quantization.int8_kv import dequantize_per_token_head

    results = []
    for context in parse_contexts(args.contexts):
        case = build_case(batch=args.batch, context=context, q_heads=args.q_heads, kv_heads=args.kv_heads,
                          head_dim=args.head_dim, kv_splits=args.kv_splits)
        run = make_runner(args.dtype, case, decode_fn=decode_attention_fwd, store_fn=int8_quantize_and_store,
                          dequant_fn=dequantize_per_token_head)
        stats = time_kernel(run, warmup=args.warmup, iters=args.iters)
        results.append({"context": context, **stats})
        print(f"KERNEL {args.dtype:10s} batch={args.batch} context={context:6d} median={stats['median_ms']:.3f} ms "
              f"min={stats['min_ms']:.3f} p90={stats['p90_ms']:.3f}", flush=True)
        del case, run
        torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "dtype": args.dtype, "batch": args.batch, "q_heads": args.q_heads, "kv_heads": args.kv_heads,
        "head_dim": args.head_dim, "kv_splits": args.kv_splits, "warmup": args.warmup,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
