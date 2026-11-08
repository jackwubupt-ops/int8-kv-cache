"""Environment smoke test: load the model in SGLang with the Triton backend and generate."""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--kv-dtypes", default="auto,fp8_e4m3")
    args = parser.parse_args(argv)

    import sglang
    import torch

    print("sglang", sglang.__version__, sglang.__file__)
    print("torch", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
    for kv_dtype in args.kv_dtypes.split(","):
        engine = sglang.Engine(
            model_path=args.model, attention_backend="triton", kv_cache_dtype=kv_dtype,
            mem_fraction_static=0.5, log_level="error",
        )
        out = engine.generate("The capital of France is", {"max_new_tokens": 8, "temperature": 0})
        engine.shutdown()
        print(f"[{kv_dtype}] {out['text']!r}", flush=True)
        if not out["text"].strip():
            print(f"FAIL: empty generation with kv_cache_dtype={kv_dtype}", file=sys.stderr)
            return 1
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
