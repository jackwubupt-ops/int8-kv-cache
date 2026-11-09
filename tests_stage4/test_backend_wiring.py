"""int8_per_token_head wired through the Triton backend (Task 8), end to end with sglang.Engine on Qwen2.5-1.5B.

CUDA graphs are off here to keep L0 short; L1 covers fused and plain with graphs on and off."""
import gc
import os
import shutil
from pathlib import Path

import pytest
import torch

from tests_stage4.conftest import needs_gpu

pytestmark = needs_gpu

MODEL = Path(os.environ.get("MODELS", "/root/autodl-tmp/models")) / "Qwen2.5-1.5B-Instruct"


def _encode(text):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    chat = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
    return tok(chat, add_special_tokens=False).input_ids


def _needle(facts):
    from int8kv.prompts import make_needle_prompt

    return _encode(make_needle_prompt(facts, 0)["prompt"])


def _engine(monkeypatch, impl, chunked_prefill_size=-1):
    if not MODEL.exists():
        pytest.skip(f"{MODEL} not found")
    missing = [tool for tool in ("ninja", "nvcc") if shutil.which(tool) is None]
    if missing:
        pytest.fail(f"{missing} not on PATH: SGLang's kernel JIT would kill this process (exit 137). "
                    "Run through scripts/stage4/l0.sh or test.sh, which source common.sh.")
    import sglang

    monkeypatch.setenv("SGLANG_INT8_KV_IMPL", impl)
    return sglang.Engine(model_path=str(MODEL), attention_backend="triton", kv_cache_dtype="int8_per_token_head",
                         disable_cuda_graph=True, chunked_prefill_size=chunked_prefill_size,
                         mem_fraction_static=0.6, log_level="error")


def _greedy(engine, ids, n):
    out = engine.generate(input_ids=ids, sampling_params={"max_new_tokens": n, "temperature": 0})
    return out["output_ids"], out["meta_info"]


def _shutdown(engine):
    engine.shutdown()
    gc.collect()
    torch.cuda.empty_cache()


def _with_engine(monkeypatch, impl, fn, chunked_prefill_size=-1):
    engine = _engine(monkeypatch, impl, chunked_prefill_size)
    try:
        return fn(engine)
    finally:
        _shutdown(engine)


def test_fused_engine_generates(monkeypatch):
    ids = _encode("Explain in two sentences why the sky is blue.")
    fused = _with_engine(monkeypatch, "fused", lambda e: _greedy(e, ids, 16)[0])
    plain = _with_engine(monkeypatch, "plain", lambda e: _greedy(e, ids, 16)[0])
    assert len(fused) == 16
    assert fused[:8] == plain[:8]


def test_prefix_cache_hit_matches_cold_request(monkeypatch):
    ids = _needle(150)

    def twice(engine):
        engine.flush_cache()
        cold, _ = _greedy(engine, ids, 16)
        warm, meta = _greedy(engine, ids, 16)
        return cold, warm, meta

    cold, warm, meta = _with_engine(monkeypatch, "fused", twice)
    assert meta["cached_tokens"] > len(ids) // 2, "second request should read its prefix from the radix cache"
    assert warm == cold


def test_chunked_prefill_long_prompt(monkeypatch):
    """With 512-token chunks, later chunks read earlier ones from the INT8 cache. Fused must retrieve every needle
    the plain (PyTorch dequant) implementation retrieves under the same chunking.

    Absolute correctness is not the bar: chunk_diag on 4 x 150-fact needles found BF16 itself wrong on seed 3, and
    chunked INT8 (plain and fused alike) losing the last digit on seed 0 -- a cost of the INT8 design, not of the
    fused kernels. Token equality with an unchunked run measured near-tie noise on 600-fact needles."""
    from int8kv.prompts import make_needle_prompt

    needles = [make_needle_prompt(150, seed) for seed in range(4)]
    ids = [_encode(n["prompt"]) for n in needles]
    assert min(map(len, ids)) > 4 * 512, "prompts must span several 512-token chunks"

    def correct(engine):
        outs = engine.generate(input_ids=ids, sampling_params={"max_new_tokens": 16, "temperature": 0})
        return [n["answer"] in o["text"] for n, o in zip(needles, outs)]

    plain = _with_engine(monkeypatch, "plain", correct, chunked_prefill_size=512)
    fused = _with_engine(monkeypatch, "fused", correct, chunked_prefill_size=512)
    assert any(plain), "plain chunked INT8 retrieved no needle; the probe is too hard to compare implementations"
    missed = [seed for seed, (p, f) in enumerate(zip(plain, fused)) if p and not f]
    assert not missed, f"fused missed needles that plain retrieved: seeds {missed}"
