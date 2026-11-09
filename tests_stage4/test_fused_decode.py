"""Fused INT8 decode attention (Task 6): the grouped stage-1 kernel reads int8 K/V plus per-(token, head) scales."""
import pytest
import torch

from tests_stage4.conftest import load_sglang_file, needs_gpu
from tests_stage4.reference import reference_attention

pytestmark = needs_gpu

D = 128
MAX_SPLITS = 8


def _modules():
    patched = load_sglang_file("srt/layers/attention/triton_ops/decode_attention.py")
    pristine = load_sglang_file("srt/layers/attention/triton_ops/decode_attention.py", pristine=True)
    from sglang.srt.layers.attention.triton_ops.int8_kv_store import int8_quantize_and_store
    from sglang.srt.layers.quantization.int8_kv import dequantize_per_token_head

    return patched, pristine, int8_quantize_and_store, dequantize_per_token_head


def _case(batch, ctx, group, kv_heads, seed=0):
    """Random decode batch over a shuffled KV pool. Sequence i has max(1, ctx - 3i) tokens."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    lens = [max(1, ctx - 3 * i) for i in range(batch)]
    slots = sum(lens) + 64
    hq = group * kv_heads
    k = torch.randn(slots, kv_heads, D, device="cuda", generator=g)
    k = (k * torch.logspace(0, 2, kv_heads, device="cuda").view(1, -1, 1)).to(torch.bfloat16)
    v = torch.randn(slots, kv_heads, D, device="cuda", generator=g).to(torch.bfloat16)
    q = torch.randn(batch, hq, D, device="cuda", generator=g).to(torch.bfloat16)
    perm = torch.randperm(slots, device="cuda", generator=g)
    kv_indices = perm[: sum(lens)].to(torch.int64)
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
    kv_indptr[1:] = torch.tensor(lens, device="cuda").cumsum(0)
    return dict(q=q, k=k, v=v, lens=lens, kv_indptr=kv_indptr, kv_indices=kv_indices)


def _run(mod, c, k_buf, v_buf, k_scale=1.0, v_scale=1.0, **extra):
    batch, hq = c["q"].shape[:2]
    o = torch.empty(batch, hq, D, dtype=torch.bfloat16, device="cuda")
    logits = torch.empty(batch, hq, MAX_SPLITS, D, dtype=torch.float32, device="cuda")
    lse = torch.empty(batch, hq, MAX_SPLITS, dtype=torch.float32, device="cuda")
    splits = torch.full((batch,), MAX_SPLITS, dtype=torch.int32, device="cuda")
    mod.decode_attention_fwd(c["q"], k_buf, v_buf, o, c["kv_indptr"], c["kv_indices"], logits, lse, splits,
                             MAX_SPLITS, D**-0.5, k_scale, v_scale, **extra)
    return o


def _quantize(store, x):
    q = torch.empty(x.shape, dtype=torch.int8, device="cuda")
    s = torch.empty(x.shape[:2], dtype=torch.bfloat16, device="cuda")
    store(x, torch.arange(x.shape[0], device="cuda"), q, s)
    return q, s


def _rel_l2(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("ctx", [1, 17, 1000, 4097])
@pytest.mark.parametrize("group,kv_heads", [(6, 2), (7, 4)])
def test_fused_decode_matches_reference(batch, ctx, group, kv_heads):
    patched, _, store, dequant = _modules()
    c = _case(batch, ctx, group, kv_heads, seed=ctx + batch)
    kq, ks = _quantize(store, c["k"])
    vq, vs = _quantize(store, c["v"])
    out = _run(patched, c, kq, vq, k_scale_buffer=ks, v_scale_buffer=vs)
    kd, vd = dequant(kq, ks), dequant(vq, vs)
    starts = c["kv_indptr"].tolist()
    for b in range(batch):
        idx = c["kv_indices"][starts[b] : starts[b + 1]]
        ref = reference_attention(c["q"][b : b + 1], kd[idx], vd[idx])[0]
        assert _rel_l2(out[b], ref) < 1e-2


@pytest.mark.parametrize("ctx", [17, 4097])
def test_fused_decode_matches_plain(ctx):
    patched, _, store, dequant = _modules()
    c = _case(4, ctx, 7, 4, seed=ctx)
    kq, ks = _quantize(store, c["k"])
    vq, vs = _quantize(store, c["v"])
    fused = _run(patched, c, kq, vq, k_scale_buffer=ks, v_scale_buffer=vs)
    plain = _run(patched, c, dequant(kq, ks), dequant(vq, vs))
    cos = torch.nn.functional.cosine_similarity(fused.float().flatten(1), plain.float().flatten(1), dim=-1)
    assert cos.min().item() > 0.9999


@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_flag_off_bitwise_equals_pristine(kv_dtype):
    patched, pristine, _, _ = _modules()
    c = _case(4, 1000, 7, 4, seed=3)
    k_scale = v_scale = 1.0
    k_buf, v_buf = c["k"], c["v"]
    if kv_dtype != torch.bfloat16:
        k_scale, v_scale = 0.5, 0.25
        k_buf = (c["k"].float() / k_scale).to(kv_dtype)
        v_buf = (c["v"].float() / v_scale).to(kv_dtype)
    a = _run(patched, c, k_buf, v_buf, k_scale, v_scale)
    b = _run(pristine, c, k_buf, v_buf, k_scale, v_scale)
    assert torch.equal(a, b)


def test_non_grouped_rejected():
    patched, _, store, _ = _modules()
    c = _case(1, 17, 1, 4)
    kq, ks = _quantize(store, c["k"])
    vq, vs = _quantize(store, c["v"])
    with pytest.raises(NotImplementedError):
        _run(patched, c, kq, vq, k_scale_buffer=ks, v_scale_buffer=vs)


def test_int8_rejects_tensor_wide_scales():
    patched, _, store, _ = _modules()
    c = _case(1, 17, 7, 4)
    kq, ks = _quantize(store, c["k"])
    vq, vs = _quantize(store, c["v"])
    with pytest.raises(ValueError):
        _run(patched, c, kq, vq, k_scale=0.5, k_scale_buffer=ks, v_scale_buffer=vs)


def test_int8_needs_both_scale_buffers():
    patched, _, store, _ = _modules()
    c = _case(1, 17, 7, 4)
    kq, ks = _quantize(store, c["k"])
    vq, _ = _quantize(store, c["v"])
    with pytest.raises(ValueError):
        _run(patched, c, kq, vq, k_scale_buffer=ks)
