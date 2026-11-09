"""Fused INT8 extend attention (Task 7): the prefix read from K_Buffer is int8 + per-(token, head) scales;
the tokens of this extend stay in their original precision."""
import pytest
import torch

from tests_stage4.conftest import load_sglang_file, needs_gpu
from tests_stage4.reference import reference_attention

pytestmark = needs_gpu

D = 128
GROUP, KV_HEADS = 7, 4


def _modules():
    patched = load_sglang_file("srt/layers/attention/triton_ops/extend_attention.py")
    pristine = load_sglang_file("srt/layers/attention/triton_ops/extend_attention.py", pristine=True)
    from sglang.srt.layers.attention.triton_ops.int8_kv_store import int8_quantize_and_store
    from sglang.srt.layers.quantization.int8_kv import dequantize_per_token_head

    return patched, pristine, int8_quantize_and_store, dequantize_per_token_head


def _kv(n, g):
    """KV heads span 1x-100x so their int8 scales differ."""
    x = torch.randn(n, KV_HEADS, D, device="cuda", generator=g)
    return (x * torch.logspace(0, 2, KV_HEADS, device="cuda").view(1, -1, 1)).to(torch.bfloat16)


def _case(prefix, extend, seed=0):
    """Two sequences: (prefix, extend) and (max(0, prefix - 2), max(1, extend - 3)), prefixes in a shuffled pool.

    q is scaled by 0.05 so attention logits stay in a realistic range (see test_fused_decode._case)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    prefixes, extends = [prefix, max(0, prefix - 2)], [extend, max(1, extend - 3)]
    slots = sum(prefixes) + 64
    k_pool, v_pool = _kv(slots, g), _kv(slots, g)
    n_ext = sum(extends)
    q = (torch.randn(n_ext, GROUP * KV_HEADS, D, device="cuda", generator=g) * 0.05).to(torch.bfloat16)
    k_ext, v_ext = _kv(n_ext, g), _kv(n_ext, g)
    perm = torch.randperm(slots, device="cuda", generator=g)[: sum(prefixes)].to(torch.int64)
    qo_indptr = torch.tensor([0, extends[0], n_ext], dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor([0, prefixes[0], sum(prefixes)], dtype=torch.int32, device="cuda")
    return dict(q=q, k_ext=k_ext, v_ext=v_ext, k_pool=k_pool, v_pool=v_pool, kv_indices=perm,
                qo_indptr=qo_indptr, kv_indptr=kv_indptr, prefixes=prefixes, extends=extends)


def _run(mod, c, k_buf, v_buf, k_scale=1.0, v_scale=1.0, **extra):
    o = torch.empty_like(c["q"])
    mod.extend_attention_fwd(c["q"], c["k_ext"], c["v_ext"], o, k_buf, v_buf, c["qo_indptr"], c["kv_indptr"],
                             c["kv_indices"], None, True, None, max(c["extends"]), k_scale, v_scale, **extra)
    return o


def _quantize(store, x):
    q = torch.empty(x.shape, dtype=torch.int8, device="cuda")
    s = torch.empty(x.shape[:2], dtype=torch.bfloat16, device="cuda")
    store(x, torch.arange(x.shape[0], device="cuda"), q, s)
    return q, s


def _exact_dequant(q, s):
    return q.float() * s.float().unsqueeze(-1)


def _int8_pool(store, c):
    kq, ks = _quantize(store, c["k_pool"])
    vq, vs = _quantize(store, c["v_pool"])
    return kq, ks, vq, vs


@pytest.mark.parametrize("prefix", [0, 5, 300])
@pytest.mark.parametrize("extend", [1, 64, 257])
def test_fused_extend_matches_reference(prefix, extend):
    patched, _, store, _ = _modules()
    c = _case(prefix, extend, seed=prefix * 1000 + extend)
    kq, ks, vq, vs = _int8_pool(store, c)
    out = _run(patched, c, kq, vq, k_scale_buffer=ks, v_scale_buffer=vs)
    kd, vd = _exact_dequant(kq, ks), _exact_dequant(vq, vs)
    q0 = k0 = 0
    for p, e in zip(c["prefixes"], c["extends"]):
        idx = c["kv_indices"][k0 : k0 + p]
        keys = torch.cat([kd[idx], c["k_ext"][q0 : q0 + e].float()])
        vals = torch.cat([vd[idx], c["v_ext"][q0 : q0 + e].float()])
        ref = reference_attention(c["q"][q0 : q0 + e], keys, vals, causal_offset=p)
        got = out[q0 : q0 + e].float()
        assert ((got - ref).norm() / ref.norm()).item() < 1e-2
        q0, k0 = q0 + e, k0 + p


@pytest.mark.parametrize("prefix,extend", [(5, 64), (300, 257)])
def test_fused_extend_matches_plain(prefix, extend):
    patched, _, store, dequant = _modules()
    c = _case(prefix, extend, seed=7)
    kq, ks, vq, vs = _int8_pool(store, c)
    fused = _run(patched, c, kq, vq, k_scale_buffer=ks, v_scale_buffer=vs)
    plain = _run(patched, c, dequant(kq, ks), dequant(vq, vs))
    cos = torch.nn.functional.cosine_similarity(fused.float().flatten(1), plain.float().flatten(1), dim=-1)
    assert cos.min().item() > 0.9999


def test_fused_extend_without_prefix_ignores_scales():
    """With no cached prefix, the INT8 path reads nothing from the pool and must equal the BF16 kernel exactly."""
    patched, _, store, _ = _modules()
    c = _case(0, 64, seed=11)
    kq, ks, vq, vs = _int8_pool(store, c)
    fused = _run(patched, c, kq, vq, k_scale_buffer=ks, v_scale_buffer=vs)
    plain = _run(patched, c, c["k_pool"], c["v_pool"])
    assert torch.equal(fused, plain)


@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_extend_flag_off_bitwise_equals_pristine(kv_dtype):
    patched, pristine, _, _ = _modules()
    c = _case(300, 64, seed=3)
    k_scale = v_scale = 1.0
    k_buf, v_buf = c["k_pool"], c["v_pool"]
    if kv_dtype != torch.bfloat16:
        k_scale = c["k_pool"].float().abs().max().item() / 200
        v_scale = c["v_pool"].float().abs().max().item() / 200
        k_buf = (c["k_pool"].float() / k_scale).to(kv_dtype)
        v_buf = (c["v_pool"].float() / v_scale).to(kv_dtype)
    a = _run(patched, c, k_buf, v_buf, k_scale, v_scale)
    b = _run(pristine, c, k_buf, v_buf, k_scale, v_scale)
    assert torch.isfinite(a.float()).all()
    assert torch.equal(a, b)


def test_int8_rejects_tensor_wide_scales():
    patched, _, store, _ = _modules()
    c = _case(5, 8)
    kq, ks, vq, vs = _int8_pool(store, c)
    with pytest.raises(ValueError):
        _run(patched, c, kq, vq, v_scale=0.5, k_scale_buffer=ks, v_scale_buffer=vs)


def test_int8_needs_both_scale_buffers():
    patched, _, store, _ = _modules()
    c = _case(5, 8)
    kq, _, vq, vs = _int8_pool(store, c)
    with pytest.raises(ValueError):
        _run(patched, c, kq, vq, v_scale_buffer=vs)


def test_unified_extend_rejects_int8():
    patched, _, store, _ = _modules()
    c = _case(5, 8)
    kq, ks, vq, vs = _int8_pool(store, c)
    o = torch.empty_like(c["q"])
    prefix_lens = torch.tensor(c["prefixes"], dtype=torch.int32, device="cuda")
    with pytest.raises(NotImplementedError):
        patched.extend_attention_fwd_unified(c["q"], o, kq, vq, 1.0, 1.0, c["qo_indptr"], c["kv_indptr"],
                                             c["kv_indices"], prefix_lens, max(c["extends"]),
                                             k_scale_buffer=ks, v_scale_buffer=vs)
