"""Fused INT8 quantize-and-store Triton kernel (Task 5)."""
import pytest
import torch

from tests_stage4.conftest import load_sglang_file, needs_gpu

M = load_sglang_file("srt/layers/quantization/int8_kv.py")

D = 128


def test_impl_defaults_to_fused(monkeypatch):
    monkeypatch.delenv("SGLANG_INT8_KV_IMPL", raising=False)
    assert M.int8_kv_impl() == "fused"


def test_impl_plain_selects_torch(monkeypatch):
    monkeypatch.setenv("SGLANG_INT8_KV_IMPL", "plain")
    assert M.int8_kv_impl() == "plain"


def test_impl_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("SGLANG_INT8_KV_IMPL", "fast")
    with pytest.raises(ValueError, match="SGLANG_INT8_KV_IMPL"):
        M.int8_kv_impl()


def _kernel():
    from sglang.srt.layers.attention.triton_ops.int8_kv_store import int8_quantize_and_store

    return int8_quantize_and_store


def _cache(n, h, seed=0, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(n, h, D, device="cuda", generator=g)
    mult = torch.logspace(0, torch.log10(torch.tensor(300.0)).item(), h, device="cuda").view(1, h, 1)
    return (x * mult).to(dtype)


def _buffers(slots, h):
    q_buf = torch.full((slots, h, D), 55, dtype=torch.int8, device="cuda")
    scale_buf = torch.full((slots, h), 3.0, dtype=torch.bfloat16, device="cuda")
    return q_buf, scale_buf


@needs_gpu
@pytest.mark.parametrize("n", [1, 7, 4096])
@pytest.mark.parametrize("h", [2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_store_matches_torch_reference(n, h, dtype):
    cache = _cache(n, h, seed=n * 10 + h, dtype=dtype)
    slots = n + 16
    loc = torch.randperm(slots, device="cuda")[:n]
    q_buf, scale_buf = _buffers(slots, h)
    _kernel()(cache, loc, q_buf, scale_buf)
    ref_q, ref_scale = M.quantize_per_token_head(cache)
    assert torch.equal(scale_buf[loc], ref_scale)
    mismatch = (q_buf[loc] != ref_q).float().mean().item()
    assert mismatch < 1e-4


@needs_gpu
def test_store_accepts_strided_cache():
    """K and V arrive as views of a fused qkv projection, so rows are not contiguous."""
    h, n = 4, 33
    qkv = _cache(n, 3 * h)
    k = qkv[:, h : 2 * h]
    assert not k.is_contiguous()
    loc = torch.arange(n, device="cuda")
    q_buf, scale_buf = _buffers(n, h)
    _kernel()(k, loc, q_buf, scale_buf)
    ref_q, ref_scale = M.quantize_per_token_head(k)
    assert torch.equal(scale_buf, ref_scale) and torch.equal(q_buf, ref_q)


@needs_gpu
def test_store_only_touches_loc():
    h, n, slots = 4, 5, 32
    cache = _cache(n, h)
    loc = torch.tensor([3, 9, 10, 20, 31], device="cuda")
    q_buf, scale_buf = _buffers(slots, h)
    q_before, s_before = q_buf.clone(), scale_buf.clone()
    _kernel()(cache, loc, q_buf, scale_buf)
    untouched = torch.ones(slots, dtype=torch.bool, device="cuda")
    untouched[loc] = False
    assert torch.equal(q_buf[untouched], q_before[untouched])
    assert torch.equal(scale_buf[untouched], s_before[untouched])


@needs_gpu
def test_store_zero_rows():
    h, n = 4, 3
    cache = _cache(n, h)
    cache[1] = 0
    cache[2, 2] = 0
    loc = torch.arange(n, device="cuda")
    q_buf, scale_buf = _buffers(n, h)
    _kernel()(cache, loc, q_buf, scale_buf)
    assert torch.equal(scale_buf[1], torch.ones(h, dtype=torch.bfloat16, device="cuda"))
    assert scale_buf[2, 2].item() == 1.0
    assert (q_buf[1] == 0).all() and (q_buf[2, 2] == 0).all()
    assert torch.isfinite(scale_buf.float()).all()


@needs_gpu
def test_store_empty_batch_is_a_no_op():
    q_buf, scale_buf = _buffers(4, 2)
    _kernel()(torch.empty(0, 2, D, dtype=torch.bfloat16, device="cuda"),
              torch.empty(0, dtype=torch.int64, device="cuda"), q_buf, scale_buf)
    assert (q_buf == 55).all()


@needs_gpu
@pytest.mark.parametrize("impl", ["fused", "plain"])
def test_pool_set_kv_buffer_uses_selected_impl(monkeypatch, impl):
    """Both implementations write identical scales through the pool."""
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolINT8

    monkeypatch.setenv("SGLANG_INT8_KV_IMPL", impl)
    h = 4
    pool = MHATokenToKVPoolINT8(16, page_size=1, dtype=torch.int8, head_num=h, head_dim=D, layer_num=1,
                                device="cuda", enable_memory_saver=False, compute_dtype=torch.bfloat16)
    k, v = _cache(6, h, seed=1), _cache(6, h, seed=2)
    loc = torch.tensor([1, 2, 5, 8, 9, 15], device="cuda")
    pool.set_kv_buffer(None, loc, k, v, layer_id_override=0)
    k_scale, v_scale = pool.get_kv_scale_buffer(0)
    assert torch.equal(k_scale[loc], M.quantize_per_token_head(k)[1])
    assert torch.equal(v_scale[loc], M.quantize_per_token_head(v)[1])
