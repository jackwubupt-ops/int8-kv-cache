import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from int8kv.cache import QuantizedKVCache
from int8kv.metrics import first_divergence, kl_from_logits, top1_agreement
from int8kv.qwen2_patch import OperatorRecorder, patch_qwen2
from int8kv.quant import compute_scale, dequantize, quantize


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    config = Qwen2Config(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2,
        intermediate_size=128, vocab_size=100, max_position_embeddings=128, attn_implementation="sdpa",
    )
    m = Qwen2ForCausalLM(config).eval()
    patch_qwen2(m)
    return m


IDS = torch.arange(1, 13).unsqueeze(0)


def test_cache_stores_int8_and_reports_memory():
    c = QuantizedKVCache()
    k, v = torch.randn(1, 2, 5, 16), torch.randn(1, 2, 5, 16)
    c.update(k, v, 0)
    c.update(k[:, :, :1], v[:, :, :1], 0)
    ki, sk, vi, sv = c.get_kv(0)
    assert ki.dtype == torch.int8 and vi.dtype == torch.int8
    assert ki.shape == (1, 2, 6, 16) and sk.shape == (1, 2, 6, 1) and sv.shape == (1, 2, 6, 1)
    mem = c.memory_bytes()
    assert mem["kv_bytes"] == 2 * 2 * 6 * 16
    assert mem["scale_bytes"] == 2 * 2 * 6 * 2


def test_coarse_granularity_keeps_one_scale_per_write_and_counts_it_once():
    c = QuantizedKVCache(granularity="per_tensor")
    k = torch.randn(1, 2, 5, 16)
    c.update(k, k, 0)
    assert c.get_kv(0)[1].shape == (1, 2, 5, 1)
    assert c.memory_bytes()["scale_bytes"] == 2 * 2


def test_crop_keeps_scales_aligned_with_tokens():
    c = QuantizedKVCache()
    k = torch.randn(1, 2, 6, 16)
    c.update(k, k, 0)
    expected = compute_scale(k[:, :, :3])
    c.crop(3)
    ki, sk, _, _ = c.get_kv(0)
    assert ki.shape[-2] == 3 and sk.shape[-2] == 3
    assert torch.allclose(sk.float(), expected, rtol=0.01)


def test_plain_cache_path_is_unchanged_by_patch():
    torch.manual_seed(1)
    config = Qwen2Config(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2,
        intermediate_size=128, vocab_size=100, max_position_embeddings=128, attn_implementation="sdpa",
    )
    m = Qwen2ForCausalLM(config).eval()
    before = m(IDS).logits
    patch_qwen2(m)
    assert torch.equal(before, m(IDS).logits)


@pytest.mark.parametrize("mode", ["dequant", "integer"])
def test_prefill_logits_close_to_baseline(model, mode):
    with torch.no_grad():
        base = model(IDS).logits
        out = model(IDS, past_key_values=QuantizedKVCache(mode=mode), use_cache=True).logits
    assert kl_from_logits(base, out).item() < 1e-2


def test_int8_path_is_causal_in_prefill(model):
    changed = IDS.clone()
    changed[0, -1] = 99
    with torch.no_grad():
        a = model(IDS, past_key_values=QuantizedKVCache(), use_cache=True).logits
        b = model(changed, past_key_values=QuantizedKVCache(), use_cache=True).logits
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)
    assert not torch.allclose(a[:, -1], b[:, -1], atol=1e-5)


def test_int8_path_respects_padding_mask(model):
    ids = torch.cat([IDS, IDS.flip(-1)])
    padded = ids.clone()
    padded[1, :3] = 0
    mask = torch.ones_like(padded)
    mask[1, :3] = 0
    pos = (mask.cumsum(-1) - 1).clamp_min(0)
    other = padded.clone()
    other[1, :3] = 7
    kwargs = dict(attention_mask=mask, position_ids=pos, use_cache=True)
    with torch.no_grad():
        a = model(padded, past_key_values=QuantizedKVCache(), **kwargs).logits
        b = model(other, past_key_values=QuantizedKVCache(), **kwargs).logits
    assert torch.allclose(a[1, 3:], b[1, 3:], atol=1e-5)


@pytest.mark.parametrize("mode", ["dequant", "integer"])
def test_generate_runs_and_grows_cache(model, mode):
    cache = QuantizedKVCache(mode=mode)
    with torch.no_grad():
        out = model.generate(IDS, past_key_values=cache, max_new_tokens=5, do_sample=False, pad_token_id=0)
    assert out.shape[1] == 17
    assert cache.get_seq_length() == 16
    assert cache.get_kv(0)[0].dtype == torch.int8


def test_recorder_splits_prefill_and_decode(model):
    torch.manual_seed(2)
    config = model.config
    m = Qwen2ForCausalLM(config).eval()
    rec = OperatorRecorder()
    patch_qwen2(m, rec)
    cache = QuantizedKVCache(keep_float=True)
    with torch.no_grad():
        m.generate(IDS, past_key_values=cache, max_new_tokens=3, do_sample=False, pad_token_id=0)
    s = rec.summary()
    assert s["prefill"]["samples"] == 2
    assert s["decode"]["samples"] == 4
    assert s["decode"]["output_rel_l2"] < 0.2


def test_recorder_without_float_copy_fails_clearly():
    torch.manual_seed(3)
    config = Qwen2Config(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=1,
        intermediate_size=128, vocab_size=100, max_position_embeddings=128, attn_implementation="sdpa",
    )
    m = Qwen2ForCausalLM(config).eval()
    patch_qwen2(m, OperatorRecorder())
    with pytest.raises(ValueError, match="keep_float"), torch.no_grad():
        m(IDS, past_key_values=QuantizedKVCache(), use_cache=True)


def test_patch_rejects_non_qwen2_model():
    with pytest.raises(ValueError, match="Qwen2"):
        patch_qwen2(torch.nn.Linear(2, 2))
