"""Per-channel / per-head INT8 KV wired through the patched Qwen2 model (tiny random model)."""
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from int8kv.cache import QuantizedKVCache
from int8kv.calibration import ChannelStats
from int8kv.metrics import kl_from_logits
from int8kv.qwen2_patch import OperatorRecorder, patch_qwen2
from int8kv.quant import calibrate_scale
from int8kv.runtime import parse_cache_configs

IDS = torch.arange(1, 13).unsqueeze(0)


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


@pytest.fixture(scope="module")
def static_scales(model):
    from transformers import DynamicCache
    stats = ChannelStats()
    cache = DynamicCache()
    with torch.no_grad():
        model(torch.arange(1, 40).unsqueeze(0) % 100, past_key_values=cache, use_cache=True)
    for i, layer in enumerate(cache.layers):
        stats.add(i, layer.keys, layer.values)
    return {g: {i: (calibrate_scale(stats.absmax(i, "k"), g), calibrate_scale(stats.absmax(i, "v"), g))
                for i in stats.layers()} for g in ("per_head", "per_channel")}


def make_cache(static_scales, gran, mode, source, **kw):
    scales = static_scales[gran] if source == "static" else None
    return QuantizedKVCache(granularity=gran, mode=mode, scale_source=source, static_scales=scales, **kw)


CONFIGS = [(g, m, s) for g in ("per_head", "per_channel") for m in ("dequant", "int8") for s in ("dynamic", "static")]


@pytest.mark.parametrize("gran, mode, source", CONFIGS)
def test_prefill_logits_close_to_baseline(model, static_scales, gran, mode, source):
    with torch.no_grad():
        base = model(IDS).logits
        out = model(IDS, past_key_values=make_cache(static_scales, gran, mode, source), use_cache=True).logits
    assert kl_from_logits(base, out).item() < 1e-2


@pytest.mark.parametrize("gran, mode, source", CONFIGS)
def test_generate_with_float_prefill(model, static_scales, gran, mode, source):
    cache = make_cache(static_scales, gran, mode, source, float_prefill=True)
    with torch.no_grad():
        out = model.generate(IDS, past_key_values=cache, max_new_tokens=5, do_sample=False, pad_token_id=0)
    assert out.shape[1] == 17 and cache.get_seq_length() == 16
    ki, sk, _, _ = cache.get_kv(0)
    assert ki.dtype == torch.int8
    assert sk.shape[-2] == 1  # one scale for all tokens
    assert sk.shape[-1] == (16 if gran == "per_channel" else 1)


def test_recorder_reports_actual_output_error_for_a_and_b():
    torch.manual_seed(2)
    config = Qwen2Config(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2,
        intermediate_size=128, vocab_size=100, max_position_embeddings=128, attn_implementation="sdpa",
    )
    m = Qwen2ForCausalLM(config).eval()
    rec = OperatorRecorder()
    patch_qwen2(m, rec)
    errors = {}
    for mode in ("dequant", "int8"):
        rec.rows.clear()
        cache = QuantizedKVCache(granularity="per_channel", mode=mode, scale_source="dynamic")
        with torch.no_grad():
            m.generate(IDS, past_key_values=cache, max_new_tokens=3, do_sample=False, pad_token_id=0)
        errors[mode] = rec.summary()["decode"]["actual_output_rel_l2"]
    assert 0 < errors["dequant"] < errors["int8"] < 0.1


def test_parse_cache_configs():
    got = parse_cache_configs("per_channel:int8:static, per_token_head:dequant")
    assert [c["label"] for c in got] == ["per_channel:int8:static", "per_token_head:dequant:write"]
    with pytest.raises(ValueError, match="granularity"):
        parse_cache_configs("per_channel")
