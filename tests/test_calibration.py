import pytest
import torch

from int8kv.cache import QuantizedKVCache
from int8kv.calibration import ChannelStats, load_static_scales


def test_absmax_and_exact_quantile_over_several_writes():
    s = ChannelStats(top_k=16)
    x = torch.arange(1, 1001, dtype=torch.float32)  # 1000 tokens, one head, one channel
    for chunk in x.split(300):
        t = chunk.view(1, 1, -1, 1)
        s.add(0, -t, t)
    assert s.absmax(0, "k").item() == 1000
    assert s.tokens[0] == 1000
    # 99% quantile: 10 values above it -> 990
    assert s.quantile(0, "v", 0.99).item() == 990


def test_quantile_beyond_kept_values_is_rejected():
    s = ChannelStats(top_k=4)
    t = torch.randn(1, 1, 1000, 2)
    s.add(0, t, t)
    with pytest.raises(ValueError, match="top values"):
        s.quantile(0, "k", 0.99)


def test_save_and_load_static_scales(tmp_path):
    s = ChannelStats()
    k = torch.randn(1, 2, 50, 8)
    k[..., 1] *= 20
    s.add(0, k, k)
    s.add(1, k, k * 2)
    s.save(tmp_path / "scales.pt", quantile=0.99)
    scales = load_static_scales(tmp_path / "scales.pt", "per_channel", use_quantile=False)
    assert set(scales) == {0, 1}
    s_k, s_v = scales[1]
    assert s_k.shape == (1, 2, 1, 8)
    assert torch.allclose(s_v, s_k * 2)
    assert torch.allclose(s_k[0, :, 0], k.abs().amax(dim=(0, 2)) / 127)
    head = load_static_scales(tmp_path / "scales.pt", "per_head", use_quantile=True)
    assert head[0][0].shape == (1, 2, 1, 1)


def test_channel_spread_flags_outlier_channel():
    s = ChannelStats()
    k = torch.randn(1, 2, 200, 16)
    k[..., 3] *= 50
    v = torch.randn(1, 2, 200, 16)
    s.add(0, k, v)
    spread = s.channel_spread()
    assert spread["k"]["within_head_max_over_median_p50"] > 10 * spread["v"]["within_head_max_over_median_p50"]


def test_static_cache_stores_one_scale_per_layer():
    s_k = torch.full((1, 2, 1, 8), 0.1)
    c = QuantizedKVCache(granularity="per_channel", scale_source="static", static_scales={0: (s_k, s_k)})
    k = torch.randn(1, 2, 5, 8)
    c.update(k, k, 0)
    c.update(k[:, :, :1], k[:, :, :1], 0)
    ki, sk, vi, sv = c.get_kv(0)
    assert ki.dtype == torch.int8 and ki.shape == (1, 2, 6, 8)
    assert sk.shape == (1, 2, 1, 8)
    assert c.memory_bytes()["scale_bytes"] == 2 * 16 * 2


def test_dynamic_cache_requantizes_over_all_tokens():
    c = QuantizedKVCache(granularity="per_channel", scale_source="dynamic")
    k = torch.randn(1, 2, 5, 8)
    c.update(k, k, 0)
    big = k[:, :, :1] * 10
    c.update(big, big, 0)
    ki, sk, _, _ = c.get_kv(0)
    assert ki.shape == (1, 2, 6, 8)
    # the new larger token sets the scale for every earlier token too
    assert torch.allclose(sk.float(), torch.cat([k, big], -2).abs().amax(-2, keepdim=True) / 127, rtol=0.01)


@pytest.mark.parametrize("kwargs, match", [
    ({"granularity": "per_channel"}, "per_channel"),
    ({"granularity": "per_token_head", "scale_source": "static", "static_scales": {0: (1, 1)}}, "needs one of"),
    ({"granularity": "per_channel", "scale_source": "static"}, "static_scales"),
    ({"scale_source": "later"}, "scale_source"),
])
def test_bad_cache_configs_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        QuantizedKVCache(**kwargs)
