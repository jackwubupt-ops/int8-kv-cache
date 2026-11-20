"""MHATokenToKVPoolINT8 and the int8_per_token_head CLI/dtype wiring (GPU server only)."""
import pytest
import torch

from tests_stage4.conftest import needs_gpu

pytestmark = needs_gpu

SIZE, HEADS, DIM, LAYERS = 10, 4, 128, 2


@pytest.fixture
def pool():
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolINT8

    return MHATokenToKVPoolINT8(
        SIZE, page_size=1, dtype=torch.int8, head_num=HEADS, head_dim=DIM, layer_num=LAYERS,
        device="cuda", enable_memory_saver=False, compute_dtype=torch.bfloat16,
    )


def _kv(n, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(n, HEADS, DIM, device="cuda", generator=g)
    return (x * torch.tensor([1.0, 50.0, 300.0, 0.01], device="cuda").view(1, HEADS, 1)).to(torch.bfloat16)


def test_pool_buffer_shapes(pool):
    assert pool.k_buffer[0].shape == (SIZE + 1, HEADS, DIM) and pool.k_buffer[0].dtype == torch.int8
    k_scale, v_scale = pool.get_kv_scale_buffer(1)
    assert k_scale.shape == (SIZE + 1, HEADS) and v_scale.dtype == torch.bfloat16


def test_cell_size_matches_allocated_bytes(pool):
    from sglang.srt.layers.quantization.int8_kv import int8_kv_cell_size

    k_bytes, v_bytes = pool.get_kv_size_bytes()
    per_token = int8_kv_cell_size(head_num=HEADS, head_dim=DIM, v_head_dim=DIM, num_layers=LAYERS)
    assert k_bytes + v_bytes == per_token * (SIZE + 1)


def test_set_then_get_round_trip(pool):
    k, v = _kv(2, 0), _kv(2, 1)
    loc = torch.tensor([3, 7], device="cuda")
    pool.set_kv_buffer(None, loc, k, v, layer_id_override=1)
    k_scale, _ = pool.get_kv_scale_buffer(1)
    back = pool.get_key_buffer(1)[loc].float()
    assert pool.get_key_buffer(1).dtype == torch.bfloat16
    bound = k_scale[loc].float().unsqueeze(-1) / 2 + k.float().abs() * 2**-8 + 1e-6
    assert ((back - k.float()).abs() <= bound).all()
    assert torch.count_nonzero(pool.k_buffer[0]) == 0  # other layer untouched


def test_static_scales_rejected(pool):
    k = _kv(1, 2)
    with pytest.raises(ValueError, match="k_scale"):
        pool.set_kv_buffer(None, torch.tensor([1], device="cuda"), k, k, k_scale=1.0, layer_id_override=0)


def test_cli_accepts_int8_choice():
    import argparse

    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(["--model-path", "x", "--kv-cache-dtype", "int8_per_token_head"])
    assert args.kv_cache_dtype == "int8_per_token_head"
