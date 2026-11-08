import numpy as np
import pytest
import torch

from int8kv.analysis import GranularityStats, absmax_histogram, crest_factor, recommend_granularity, write_chunks
from int8kv.prompts import extract_number, make_needle_prompt, needle_correct
from int8kv.runtime import check_model_name, parse_configs, pick_device


def test_needle_prompt_is_deterministic_and_contains_answer():
    a, b = make_needle_prompt(50, 7), make_needle_prompt(50, 7)
    assert a == b and a["answer"] in a["prompt"]
    assert a["prompt"].count("access code for") == 51


def test_needle_correct_requires_answer_text():
    assert needle_correct("The code is 123456.", "123456")
    assert not needle_correct("I do not know", "123456")


@pytest.mark.parametrize("text,expected", [
    ("so #### 1,234", "1234"), ("answer is 42.", "42"), ("first 3 then 5", "5"),
    ("#### -7", "-7"), ("#### 18.0", "18"), ("no digits", None),
    ("so 2 + 1 = 3, answer \\(\\boxed{3}\\).", "3"), ("\\boxed{70,000} not 5", "70000"),
    ("\\boxed{\\$18}", "18"), ("try \\boxed{1} then \\boxed{2}", "2"),
])
def test_extract_number(text, expected):
    assert extract_number(text) == expected


def test_write_chunks_splits_prefill_then_single_tokens():
    x = torch.zeros(1, 2, 6, 4)
    assert [c.shape[-2] for c in write_chunks(x, 4)] == [4, 1, 1]


def test_finer_granularity_has_lower_error_on_uneven_tokens():
    x = torch.randn(1, 2, 20, 16)
    x[:, :, :5] *= 100
    stats = GranularityStats()
    stats.add_chunk(x)
    t = stats.table()
    assert t["per_token_head"]["rel_l2"] < t["per_tensor"]["rel_l2"]
    assert t["per_tensor"]["scale_bytes_per_token_head"] == pytest.approx(2 / 40)
    assert t["per_token_head"]["scale_bytes_per_token_head"] == 2


def test_recommend_prefers_cheaper_when_error_is_similar():
    table = {
        "per_tensor": {"rel_l2": 0.0105, "scale_bytes_per_token_head": 0.1},
        "per_head": {"rel_l2": 0.0104, "scale_bytes_per_token_head": 0.2},
        "per_token": {"rel_l2": 0.0102, "scale_bytes_per_token_head": 1.0},
        "per_token_head": {"rel_l2": 0.0100, "scale_bytes_per_token_head": 2.0},
    }
    assert recommend_granularity(table) == "per_tensor"
    table["per_tensor"]["rel_l2"] = table["per_head"]["rel_l2"] = 0.05
    assert recommend_granularity(table) == "per_token"


def test_histogram_counts_every_token_head_even_outliers():
    x = torch.zeros(1, 2, 3, 4)
    x[0, 0, 0, 0] = 1e9
    assert absmax_histogram(x).sum() == 6


def test_crest_factor_of_constant_vector_is_one():
    assert crest_factor(torch.ones(1, 1, 1, 8)).item() == pytest.approx(1.0)


def test_parse_configs():
    assert parse_configs("per_head:dequant, per_token:integer") == [("per_head", "dequant"), ("per_token", "integer")]
    with pytest.raises(ValueError, match="granularity:mode"):
        parse_configs("per_head")


def test_cuda_request_without_gpu_fails_with_setup_hint(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="setup_windows"):
        pick_device("cuda")
    assert pick_device("auto") == "cpu"


@pytest.mark.parametrize("name", ["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-0.5B"])
def test_base_qwen_models_are_rejected_for_chat_prompts(name):
    with pytest.raises(ValueError, match="Instruct"):
        check_model_name(name)


@pytest.mark.parametrize("name", ["Qwen/Qwen2.5-1.5B-Instruct", "/tmp/tiny-model"])
def test_instruct_and_local_models_are_accepted(name):
    check_model_name(name)
