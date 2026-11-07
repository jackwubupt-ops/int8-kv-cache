"""Run each experiment script end to end on a tiny random model saved locally."""
import importlib.util
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

ROOT = Path(__file__).resolve().parent.parent


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny")
    config = Qwen2Config(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2,
        intermediate_size=128, vocab_size=200, max_position_embeddings=2048,
    )
    Qwen2ForCausalLM(config).save_pretrained(path)
    inner = Tokenizer(WordLevel({"<unk>": 0, "the": 1, "code": 2}, unk_token="<unk>"))
    inner.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=inner, unk_token="<unk>", pad_token="<unk>")
    tok.chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}"
    tok.save_pretrained(path)
    return str(path)


COMMON = ["--device", "cpu", "--dtype", "fp32"]


def test_stage1_writes_stats_and_plots(tiny_model, tmp_path):
    load_script("stage1_kv_distribution").main(
        ["--model", tiny_model, "--out", str(tmp_path), "--max-new-tokens", "3", "--max-prompts", "2",
         "--needles", "1", "--needle-facts", "3", *COMMON]
    )
    data = json.loads((tmp_path / "kv_stats.json").read_text(encoding="utf-8"))
    assert data["K"]["recommended_granularity"] in data["K"]["granularity"]
    assert data["environment"]["model"] == tiny_model
    assert (tmp_path / "absmax_hist.png").stat().st_size > 0
    assert (tmp_path / "granularity_error.png").stat().st_size > 0


def test_stage2_reports_each_config_and_phase(tiny_model, tmp_path):
    load_script("stage2_operator").main(
        ["--model", tiny_model, "--out", str(tmp_path), "--max-new-tokens", "3", "--max-prompts", "1",
         "--needle-facts", "2", "--configs", "per_token_head:dequant,per_head:integer", *COMMON]
    )
    data = json.loads((tmp_path / "operator_metrics.json").read_text(encoding="utf-8"))
    assert set(data["results"]) == {"per_token_head:dequant", "per_head:integer"}
    assert {"prefill", "decode"} <= set(data["results"]["per_head:integer"])


def test_stage3_noise_floor_is_zero_and_memory_is_lower(tiny_model, tmp_path):
    load_script("stage3_model").main(
        ["--model", tiny_model, "--out", str(tmp_path), "--max-new-tokens", "4", "--max-prompts", "1",
         "--needles", "1", "--needle-facts", "2", "--configs", "per_token_head:integer", *COMMON]
    )
    data = json.loads((tmp_path / "model_metrics.json").read_text(encoding="utf-8"))["results"]
    floor = data["baseline_repeat_noise_floor"]
    assert floor["logits_kl"] == 0 and floor["exact_generation_rate"] == 1
    assert data["per_token_head:integer"]["kv_memory_ratio_vs_baseline"] < 1


def test_gsm8k_runs_from_local_file(tiny_model, tmp_path):
    data_file = tmp_path / "g.jsonl"
    data_file.write_text(json.dumps({"question": "1+1?", "answer": "#### 2"}) + "\n", encoding="utf-8")
    load_script("stage3_gsm8k").main(
        ["--model", tiny_model, "--out", str(tmp_path), "--data-file", str(data_file), "--samples", "1",
         "--max-new-tokens", "3", "--configs", "per_token_head:dequant", *COMMON]
    )
    data = json.loads((tmp_path / "gsm8k.json").read_text(encoding="utf-8"))
    assert data["samples"] == 1 and "baseline" in data["results"]
