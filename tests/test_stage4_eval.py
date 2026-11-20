"""Stage 4 accuracy evaluation (experiments/stage4/eval_accuracy.py) with a fake Engine and tokenizer."""
import importlib.util
import json
from pathlib import Path

import pytest

from int8kv.metrics import mcnemar_exact

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "stage4" / "eval_accuracy.py"
_spec = importlib.util.spec_from_file_location("eval_accuracy", _PATH)
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)


class FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return "<user>" + messages[0]["content"] + "<assistant>"

    def __call__(self, text, add_special_tokens):
        class Out:
            input_ids = [ord(c) % 1000 for c in text[:50]]
        return Out()


class FakeEngine:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate(self, input_ids, sampling_params):
        self.calls.append((input_ids, sampling_params))
        return [{"text": self.reply(i)} for i in range(len(input_ids))]


def _gsm_file(tmp_path, answers):
    path = tmp_path / "gsm.jsonl"
    rows = [{"question": f"Question {i}?", "answer": f"Work...\n#### {a}"} for i, a in enumerate(answers)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_run_eval_scores_with_fake_engine(tmp_path):
    engine = FakeEngine(lambda i: "So the answer is \\boxed{18}")
    out = E.run_eval(engine, FakeTokenizer(), "gsm8k", 2, 1024, data_file=_gsm_file(tmp_path, [18, 7]))
    assert out["accuracy"] == 0.5 and out["correct"] == 1 and out["total"] == 2
    assert [p["correct"] for p in out["predictions"]] == [True, False]
    assert out["predictions"][0] == {"id": 0, "pred": "18", "gold": "18", "correct": True}
    assert engine.calls[0][1] == {"max_new_tokens": 1024, "temperature": 0}


def test_run_eval_uses_the_shared_gsm8k_instruction(tmp_path):
    from int8kv.prompts import GSM8K_INSTRUCTION

    assert E.build_prompt("gsm8k", {"question": "Q?"}) == "Q?" + GSM8K_INSTRUCTION


def test_run_eval_takes_the_first_n_questions(tmp_path):
    engine = FakeEngine(lambda i: "\\boxed{1}")
    out = E.run_eval(engine, FakeTokenizer(), "gsm8k", 3, 16, data_file=_gsm_file(tmp_path, [1, 2, 3, 4, 5]))
    assert out["total"] == 3 and len(engine.calls[0][0]) == 3


def test_needle_tasks_score_by_code_in_text():
    from int8kv.prompts import make_needle_prompt

    answers = [make_needle_prompt(150, seed)["answer"] for seed in range(3)]
    engine = FakeEngine(lambda i: f"The code is {answers[i]}." if i != 1 else "I do not know")
    out = E.run_eval(engine, FakeTokenizer(), "needle150", 3, 64)
    assert [p["correct"] for p in out["predictions"]] == [True, False, True]
    assert out["predictions"][0]["gold"] == answers[0]


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError, match="task"):
        E.run_eval(FakeEngine(lambda i: ""), FakeTokenizer(), "mmlu", 1, 8)


@pytest.mark.parametrize("kv,impl,name", [("auto", "plain", "bf16"), ("fp8_e4m3", "plain", "fp8_e4m3"),
                                          ("fp8_e5m2", "plain", "fp8_e5m2"),
                                          ("int8_per_token_head", "plain", "int8_plain"),
                                          ("int8_per_token_head", "fused", "int8_fused")])
def test_config_names(kv, impl, name):
    assert E.config_name(kv, impl) == name


def _write_preds(d, task, config, correct):
    preds = [{"id": i, "pred": "x", "gold": "y", "correct": c} for i, c in enumerate(correct)]
    (d / f"{task}_{config}.json").write_text(json.dumps(
        {"task": task, "config": config, "accuracy": sum(correct) / len(correct), "correct": sum(correct),
         "total": len(correct), "predictions": preds}))


def test_summarize_uses_paired_counts(tmp_path):
    _write_preds(tmp_path, "gsm8k", "bf16", [True, True, True, False, False, True])
    _write_preds(tmp_path, "gsm8k", "int8_fused", [True, False, False, False, True, False])
    _write_preds(tmp_path, "needle150", "bf16", [True, True])
    s = E.summarize_accuracy(tmp_path)
    row = s["gsm8k"]["int8_fused"]
    assert row["only_baseline"] == 3 and row["only_quantized"] == 1
    assert row["mcnemar_p"] == pytest.approx(mcnemar_exact(3, 1)) == pytest.approx(0.625)
    assert row["accuracy"] == pytest.approx(2 / 6) and row["baseline_accuracy"] == pytest.approx(4 / 6)
    assert s["needle150"] == {}
    assert json.loads((tmp_path / "accuracy_summary.json").read_text()) == s


def test_summarize_rejects_mismatched_question_sets(tmp_path):
    _write_preds(tmp_path, "gsm8k", "bf16", [True, True])
    _write_preds(tmp_path, "gsm8k", "fp8_e4m3", [True, True, False])
    with pytest.raises(ValueError, match="paired"):
        E.summarize_accuracy(tmp_path)


def test_cli_accepts_fp8_e5m2():
    assert "fp8_e5m2" in E.KV_DTYPES


@pytest.mark.parametrize("kv,impl,backend,name", [
    ("fp8_e4m3", "plain", "flashinfer", "fp8_e4m3_flashinfer"),
    ("auto", "plain", "flashinfer", "bf16_flashinfer"),
    ("int8_per_token_head", "fused", "triton", "int8_fused"),
])
def test_config_names_carry_a_non_triton_backend(kv, impl, backend, name):
    assert E.config_name(kv, impl, backend) == name


def test_summarize_pairs_only_with_triton_bf16(tmp_path):
    _write_preds(tmp_path, "gsm8k", "bf16", [True, True, False])
    _write_preds(tmp_path, "gsm8k", "bf16_flashinfer", [True, False, False])
    s = E.summarize_accuracy(tmp_path)
    assert s["gsm8k"]["bf16_flashinfer"]["only_baseline"] == 1
