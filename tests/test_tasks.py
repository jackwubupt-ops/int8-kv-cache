import json

from int8kv.tasks import extract_code, make_task, normalize_int, run_humaneval

HE = {
    "task_id": "t/0",
    "prompt": "def add(a, b):\n    \"\"\"Return a + b.\"\"\"\n",
    "entry_point": "add",
    "test": "def check(f):\n    assert f(1, 2) == 3\n    assert f(-1, 1) == 0\n",
}


def write(tmp_path, rows):
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return str(p)


def test_humaneval_passes_correct_code_from_block():
    reply = "Sure:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
    assert run_humaneval(HE, extract_code(HE, reply))


def test_humaneval_body_only_gets_prompt_prepended():
    code = extract_code(HE, "    return a + b\n")
    assert code.startswith("def add") and run_humaneval(HE, code)


def test_humaneval_fails_wrong_code_and_timeouts():
    assert not run_humaneval(HE, "def add(a, b):\n    return a - b\n")
    assert not run_humaneval(HE, "def add(a, b):\n    while True: pass\n", timeout=1)


def test_humaneval_task_scores_from_file(tmp_path):
    task = make_task("humaneval", write(tmp_path, [HE]))
    assert "def add" in task.prompt(task.samples[0])
    assert task.score(task.samples[0], "```python\ndef add(a, b):\n    return b + a\n```")


def test_aime_scoring_and_answer_normalization(tmp_path):
    task = make_task("aime", write(tmp_path, [{"problem": "Find x.", "answer": "073"}, {"problem": "y", "answer": 5}]), 2)
    assert task.score(task.samples[0], "... so \\boxed{73}")
    assert not task.score(task.samples[0], "\\boxed{74}")
    assert task.score(task.samples[1], "\\boxed{5}")
    assert normalize_int(None) is None and normalize_int("1,000") == 1000


def test_gsm8k_from_file(tmp_path):
    task = make_task("gsm8k", write(tmp_path, [{"question": "q", "answer": "x #### 12"}]))
    assert task.score(task.samples[0], "\\boxed{12}")
