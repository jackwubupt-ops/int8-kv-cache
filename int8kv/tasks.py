"""Benchmark tasks for the accuracy comparison: GSM8K, AIME and HumanEval.

Each task gives a list of samples, a chat prompt per sample and a scorer for the generated text.
Datasets come from the Hugging Face hub unless a local JSONL file is given.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from .prompts import GSM8K_INSTRUCTION, extract_number

HF_SOURCES = {
    "gsm8k": ("openai/gsm8k", "main", "test"),
    "aime": ("AI-MO/aimo-validation-aime", None, "train"),  # 90 problems, AIME 2022-2024
    "humaneval": ("openai/openai_humaneval", None, "test"),  # 164 problems
}
DEFAULT_MAX_NEW_TOKENS = {"gsm8k": 512, "aime": 4096, "humaneval": 768}
AIME_INSTRUCTION = "\n\nPlease reason step by step, and put your final answer (an integer from 0 to 999) within \\boxed{}."
HUMANEVAL_INSTRUCTION = (
    "Complete the following Python function. Reply with the complete function "
    "(including the signature and any imports) in a single ```python code block.\n\n"
)
EXEC_TIMEOUT_S = 10


@dataclass
class Task:
    name: str
    samples: List[dict]
    prompt: Callable[[dict], str]
    score: Callable[[dict, str], bool]
    extract: Callable[[dict, str], Optional[str]]


def read_jsonl(path) -> List[dict]:
    """One JSON object per line. Splits on "\\n" only: str.splitlines() also breaks on U+2028 and
    similar characters, which ensure_ascii=False leaves unescaped inside the JSON strings."""
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").split("\n") if l.strip()]


def load_rows(name: str, data_file: Optional[str]) -> List[dict]:
    if data_file:
        return read_jsonl(data_file)
    from datasets import load_dataset

    repo, config, split = HF_SOURCES[name]
    return [dict(r) for r in load_dataset(repo, config, split=split)]


def normalize_int(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    try:
        return int(float(str(text).replace(",", "").strip()))
    except ValueError:
        return None


def extract_code(sample: dict, text: str) -> str:
    """The last ```python block (or the whole reply); prepend the prompt if the function is missing."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    code = blocks[-1] if blocks else text
    if f"def {sample['entry_point']}" not in code:
        code = sample["prompt"] + code
    return code


def run_humaneval(sample: dict, code: str, timeout: float = EXEC_TIMEOUT_S) -> bool:
    """Run the generated function against the task's tests in an isolated subprocess.

    The code is model output, so it runs with `python -I` in an empty temporary directory, with a
    timeout. This is the standard HumanEval check; it is not a security sandbox.
    """
    program = f"{code}\n\n{sample['test']}\n\ncheck({sample['entry_point']})\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "solution.py"
        path.write_text(program, encoding="utf-8")
        try:
            done = subprocess.run([sys.executable, "-I", str(path)], cwd=tmp, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
    return done.returncode == 0


def make_task(name: str, data_file: Optional[str] = None, count: Optional[int] = None) -> Task:
    rows = load_rows(name, data_file)
    rows = rows[: count] if count else rows
    if name == "gsm8k":
        return Task(
            name, rows,
            prompt=lambda s: s["question"] + GSM8K_INSTRUCTION,
            extract=lambda s, t: extract_number(t),
            score=lambda s, t: extract_number(t) == extract_number(s["answer"]),
        )
    if name == "aime":
        return Task(
            name, rows,
            prompt=lambda s: s["problem"] + AIME_INSTRUCTION,
            extract=lambda s, t: extract_number(t),
            score=lambda s, t: normalize_int(extract_number(t)) == normalize_int(s["answer"]),
        )
    if name == "humaneval":
        return Task(
            name, rows,
            prompt=lambda s: HUMANEVAL_INSTRUCTION + s["prompt"],
            extract=extract_code,
            score=lambda s, t: run_humaneval(s, extract_code(s, t)),
        )
    raise ValueError(f"unknown task {name!r}, expected one of {sorted(HF_SOURCES)}")
