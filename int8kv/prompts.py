"""Deterministic prompts for experiments."""
from __future__ import annotations

import random
import re
from typing import Optional

SHORT_PROMPTS = [
    "Explain how a hash table handles collisions.",
    "Why does the moon always show the same face to Earth?",
    "Write a Python function that reverses a linked list.",
    "Summarize the causes of the 1929 stock market crash in three sentences.",
    "What is the difference between TCP and UDP?",
    "Give three tips for writing clear technical documentation.",
    "Translate into French: The train leaves at seven tomorrow morning.",
    "A farmer has 17 sheep and buys 8 more, then sells 5. How many are left?",
    "用三句话解释什么是注意力机制。",
    "北京和上海的气候有什么主要区别？",
    "写一个判断素数的 Python 函数，并说明时间复杂度。",
    "请解释为什么 GPU 适合做深度学习计算。",
]

_WORDS = ["harbor", "cobalt", "meadow", "lantern", "falcon", "granite", "willow", "ember", "quartz", "summit"]


def make_needle_prompt(n_facts: int, seed: int) -> dict:
    """Long-context retrieval: one record among n_facts holds the answer."""
    if n_facts < 1:
        raise ValueError("n_facts must be at least 1")
    rng = random.Random(seed)
    records = []
    for i in range(n_facts):
        name = f"{rng.choice(_WORDS)}-{rng.choice(_WORDS)}-{i:04d}"
        records.append((name, str(rng.randint(100000, 999999))))
    name, answer = records[rng.randrange(n_facts)]
    body = "\n".join(f"The access code for {n} is {code}." for n, code in records)
    question = f"What is the access code for {name}? Reply with only the code."
    return {"prompt": f"{body}\n\n{question}", "answer": answer}


def needle_correct(text: str, answer: str) -> bool:
    return answer in text


def extract_number(text: str) -> Optional[str]:
    """Answer in the last \\boxed{}, else after the last ####, else the last number in the text."""
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if boxed:
        tail, first = boxed[-1], True
    elif "####" in text:
        tail, first = text.rsplit("####", 1)[-1], True
    else:
        tail, first = text, False
    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", tail)
    if not matches:
        return None
    value = matches[0] if first else matches[-1]
    value = value.replace(",", "")
    return value[:-2] if value.endswith(".0") else value.rstrip(".")
