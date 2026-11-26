"""Download the stage 5 datasets once and save them as JSONL under data/.

On AutoDL the Hugging Face hub is not reachable directly; run with HF_ENDPOINT=https://hf-mirror.com
(run_autodl.sh sets it). Re-running skips files that already exist.
"""
import json
import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "gsm8k/test.jsonl": ("openai/gsm8k", "main", "test"),
    "gsm8k/train.jsonl": ("openai/gsm8k", "main", "train"),
    "aime/aime_2022_2024.jsonl": ("AI-MO/aimo-validation-aime", None, "train"),
    "humaneval/test.jsonl": ("openai/openai_humaneval", None, "test"),
}
EXPECTED = {"gsm8k/test.jsonl": 1319, "aime/aime_2022_2024.jsonl": 90, "humaneval/test.jsonl": 164}


def main():
    for rel, (repo, config, split) in SOURCES.items():
        path = ROOT / "data" / rel
        if path.exists():
            print(f"{rel}: exists, {sum(1 for _ in path.open())} rows")
            continue
        rows = load_dataset(repo, config, split=split)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        print(f"{rel}: {len(rows)} rows from {repo}")
        if rel in EXPECTED and len(rows) != EXPECTED[rel]:
            sys.exit(f"{rel}: expected {EXPECTED[rel]} rows, got {len(rows)}")


if __name__ == "__main__":
    main()
