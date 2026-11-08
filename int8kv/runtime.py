"""Model loading, greedy generation helpers and result files."""
from __future__ import annotations

import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def pick_device(requested: str = "auto") -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. "
            "Run scripts/setup_windows.ps1 to install the CUDA build of PyTorch and check nvidia-smi."
        )
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


_BASE_QWEN = re.compile(r"^Qwen/Qwen[\d.]*-[\d.]+B$", re.IGNORECASE)


def check_model_name(name: str) -> None:
    """Experiments wrap prompts in the chat template, which base models answer with garbage."""
    if _BASE_QWEN.match(name):
        raise ValueError(
            f"{name} is a base model; the experiments use chat prompts, so use {name}-Instruct instead."
        )


def load_model(name: str, device: str, dtype: str = "bf16"):
    check_model_name(name)
    if dtype not in _DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}, expected one of {sorted(_DTYPES)}")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=_DTYPES[dtype], attn_implementation="sdpa")
    return model.to(device).eval(), tokenizer


def encode_chat(tokenizer, text: str, device: str) -> torch.Tensor:
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    return tokenizer(prompt, return_tensors="pt").input_ids.to(device)


@torch.no_grad()
def greedy_generate(model, ids: torch.Tensor, max_new_tokens: int, cache) -> List[int]:
    """Greedy decode with sampling and repetition penalties explicitly off."""
    pad = model.config.pad_token_id if model.config.pad_token_id is not None else model.config.eos_token_id
    out = model.generate(
        ids,
        attention_mask=torch.ones_like(ids),
        past_key_values=cache,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=1.0,
        pad_token_id=pad,
    )
    return out[0, ids.shape[1] :].tolist()


@torch.no_grad()
def teacher_forced_logits(model, ids: torch.Tensor, tokens: List[int], cache) -> torch.Tensor:
    """Next-token logits through the cache path while feeding a fixed token sequence."""
    logits = [model(ids, past_key_values=cache, use_cache=True).logits[:, -1].cpu()]
    for token in tokens[:-1]:
        step = torch.tensor([[token]], device=ids.device)
        logits.append(model(step, past_key_values=cache, use_cache=True).logits[:, -1].cpu())
    return torch.cat(logits).to(torch.float32)


def plain_cache_bytes(cache: DynamicCache) -> int:
    return sum(
        t.numel() * t.element_size() for layer in cache.layers for t in (layer.keys, layer.values)
    )


def parse_configs(text: str) -> List[Tuple[str, str]]:
    """Parse "per_token_head:integer,per_head:dequant"."""
    configs = []
    for item in text.split(","):
        parts = item.strip().split(":")
        if len(parts) != 2:
            raise ValueError(f"bad config {item!r}, expected granularity:mode")
        configs.append((parts[0], parts[1]))
    return configs


def environment_info(device: str, model_name: str, dtype: str) -> Dict[str, Optional[str]]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "model": model_name,
        "dtype": dtype,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": commit,
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def save_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
