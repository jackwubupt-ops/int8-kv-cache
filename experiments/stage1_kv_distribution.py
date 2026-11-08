"""Stage 1: capture KV caches and analyse their distribution."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import DynamicCache

from int8kv.analysis import (
    ABSMAX_BINS, GranularityStats, absmax_histogram, crest_factor, recommend_granularity, write_chunks,
)
from int8kv.prompts import SHORT_PROMPTS, make_needle_prompt
from int8kv.quant import GRANULARITIES
from int8kv.runtime import encode_chat, environment_info, greedy_generate, load_model, pick_device, save_json

COLOR, GRID, INK = "#2F6DB5", "#E3E6EA", "#3A3F47"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK)


def plot_absmax(hists, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, constrained_layout=True)
    centers = np.sqrt(ABSMAX_BINS[:-1] * ABSMAX_BINS[1:])
    widths = np.diff(ABSMAX_BINS) * 0.85
    for row, kind in enumerate("KV"):
        for col, phase in enumerate(("prefill", "decode")):
            ax = axes[row][col]
            ax.bar(centers, hists[kind][phase], width=widths, color=COLOR)
            ax.set_xscale("log")
            _style(ax, f"{kind} / {phase}", "per (token, head) absmax" if row else "", "count" if not col else "")
    fig.suptitle("KV absmax distribution", x=0.01, ha="left", fontsize=13, color=INK)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_granularity(tables, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True, constrained_layout=True)
    for ax, kind in zip(axes, "KV"):
        values = [tables[kind][g]["rel_l2"] * 100 for g in GRANULARITIES]
        bars = ax.bar(GRANULARITIES, values, width=0.5, color=COLOR)
        ax.bar_label(bars, fmt="%.2f%%", color=INK, fontsize=9)
        _style(ax, f"{kind} reconstruction error", "", "relative L2 (%)" if kind == "K" else "")
    fig.suptitle("INT8 error by granularity", x=0.01, ha="left", fontsize=13, color=INK)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--needle-facts", type=int, default=150)
    parser.add_argument("--needles", type=int, default=3)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/stage1"))
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    model, tokenizer = load_model(args.model, device, args.dtype)
    prompts = SHORT_PROMPTS[: args.max_prompts] + [
        make_needle_prompt(args.needle_facts, seed)["prompt"] for seed in range(args.needles)
    ]

    stats = {kind: GranularityStats() for kind in "KV"}
    hists = {kind: {p: np.zeros(len(ABSMAX_BINS) - 1, dtype=np.int64) for p in ("prefill", "decode")} for kind in "KV"}
    crest = {kind: [] for kind in "KV"}
    layer_absmax = {kind: {} for kind in "KV"}

    for index, text in enumerate(prompts):
        ids = encode_chat(tokenizer, text, device)
        cache = DynamicCache()
        greedy_generate(model, ids, args.max_new_tokens, cache)
        prompt_len = ids.shape[1]
        for layer_idx, layer in enumerate(cache.layers):
            for kind, tensor in (("K", layer.keys), ("V", layer.values)):
                x = tensor.cpu().to(torch.float32)
                for chunk in write_chunks(x, prompt_len):
                    if chunk.shape[-2]:
                        stats[kind].add_chunk(chunk)
                hists[kind]["prefill"] += absmax_histogram(x[..., :prompt_len, :])
                if x.shape[-2] > prompt_len:
                    hists[kind]["decode"] += absmax_histogram(x[..., prompt_len:, :])
                crest[kind].append(crest_factor(x).ravel())
                layer_absmax[kind].setdefault(layer_idx, []).append(x.abs().amax(-1).ravel())
        print(f"prompt {index + 1}/{len(prompts)}: {prompt_len} tokens", flush=True)

    tables = {kind: stats[kind].table() for kind in "KV"}
    summary = {"environment": environment_info(device, args.model, args.dtype)}
    for kind in "KV":
        all_crest = torch.cat(crest[kind])
        per_layer = {
            str(i): {
                "absmax_p50": torch.cat(v).quantile(0.5).item(),
                "absmax_max": torch.cat(v).max().item(),
            }
            for i, v in sorted(layer_absmax[kind].items())
        }
        summary[kind] = {
            "granularity": tables[kind],
            "recommended_granularity": recommend_granularity(tables[kind]),
            "crest_factor_p50": all_crest.quantile(0.5).item(),
            "crest_factor_p99": all_crest.quantile(0.99).item(),
            "per_layer": per_layer,
        }

    args.out.mkdir(parents=True, exist_ok=True)
    save_json(args.out / "kv_stats.json", summary)
    plot_absmax(hists, args.out / "absmax_hist.png")
    plot_granularity(tables, args.out / "granularity_error.png")
    print("recommended:", summary["K"]["recommended_granularity"], "(K),", summary["V"]["recommended_granularity"], "(V)")
    return summary


if __name__ == "__main__":
    main()
