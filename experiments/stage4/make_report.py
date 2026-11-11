"""Turn the stage 4 result files of one model into Markdown tables for the report.

  python experiments/stage4/make_report.py results/stage4/Qwen2.5-7B-Instruct

Reads accuracy/ (eval.sh), bench/ (bench.sh), fp8_probe.json and format_sim.json (fp8_probe.sh), and writes
report_tables.md next to them. Numbers are copied from the result files, never typed by hand; sections whose
inputs are missing say so. `checks()` lists what to look at before trusting the tables.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_accuracy import BASELINE, TASKS, summarize_accuracy  # noqa: E402
from parse_logs import summarize_bench  # noqa: E402

EXPECTED_CONFIGS = ("bf16", "fp8_e4m3", "fp8_e5m2", "int8_plain", "int8_fused")
HEAD_DIM = 128
THEORY_RATIO = 4 * HEAD_DIM / (2 * HEAD_DIM + 4)  # bf16 bytes / int8 bytes per (token, head): 512 / 260
RATIO_TOLERANCE = 0.03
SIM_FORMATS = ("fp8_e4m3", "fp8_e5m2", "fp8_e4m3_calib", "int8_per_token_head")


def _table(header: List[str], rows: List[List[object]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _load(path: Path) -> Optional[dict]:
    return json.loads(path.read_text()) if path.exists() else None


def _accuracy_runs(acc_dir: Path) -> Dict[str, Dict[str, dict]]:
    runs: Dict[str, Dict[str, dict]] = {}
    if acc_dir.is_dir():
        for path in sorted(acc_dir.glob("*.json")):
            if path.name != "accuracy_summary.json":
                data = json.loads(path.read_text())
                runs.setdefault(data["task"], {})[data["config"]] = data
    return runs


def _config_order(configs) -> List[str]:
    known = [c for c in EXPECTED_CONFIGS if c in configs]
    return known + sorted(c for c in configs if c not in EXPECTED_CONFIGS)


def accuracy_section(model_dir: Path) -> str:
    acc_dir = model_dir / "accuracy"
    runs = _accuracy_runs(acc_dir)
    if not runs:
        return "## 精度（SGLang 实测）\n\n没有精度结果（`accuracy/` 为空）。"
    summary = summarize_accuracy(acc_dir)
    parts = ["## 精度（SGLang 实测）"]
    for task in [t for t in TASKS if t in runs] + sorted(t for t in runs if t not in TASKS):
        rows = []
        for config in _config_order(runs[task]):
            run = runs[task][config]
            acc = f"{run['accuracy']:.4f} ({run['correct']}/{run['total']})"
            paired = summary.get(task, {}).get(config)
            if config == BASELINE or paired is None:
                rows.append([config, acc, "—", "—", "—", "—"])
            else:
                rows.append([config, acc, f"{paired['delta']:+.4f}", paired["only_baseline"],
                             paired["only_quantized"], f"{paired['mcnemar_p']:.3g}"])
        chunks = sorted({str(r.get("chunked_prefill_size")) for r in runs[task].values()})
        parts.append(f"### {task}\n\nchunked_prefill_size = {', '.join(chunks)}\n\n" + _table(
            ["配置", "准确率", "相对 BF16", "仅 BF16 对", "仅该配置对", "McNemar p"], rows))
    return "\n\n".join(parts)


def bench_section(model_dir: Path) -> str:
    bench_dir = model_dir / "bench"
    if not bench_dir.is_dir():
        return "## 容量和性能\n\n没有性能结果（`bench/` 不存在）。"
    summary = summarize_bench(bench_dir)
    configs = {name: c for name, c in summary["configs"].items() if "error" not in c}
    failed = [name for name, c in summary["configs"].items() if "error" in c]
    order = _config_order(configs)
    parts = ["## 容量和性能"]
    parts.append(f"### KV 容量\n\nINT8 相对 BF16 的理论容量比：{THEORY_RATIO:.3f}（head_dim {HEAD_DIM}）。\n\n" + _table(
        ["配置", "容量 (token)", "相对 BF16", "KV 显存 (GB)"],
        [[n, configs[n]["max_total_num_tokens"],
          f"{configs[n]['capacity_vs_bf16']:.3f}" if configs[n]["capacity_vs_bf16"] else "—",
          f"{configs[n]['kv_cache_gb']:.2f}" if configs[n]["kv_cache_gb"] is not None else "—"] for n in order]))
    rows = [[n, shape, f"{s['output_throughput']:.1f}", f"{s['median_ttft_ms']:.0f}", f"{s['median_tpot_ms']:.1f}",
             s["runs"]] for n in order for shape, s in configs[n]["serving"].items()]
    parts.append("### 吞吐（bench_serving，random-ids，请求一次性发出）\n\n" + _table(
        ["配置", "输入x输出", "输出 token/s", "TTFT 中位 (ms)", "TPOT 中位 (ms)", "次数"], rows))
    rows = [[n, shape, s["peak_running_reqs"], f"{s['output_throughput']:.1f}", s["completed"]]
            for n in order for shape, s in configs[n]["long"].items()]
    parts.append("### 长请求并发\n\n" + _table(["配置", "输入x输出", "峰值并发", "输出 token/s", "完成数"], rows))
    kernel = summary["decode_kernel"]
    if kernel:
        contexts = sorted({ctx for rows in kernel.values() for ctx in rows}, key=int)
        parts.append("### decode kernel 微基准（中位耗时 ms）\n\n" + _table(
            ["dtype"] + contexts,
            [[dtype] + [f"{kernel[dtype][c]:.3f}" if c in kernel[dtype] else "—" for c in contexts]
             for dtype in sorted(kernel)]))
    if failed:
        parts.append("启动失败的配置：" + ", ".join(failed))
    return "\n\n".join(parts)


def fp8_section(model_dir: Path) -> str:
    probe, sim = _load(model_dir / "fp8_probe.json"), _load(model_dir / "format_sim.json")
    parts = ["## FP8 分析（纯 PyTorch，不经过 SGLang）"]
    if probe is None:
        parts.append("没有 FP8 探针结果（`fp8_probe.json` 不存在）。")
    else:
        layers = probe["layers"]
        n = len(layers)
        for kind, label in (("k", "K"), ("v", "V")):
            rows_k = [layer[kind] for layer in layers]
            formats = [f for f in (*SIM_FORMATS,) if f in rows_k[0]]
            overflow = sum(r["e4m3_overflow_frac"] > 0 for r in rows_k)
            parts.append(
                f"### 探针：{label}\n\n{label} 的 absmax 最大 {max(r['absmax'] for r in rows_k):.1f}；"
                f"e4m3 溢出（|x| > 448）的层：{overflow}/{n}；"
                f"通道 absmax 的最大值/中位数最高 {max(r['channel_absmax_max_over_median'] for r in rows_k):.1f}。\n\n"
                + _table(["格式", "NaN 比例（均值）", "相对 L2（均值）", "去均值相对 L2（均值）", "去均值相对 L2（最差层）"],
                         [[f, f"{sum(r[f]['nan_frac'] for r in rows_k) / n:.2e}",
                           f"{sum(r[f]['rel_l2'] for r in rows_k) / n:.4f}",
                           f"{sum(r[f]['centered_rel_l2'] for r in rows_k) / n:.4f}",
                           f"{max(r[f]['centered_rel_l2'] for r in rows_k):.4f}"] for f in formats]))
    if sim is None:
        parts.append("没有格式模拟结果（`format_sim.json` 不存在）。")
    else:
        base = sim.get("baseline") or {}
        gsm_base = sim.get("gsm8k_baseline")
        rows = []
        for fmt in [f for f in SIM_FORMATS if f in sim["formats"]]:
            e = sim["formats"][fmt]
            gsm = e.get("gsm8k")
            rows.append([fmt, f"{e['top1_per_prompt_mean']:.4f}",
                         f"{e['kl_mean_finite']:.4f}" if e["kl_mean_finite"] is not None else "—",
                         f"{e['nan_logit_prompts']}/{e['prompts']}", f"{e['exact_generation_rate']:.2f}",
                         f"{e['needle_correct']}/{e['needle_total']}",
                         f"{gsm['accuracy']:.4f} (p={gsm['mcnemar_p']:.3g})" if gsm else "—"])
        note = f"浮点基线：needle {base.get('needle_correct')}/{base.get('needle_total')}"
        if gsm_base:
            note += f"，GSM8K {gsm_base['accuracy']:.4f}（{gsm_base['total']} 题）"
        parts.append("### 格式模拟（模型层面）\n\n" + note + "。\n\n" + _table(
            ["格式", "top1/prompt", "KL", "NaN 提示词", "生成完全一致", "needle", "GSM8K"], rows))
    return "\n\n".join(parts)


def checks(model_dir: Path) -> List[str]:
    """What to verify before trusting the tables: ✓ fine, ⚠ anomaly, ○ result not present."""
    out: List[str] = []
    runs = _accuracy_runs(model_dir / "accuracy")
    for task in TASKS:
        if task not in runs:
            out.append(f"○ 缺少 {task} 的全部结果")
            continue
        missing = [c for c in EXPECTED_CONFIGS if c not in runs[task]]
        if missing:
            out.append(f"○ {task} 缺少配置：{', '.join(missing)}")
        totals = {r["total"] for r in runs[task].values()}
        out.append(f"✓ {task} 各配置题数一致（{totals.pop()}）" if len(totals) == 1
                   else f"⚠ {task} 各配置题数不一致：{sorted(totals)}")
        chunks = {str(r.get("chunked_prefill_size")) for r in runs[task].values()}
        if len(chunks) > 1:
            out.append(f"⚠ {task} 各配置的 chunked_prefill_size 不一致：{sorted(chunks)}")
    bench_dir = model_dir / "bench"
    if bench_dir.is_dir():
        for name, c in summarize_bench(bench_dir)["configs"].items():
            if "error" in c:
                out.append(f"⚠ {name} 的 server 没有启动成功")
            elif name.startswith("int8") and c["capacity_vs_bf16"]:
                ratio = c["capacity_vs_bf16"]
                ok = abs(ratio - THEORY_RATIO) / THEORY_RATIO <= RATIO_TOLERANCE
                out.append(f"{'✓' if ok else '⚠'} {name} 容量比 {ratio:.3f}，理论值 {THEORY_RATIO:.3f}")
    else:
        out.append("○ 缺少性能结果")
    for name in ("fp8_probe.json", "format_sim.json"):
        if not (model_dir / name).exists():
            out.append(f"○ 缺少 {name}")
    return out


def build_report(model_dir: Path) -> str:
    model_dir = Path(model_dir)
    commit = (model_dir.parent / "COMMIT")
    head = [f"# 阶段 4 结果表：{model_dir.name}", "",
            "由 `experiments/stage4/make_report.py` 从结果文件生成，不要手工修改。"]
    if commit.exists():
        head.append(f"\n服务器上的代码版本：`{commit.read_text().strip()[:12]}`")
    body = [accuracy_section(model_dir), bench_section(model_dir), fp8_section(model_dir),
            "## 核对\n\n" + "\n".join(f"- {line}" for line in checks(model_dir))]
    return "\n".join(head) + "\n\n" + "\n\n".join(body) + "\n"


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__)
        return 2
    model_dir = Path(args[0])
    text = build_report(model_dir)
    (model_dir / "report_tables.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
