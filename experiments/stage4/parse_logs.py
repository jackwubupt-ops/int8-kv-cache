"""Parse SGLang server logs and bench_serving results from scripts/stage4/bench.sh into one summary.

Layout written by bench.sh under results/stage4/<model>/bench/:
  <config>/server.log                      whole server log of that configuration
  <config>/serving_<in>x<out>_run<k>.jsonl bench_serving output, one file per run
  <config>/long_<in>x<out>_run<k>.jsonl    long-request concurrency test
  <config>/long_<in>x<out>_server.log      the server log lines written during that test
  decode_kernel_<dtype>.json               bench_decode_kernel.py output

  python experiments/stage4/parse_logs.py results/stage4/Qwen2.5-7B-Instruct/bench
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List

BASELINE = "bf16"
_BENCH_FIELDS = ("output_throughput", "median_ttft_ms", "median_tpot_ms", "request_throughput", "completed",
                 "duration", "total_input_tokens", "total_output_tokens", "max_concurrent_requests")
_MEDIAN_FIELDS = ("output_throughput", "median_ttft_ms", "median_tpot_ms")


def parse_max_total_tokens(server_log: str) -> int:
    """KV pool capacity in tokens, from the scheduler's "max_total_num_tokens=N, ..." line."""
    m = re.search(r"max_total_num_tokens=(\d+)", server_log)
    if not m:
        raise ValueError("no max_total_num_tokens= line in the server log (did the server start?)")
    return int(m.group(1))


def parse_peak_running_reqs(server_log: str) -> int:
    """Largest "#running-req: N" over the prefill and decode batch lines (0 if there are none)."""
    return max((int(n) for n in re.findall(r"#running-req[:=]\s*(\d+)", server_log)), default=0)


def parse_server_log(server_log: str) -> Dict[str, object]:
    def first(pattern, cast):
        m = re.search(pattern, server_log)
        return cast(m.group(1)) if m else None

    kv = re.search(r"KV Cache is allocated\. #tokens: \d+, K size: ([\d.]+) GB, V size: ([\d.]+) GB", server_log)
    return {
        "max_total_num_tokens": parse_max_total_tokens(server_log),
        "kv_cache_gb": float(kv.group(1)) + float(kv.group(2)) if kv else None,
        "chunked_prefill_size": first(r"chunked_prefill_size=(-?\d+)", int),
        "max_running_requests": first(r"max_running_requests=(\d+)", int),
        "peak_running_reqs": parse_peak_running_reqs(server_log),
        "peak_gen_throughput": max((float(x) for x in re.findall(r"gen throughput \(token/s\): ([\d.]+)", server_log)),
                                   default=None),
    }


def parse_bench_serving(output_json: Path) -> Dict[str, object]:
    """Key numbers of the last run in a bench_serving JSONL file (it appends one line per run)."""
    lines = [line for line in Path(output_json).read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{output_json} is empty: bench_serving did not finish a run")
    data = json.loads(lines[-1])
    return {key: data.get(key) for key in _BENCH_FIELDS}


def _shape_runs(config_dir: Path, prefix: str) -> Dict[str, List[Path]]:
    """{"1024x256": [run files]} for files named <prefix>_<in>x<out>_run<k>.jsonl."""
    shapes: Dict[str, List[Path]] = {}
    for path in sorted(config_dir.glob(f"{prefix}_*_run*.jsonl")):
        m = re.fullmatch(rf"{prefix}_(\d+x\d+)_run\d+\.jsonl", path.name)
        if m:
            shapes.setdefault(m.group(1), []).append(path)
    return shapes


def _median_over_runs(paths: List[Path]) -> Dict[str, object]:
    runs = [parse_bench_serving(p) for p in paths]
    out: Dict[str, object] = {"runs": len(runs)}
    for key in _MEDIAN_FIELDS:
        out[key] = statistics.median(r[key] for r in runs)
    return out


def summarize_bench(directory: Path) -> Dict[str, object]:
    directory = Path(directory)
    configs: Dict[str, dict] = {}
    for config_dir in sorted(p for p in directory.iterdir() if p.is_dir() and (p / "server.log").exists()):
        try:
            entry = parse_server_log((config_dir / "server.log").read_text(errors="replace"))
        except ValueError as e:  # the server never came up; keep the other configurations
            configs[config_dir.name] = {"error": str(e)}
            continue
        entry["serving"] = {shape: _median_over_runs(paths)
                            for shape, paths in _shape_runs(config_dir, "serving").items()}
        entry["long"] = {}
        for shape, paths in _shape_runs(config_dir, "long").items():
            long = _median_over_runs(paths)
            long["completed"] = min(parse_bench_serving(p)["completed"] for p in paths)
            segment = config_dir / f"long_{shape}_server.log"
            long["peak_running_reqs"] = (parse_peak_running_reqs(segment.read_text(errors="replace"))
                                         if segment.exists() else None)
            entry["long"][shape] = long
        configs[config_dir.name] = entry
    base = configs.get(BASELINE, {}).get("max_total_num_tokens")
    for entry in configs.values():
        if "error" not in entry:
            entry["capacity_vs_bf16"] = entry["max_total_num_tokens"] / base if base else None

    kernel: Dict[str, Dict[str, float]] = {}
    for path in sorted(directory.glob("decode_kernel_*.json")):
        data = json.loads(path.read_text())
        kernel[data["dtype"]] = {str(r["context"]): r["median_ms"] for r in data["results"]}
    summary = {"configs": configs, "decode_kernel": kernel}
    (directory / "bench_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__)
        return 2
    summary = summarize_bench(Path(args[0]))
    for name, c in summary["configs"].items():
        if "error" in c:
            print(f"BENCH {name:16s} FAILED: {c['error']}", flush=True)
            continue
        ratio = f"{c['capacity_vs_bf16']:.3f}" if c["capacity_vs_bf16"] else "n/a"
        print(f"BENCH {name:16s} tokens={c['max_total_num_tokens']} ({ratio}x bf16) kv={c['kv_cache_gb']} GB", flush=True)
        for shape, s in c["serving"].items():
            print(f"BENCH {name:16s} serving {shape:9s} out tok/s={s['output_throughput']:.1f} "
                  f"ttft={s['median_ttft_ms']:.0f} ms tpot={s['median_tpot_ms']:.1f} ms (runs={s['runs']})", flush=True)
        for shape, s in c["long"].items():
            print(f"BENCH {name:16s} long    {shape:9s} peak running={s['peak_running_reqs']} "
                  f"out tok/s={s['output_throughput']:.1f} completed={s['completed']}", flush=True)
    for dtype, rows in summary["decode_kernel"].items():
        print(f"BENCH decode kernel {dtype:10s} " + " ".join(f"{ctx}:{ms:.3f}ms" for ctx, ms in rows.items()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
