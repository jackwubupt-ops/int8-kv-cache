# Int8 KV Cache

在 Qwen2.5 上实现并评估动态 INT8 KV Cache 量化，最终接入 SGLang，并与 vLLM 的 INT8 实现对比。

## 当前进度
- 阶段 0-3 已完成，结果见 [阶段 0-3 分析报告](docs/stage0-3-report.md)：Qwen2.5-1.5B-Instruct 上 INT8 per_token_head 把 KV 显存降到 50.7%，长上下文检索 79/100（基线 80/100）。
- 阶段 4-5（SGLang 接入、Kernel）尚未开始。

本仓库目前**没有正式实验结果**，`results/` 中的数据由实验脚本生成后提交。

## 结构
| 路径 | 内容 |
|---|---|
| `int8kv/quant.py` | 量化、scale 计算与取整 |
| `int8kv/attention.py` | 基线与 INT8 Attention |
| `int8kv/cache.py` | INT8 KV Cache（基于 DynamicCache） |
| `int8kv/qwen2_patch.py` | 把 Qwen2 Attention 接到 INT8 路径 |
| `int8kv/metrics.py`、`analysis.py` | 精度指标与分布分析 |
| `experiments/` | 阶段 1-3 实验脚本 |
| `scripts/` | Linux 服务器与 Windows 的安装、运行脚本 |
| `docs/server-runbook.md` | Linux 服务器运行手册 |
| `docs/windows-runbook.md` | Windows 运行手册 |
| `docs/stage0-framework-survey.md` | vLLM / SGLang 现状调研 |

## 开发
```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```
