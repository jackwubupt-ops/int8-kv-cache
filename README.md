# Int8 KV Cache

在 Qwen2.5 上实现并评估动态 INT8 KV Cache 量化，最终接入 SGLang，并与 vLLM 的 INT8 实现对比。

## 当前进度
- 阶段 0-3 的代码和实验脚本已完成，等待在 Windows GPU 上运行。
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
| `scripts/` | Windows 安装与运行脚本 |
| `docs/windows-runbook.md` | Windows 运行手册 |
| `docs/stage0-framework-survey.md` | vLLM / SGLang 现状调研 |

## 开发
```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```
