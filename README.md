# INT8 KV Cache for SGLang

在 SGLang（v0.5.12）的 Triton 后端里实现 **per-token-head 动态 INT8 KV Cache**，从 PyTorch 原型一路做到融合 Triton kernel 和服务端压测。

- 每个 (token, head) 一个 bf16 scale，`scale = absmax / 127`，向上取整；K/V 存 int8
- 写入、decode、extend 三处都是融合 Triton kernel，在寄存器里反量化，不生成整层的 bf16 副本
- 启动参数：`--kv-cache-dtype int8_per_token_head --attention-backend triton`

## 结果（Qwen2.5-7B-Instruct，RTX 4090 D，`--mem-fraction-static 0.85`）

| | BF16 | INT8 融合版 |
|---|---|---|
| KV 容量（token） | 99,750 | **196,433（1.969×，理论值 1.969）** |
| 128 个 8192+512 长请求的峰值同时运行数 | 11 | **23** |
| 服务输出吞吐（1024×256 / 4096×512 / 8192×512） | 1285.6 / 502.3 / 252.7 tok/s | **1575.2 / 618.5 / 306.9（+22.5% / +23.1% / +21.4%）** |
| decode attention，batch 32，32k 上下文 | 2.377 ms | **1.380 ms** |
| needle 检索（约 3300 / 1.3 万 token，各 100 题） | 95 / 95 | **96 / 96** |
| GSM8K（1319 题） | 90.30% | 89.16%（McNemar p = 0.020） |

精度：融合版在 GSM8K 上没有达到设计时定的标准（p ≥ 0.05）。作为对照，只把 BF16 的后端从 Triton 换成 FlashInfer（两者都无损）就有 67 道题对错改变（p = 0.46）；融合版与简单版之间（p = 0.12）、与 FlashInfer 版 BF16 之间（p = 0.42）都不显著。分析见报告 5.3 节。

SGLang 自带的 FP8 KV 在这个模型上 GSM8K 约 2%：K 的离群通道使浮点格式误差过大，不是溢出（报告第 7 节）。

## 文档

| 文档 | 内容 |
|---|---|
| [阶段 4 报告](docs/stage4-report.md) | SGLang 实现、正确性、精度、容量与性能、FP8 分析 |
| [阶段 0-3 报告](docs/stage0-3-report.md) | PyTorch 原型：KV 分布、算子误差、模型级评估（Qwen2.5-1.5B） |
| [决策记录](docs/stage4-decision-log.md) | 每一步的决定和理由 |
| [工程底稿](docs/stage4-engineering-draft.md) | 服务器上的原始输出 |
| [框架调研](docs/stage0-framework-survey.md) | vLLM / SGLang 的 KV 量化现状 |

## 结构

| 路径 | 内容 |
|---|---|
| `sglang_patch/` | 对 SGLang v0.5.12 的 8 个补丁，`apply.sh` 应用，`overlay.py` 覆盖到已安装的 wheel |
| `scripts/stage4/` | GPU 服务器脚本：`setup.sh` 安装，`run_all.sh` 跑完全部评测 |
| `experiments/stage4/` | 精度评测、压测日志解析、decode kernel 微基准、FP8 探针与格式模拟、报告表格生成 |
| `tests_stage4/` | SGLang 侧测试（kernel 与 PyTorch 参考对比；纯 PyTorch 部分在 Mac 上也能跑） |
| `int8kv/`、`experiments/stage*.py` | 阶段 0-3 的 PyTorch 原型 |
| `results/` | 全部结果文件，报告中的数字都来自这里 |

## 复现

```bash
# GPU 服务器（Linux，CUDA 12.8+，24 GB 显存）
bash scripts/stage4/setup.sh
nohup bash scripts/stage4/run_all.sh > run_all.log 2>&1 < /dev/null &

# 本机测试
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

详细步骤见报告第 10 节和 [服务器运行手册](docs/server-runbook.md)。
