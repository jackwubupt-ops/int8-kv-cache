# Int8 KV Cache 设计（修订版）

## 目标
在 Qwen2.5 上实现动态 INT8 KV Cache，并接入 SGLang；用 vLLM 的 `INT8_PER_TOKEN_HEAD` 做对照，用实测数据说明省下多少显存、损失多少精度。

## 环境分工
| 角色 | 负责内容 |
|---|---|
| Mac | 写源码、跑单元测试和 CPU 冒烟，推送到 GitHub |
| Windows 笔记本（RTX 3070 Ti Laptop） | 阶段 0-3 的精度实验，原生 Python，不需要 WSL2 或 SGLang |
| 租用 Linux GPU 服务器（24GB+） | 阶段 4-5：SGLang 接入、Qwen2.5-7B 最终对比和性能数据 |

笔记本 GPU 功耗受限只影响运行速度，不影响精度结论；**阶段 0-3 不产出任何性能数字。**

## 范围
阶段 0-3（本次实现）：
- `int8kv`：纯 PyTorch 的量化函数、Attention、INT8 Cache、指标。
- 分布分析、算子级对比、模型级对比、GSM8K 与长上下文任务评估。

阶段 4-5（租服务器后实现）：SGLang fork（`torch_native` 路径）、三方对比、可选 Triton Kernel。

不做：PD 分离、MLA、稀疏 Attention、Speculative Decoding、FP4、多节点。

## 量化方案
- 张量布局：`[batch, heads, tokens, head_dim]`，与 Transformers 和 PyTorch Attention 一致。
- 粒度：per_tensor / per_head / per_token / per_token_head；默认 per_token_head，由阶段 1 决定是否调整。
- `scale = absmax / 127`；全零时 scale 取 1；输入含 NaN/Inf 时报错。
- `q = clamp(round(x / scale), -127, 127)`。
- scale 可存为 BF16，转换时向上取整，避免最大值被裁剪。
- 模式：`dequant`（还原后计算）与 `integer`（Q 也量化，内积只用整数值）。

## 整数模式计算
```text
logit[j] = s_q * s_k[j] * (Q_int · K_int[j]) / sqrt(d)
weight   = softmax(logit)
output   = Σ weight[j] * s_v[j] * V_int[j]
```
`head_dim=128` 时内积绝对值上限 2,064,512，小于 2^24，FP32 可精确表示，因此可以用 FP32 matmul，兼容 GPU。

## 模型接入
- `QuantizedKVCache(DynamicCache)`：写入时量化，保存 INT8 和 scale。
- patch `Qwen2Attention.forward`：只有收到 `QuantizedKVCache` 才走 INT8 路径，否则调用原实现。同一个模型实例可以同时产出基线和 INT8 结果，省显存。

## 评估
| 层级 | 指标 |
|---|---|
| 数值 | K/V 重建误差、logit 绝对误差、注意力 KL、输出相对 L2 和余弦相似度 |
| 模型 | 下一 token logits KL、Top-1 一致率、首次分叉位置 |
| 任务 | GSM8K 准确率、长上下文事实检索准确率 |

规则：先测基线噪声底；精度实验使用贪心解码；5%-10% 相对误差只作为算子层初期参考；精度不达标也如实报告。

## 模型
Qwen2.5-0.5B 冒烟，Qwen2.5-1.5B 跑阶段 0-3（`head_dim=128`，与 7B 相同）；Qwen2.5-7B 在服务器上做最终实验。

## 里程碑
- M1 分布分析：有分布图，确定量化粒度。
- M2 量化 Attention：单元测试通过，算子误差有报告。
- M3 模型端到端：1.5B 有基线与 INT8 的精度结果。
- M4 SGLang 接入：服务器上 7B 三方对比。
- M5 Kernel（选做）。
