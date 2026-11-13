# 阶段 4 报告：在 SGLang 中实现 per-token-head 动态 INT8 KV Cache

> **状态：草稿（2026-11-13）。** 正确性、7B 冒烟评测、FP8 分析已有数据。7B 上的正式精度、容量和吞吐结果还没有跑，对应位置标为 ⏳，跑完 `scripts/stage4/run_all.sh` 后由 `results/stage4/Qwen2.5-7B-Instruct/report_tables.md` 填入。本文所有数字都来自仓库里的结果文件或记录在 [工程底稿](stage4-engineering-draft.md) 中的服务器输出，没有估计值。

- 基线：SGLang v0.5.12（commit `127b9e32`），补丁 6 个，见 `sglang_patch/`
- 硬件：AutoDL RTX 4090 D（24 GB），驱动 570，PyTorch 2.11.0+cu128
- 模型：正确性检查用 Qwen2.5-1.5B-Instruct，评测用 Qwen2.5-7B-Instruct
- 前置：[阶段 0-3 报告](stage0-3-report.md)、[设计文档](superpowers/specs/2026-11-08-stage4-sglang-int8-kv-design.md)
- 过程记录：[工程底稿](stage4-engineering-draft.md)、[决策记录](stage4-decision-log.md)

## 1. 结论

1. **INT8 KV 已经接进 SGLang 的 Triton 后端，并通过了 kernel 级和模型级的正确性检查。** `--kv-cache-dtype int8_per_token_head` 在写入、decode、extend 三处都用融合的 Triton kernel，直接读写 int8 和每个 (token, head) 的 scale。
2. **在 SGLang 里，INT8 的模型行为和阶段 3 的 PyTorch 参考实现一致。** 同一批 22 个提示词上，逐 token 的 top-1 一致率是 0.9888 到 0.9902，阶段 3 是 0.9879。
3. **融合 kernel 是必要的。** 7B 冒烟评测里，融合版 20 道 GSM8K 用 11 秒，和 BF16 的 10 秒相当；不融合、每步把整个缓存池反量化的简单版要 89 秒。
4. **FP8 KV 在 Qwen 上不可用，原因是格式，不是实现。** 1.5B 上 top-1 掉到 0.86，7B 上输出乱码。纯 PyTorch 的模拟复现了同样的退化；Qwen 的 K 里有幅值是中位通道 78 倍的离群通道，浮点格式在大数值上的绝对误差太大。
5. ⏳ **容量、吞吐和完整 GSM8K 的结论待正式运行。** 理论容量比是 1.969。

## 2. 成功标准逐项

| 维度 | 标准（设计文档第 1 节） | 目前的证据 | 状态 |
|---|---|---|---|
| 正确性 | kernel 级和模型级检查全部通过 | 第 4 节 | ✅ |
| 容量 | INT8 的最大 KV token 数 ≥ BF16 的 1.9 倍 | 容量估算与实际分配的字节数一致（L0 测试）；实测容量 ⏳ | ⏳ |
| 性能 | 上下文 ≥ 4k 时融合 decode kernel 的延迟不高于 BF16；KV 容量成为瓶颈时整体吞吐高于 BF16 | 冒烟：GSM8K 20 题 11 秒对 10 秒；kernel 微基准和压测 ⏳ | ⏳ |
| 精度 | 完整 GSM8K 上 INT8 对 BF16 的 McNemar p ≥ 0.05 | 冒烟（20 题）：19/20 对 19/20；完整 1319 题 ⏳ | ⏳ |

## 3. 实现

### 3.1 量化方案

- 粒度：每个 (token, head) 一行，`scale = absmax / 127`，全 0 行的 scale 取 1。
- scale 用 bf16 存，并且只向上取整，保证这一行的最大值量化后不超过 127。
- `q = clamp(round(x / scale), -127, 127)`，舍入方式为 round-half-even。
- 每个 token 每层的字节数：`head_num × (2 × head_dim + 4)`。head_dim 为 128 时是 BF16 的 260/512 = 50.8%。
- prefill 语义：本次新输入的 token 用原始精度的 K/V，缓存里的前缀读 INT8。这和 SGLang 原有的 extend kernel 一致，也是阶段 3 验证过的做法。

### 3.2 对 SGLang 的改动

10 个文件，新增 559 行，删除 12 行。

| 文件 | 改动 |
|---|---|
| `srt/layers/quantization/int8_kv.py`（新增） | 量化数学的 PyTorch 参考实现、容量公式、不支持组合的检查。只依赖 torch，Mac 上可测 |
| `srt/layers/attention/triton_ops/int8_kv_store.py`（新增） | 融合写入 kernel：一个 program 处理一个 (token, head)，在 kernel 内算 scale、量化、写入 |
| `srt/layers/attention/triton_ops/decode_attention.py` | grouped decode kernel 增加 INT8 分支：`qk = dot(q, k) × s_k`，乘 V 之前 `p × s_v` |
| `srt/layers/attention/triton_ops/extend_attention.py` | extend kernel 的前缀循环增加 INT8 分支；本次新输入的部分不变 |
| `srt/mem_cache/memory_pool.py` | `MHATokenToKVPoolINT8`：int8 的 K/V 加两组 bf16 scale |
| `srt/layers/attention/triton_backend.py` | 融合模式下把 int8 buffer 和 scale 传给 kernel |
| `server_args.py`、`model_runner.py`、`pool_configurator.py`、`model_runner_kv_cache_mixin.py` | 命令行选项、dtype 解析、按真实字节数规划容量、启动时检查 |

- 环境变量 `SGLANG_INT8_KV_IMPL` 选择实现：`fused`（默认）用融合 kernel；`plain` 用 PyTorch 写入，读取时把整层反量化成 bf16，只用来对照。
- 不支持的组合在启动时直接报错，不静默回退：非 Triton 后端、MLA、推测解码、hicache、sliding window、TP 大于 1、page_size 不为 1、融合模式下的确定性推理。
- 不传 scale 时，decode 和 extend kernel 在 BF16 和 FP8 下的输出与未打补丁的版本逐比特相同（L0 测试保证）。

### 3.3 一个实现细节

Triton 的 fp32 除法默认是近似的，直接用会让 kernel 算出的 scale 和 PyTorch 差最后一位。写入 kernel 用 `div_rn`（IEEE 除法）和 `rint`（round-half-even），所以 scale 与参考实现逐比特一致，量化值也完全相同。

## 4. 正确性

### 4.1 Kernel 级（L0，服务器上运行）

`tests_stage4/` 共 93 项。

| 对象 | 检查 |
|---|---|
| 写入 kernel | scale 与 PyTorch 参考逐比特相等（N 取 1、7、4096，bf16 和 fp16 输入）；非连续输入；只写指定槽位；全 0 行；空 batch |
| decode kernel | 与 FP32 参考的相对 L2 < 1e-2（batch 1 和 4，上下文 1 到 4097，GQA 分组 6 和 7）；与简单版的余弦相似度 > 0.9999 |
| extend kernel | 前缀长度 {0, 5, 300} × 新输入长度 {1, 64, 257} 对 FP32 参考；没有前缀时与 BF16 kernel 逐比特相同 |
| 不影响原有路径 | 不传 scale 时，BF16 和 FP8 都与未打补丁的 kernel 逐比特相同 |
| 后端接线（1.5B，`sglang.Engine`） | 融合版能生成，前 8 个 token 与简单版相同；命中前缀缓存前后输出相同；512 分块时，简单版检索到的 needle 融合版也检索到 |

最近一次完整运行（`45dd78e`）是 92 项通过、1 项失败；失败的是分块用例，它的判定方式有问题（见 6.1），改写后后端接线的 3 项单独运行通过（`0055369`）。改写之后还没有把 93 项作为一个整体重跑过，`run_all.sh` 的第一步会做。

### 4.2 模型级（L1，Qwen2.5-1.5B-Instruct，22 个提示词）

做法：先让 BF16 贪心生成 64 个 token 作为参考，然后每个配置逐 token 强制解码，比较每一步的 top-1 是否和参考相同。prefill 不分块，和阶段 3 的测法一致。数据来自服务器运行 `20261110_001555`（`45dd78e`）。

| 配置 | top-1 一致率（按提示词平均） | \|Δlogp\| | needle 部分 |
|---|---|---|---|
| BF16，CUDA Graph 开（参考自身） | 1.0000 | 0.0000 | 1.0000 |
| BF16，CUDA Graph 关（对照） | 0.9986 | 0.0034 | 1.0000 |
| INT8 简单版，Graph 开 | 0.9888 | 0.0208 | 1.0000 |
| INT8 简单版，Graph 关 | 0.9895 | 0.0208 | 1.0000 |
| INT8 融合版，Graph 开 | 0.9902 | 0.0199 | 1.0000 |
| INT8 融合版，Graph 关 | 0.9895 | 0.0201 | 1.0000 |
| FP8 e4m3（仅参考） | 0.8644 | 0.3427 | 0.8571 |

- 判定标准是不低于阶段 3 在同一批提示词上的 0.9879 减去 0.005，四个 INT8 配置都通过。
- 融合版和简单版的差距小于 0.0015。融合版的 |Δlogp| 略小，因为简单版把反量化结果多舍入了一次 bf16。
- 简单版在 CUDA Graph 开和关时的贪心生成有 3/22 条不同（融合版的同一项检查也通过，但那几行输出当时没有保存下来，正式运行会重新记录）。这 3 处分叉点上 top-2 的差距都恰好是 0.125，也就是 logit 在 16 到 32 之间时 bf16 的最小间隔；分叉之前的 logprob 漂移是 0.0033，和 BF16 自身的 0.0032 相同。所以判为数值平局，不是 Graph 路径的错误。

## 5. 精度（Qwen2.5-7B-Instruct，SGLang 默认设置）

评测用 SGLang 的默认设置：chunked prefill 2048，CUDA Graph 开。

### 5.1 冒烟结果（每个任务前 20 题）

样本量只有 20，只能说明流程和量级，不能当结论。

| 配置 | GSM8K | needle150（约 3300 token） | needle600（约 1.3 万 token） | GSM8K 用时 |
|---|---|---|---|---|
| BF16 | 19/20 | 20/20 | 20/20 | 10 秒 |
| INT8 融合版 | 19/20 | 20/20 | 20/20 | 11 秒 |
| INT8 简单版 | 18/20 | 20/20 | 20/20 | 89 秒 |
| FP8 e4m3 | 0/20 | 1/20 | 1/20 | 18 秒 |
| FP8 e5m2 | 0/20 | 1/20 | 0/20 | 19 秒 |
| BF16，FlashInfer 后端（对照） | 18/20 | 20/20 | 20/20 | 11 秒 |
| FP8 e4m3，FlashInfer 后端 | 1/20 | 0/20 | 0/20 | 18 秒 |
| FP8 e5m2，FlashInfer 后端 | 0/20 | 0/20 | 0/20 | 19 秒 |

### 5.2 正式结果

⏳ GSM8K 全部 1319 题、needle150 和 needle600 各 100 题，每个量化配置与 BF16 逐题配对，做 McNemar 精确检验。

## 6. 已知代价和限制

### 6.1 分块 prefill 读 INT8 前缀有检索代价

长提示词被切成多块时，后面的块要从 INT8 缓存读前面块的 K/V；不分块时整段都用原始精度。在 1.5B 上用 4 个 150 条事实的 needle 测（`chunk_diag`）：

| | 不分块 | 512 分块 |
|---|---|---|
| BF16 | 3/4 | 3/4 |
| INT8 简单版 | 3/4 | 2/4 |
| INT8 融合版 | 3/4 | 2/4 |

- 有一个 needle 所有配置都答错，包括 BF16。
- 另一个 needle，INT8 只要分块就把 6 位数字的最后一位答错，简单版和融合版错得一样。所以这是“分块后读 INT8 前缀”这个设计的代价，不是融合 kernel 引入的。
- 样本只有 4 个。7B 冒烟里（默认 2048 分块，needle 最长约 1.3 万 token）INT8 是 20/20，没有看到这个问题；正式的 100 题结果 ⏳。
- 600 条事实的 needle 超出了 1.5B 的能力（BF16 自己只对 1/4），不能用来比较。

### 6.2 支持范围

只支持 MHA 结构、Triton 后端、page_size 为 1、单卡。其余组合启动时报错。KV 搬移（`move_kv_cache`，推测解码和 sliding window 会用到）、CPU 卸载、PD 分离没有实现。

### 6.3 简单版的开销

简单版每层每次前向都把整个缓存池反量化，只用于验证，不适合实际使用（5.1 里慢 8 倍）。

## 7. FP8 对比

设计时计划做 BF16、FP8、INT8 三方对比。结果是 SGLang 自带的 FP8 KV 在 Qwen 上不能用，所以这一节说明原因。

### 7.1 现象

- 1.5B：top-1 一致率 0.8644，贪心生成没有一条和 BF16 完全相同（4.2）。
- 7B：输出乱码，GSM8K 0/20，换 e5m2、换 FlashInfer 后端都一样（5.1）。

### 7.2 SGLang 做了什么（读源码，未执行验证）

- 没有 checkpoint scale 时，K/V 直接转成 FP8，scale 是 1.0（`mem_cache/memory_pool.py:1061-1067`）。PyTorch 把超过 448 的值转成 e4m3 时得到 NaN（本地验证过）。
- `--quantization-param-path` 可以给每层一个静态 scale，但对未量化的 Qwen2 走不通：加载函数只设置 `k_scale`，没有设置两个后端实际读取的 `k_scale_float`；FlashInfer 读前缀的一条路径不传 scale；Triton kernel 把 Q 也转成 FP8。
- 所以没有在 SGLang 里做“带校准 scale 的 FP8”，改为在纯 PyTorch 里测 FP8 的最好情况。

### 7.3 Qwen 的 K 分布（1.5B，本地探针）

一个 3353 token 的提示词，逐层取真实的 K/V（`results/stage4/Qwen2.5-1.5B-Instruct/local_mac/fp8_probe.json`）：

- K 的最大值 321.3（第 0 层），离群通道的幅值是中位通道的 77.7 倍；V 的最大值 25.4，只有 4.8 倍。
- softmax 对所有 token 共有的分量不敏感，所以看误差相对“去掉通道均值后的张量”有多大：

| 格式 | K 各层平均 | K 最差层 | V 各层平均 |
|---|---|---|---|
| FP8 e4m3（scale 1.0） | 0.077 | 0.83 | 0.031 |
| FP8 e5m2（scale 1.0） | 0.169 | 2.18 | 0.061 |
| FP8 e4m3 + 每层校准 scale | 0.090 | 1.19 | 0.031 |
| INT8 per-token-head | 0.036 | 0.39 | 0.011 |

FP8 是浮点格式，相对精度固定，数值越大绝对误差越大。INT8 per-token-head 按每行的最大值均匀切 127 档。每层一个静态 scale 只能防溢出，不改变相对精度。

### 7.4 模型层面的模拟（1.5B，纯 PyTorch，不经过 SGLang）

在未修改的 HF 模型上，把 K/V 存成“量化再反量化”的结果，prefill 语义和 SGLang 相同。提示词和 4.2 相同（`local_mac/format_sim.json`，MPS、fp16）。

| 格式 | top-1 一致率 | needle（浮点基线 7/10） | 生成完全相同的比例 |
|---|---|---|---|
| INT8 per-token-head | 0.985 | 7/10 | 0.64 |
| FP8 e4m3（scale 1.0） | 0.812 | 0/10 | 0.00 |
| FP8 e5m2（scale 1.0） | 0.862 | 7/10 | 0.45 |
| FP8 e4m3 + 每层校准 scale | 0.693 | 0/10 | 0.05 |

- INT8 的 0.985 与 SGLang 的 0.989、阶段 3 的 0.988 吻合，说明这个模拟可信。
- FP8 e4m3 不经过 SGLang 也一样差，所以问题在格式。
- 校准之后更差：scale 只变了不到 1.5 倍，结果从 0.81 掉到 0.69，说明 FP8 的表现对离群值落在舍入网格的哪一侧很敏感。

### 7.5 没有解释清楚的

- e5m2 的 K 误差最大，但在 1.5B 上 needle 答对 7/10，比 e4m3 好。7.3 的误差指标不能预测 FP8 各变体之间的排序。
- ⏳ 7B 的 K 最大值是否超过 448、7B 上各格式在纯 PyTorch 里的表现（含 100 道 GSM8K），待服务器运行。

## 8. 容量和性能

⏳ 待正式运行。计划的测量：

- 容量：各配置 `--mem-fraction-static 0.85` 下的 `max_total_num_tokens`，与理论比 1.969 对比。
- 吞吐：`bench_serving`，输入/输出长度 (1024, 256)、(4096, 512)、(8192, 512)，每组 3 次取中位数。
- 长请求并发：128 个 8192 + 512 的请求一次发出，看 server 实际同时运行的请求数。
- decode kernel 微基准：batch 32，上下文 1024 到 32768，bf16、fp8、int8 融合、int8 简单四种。

## 9. 简历要点（草稿）

- 在 SGLang（v0.5.12）中实现 per-token-head 动态 INT8 KV cache：新增量化写入、decode、extend 三个融合 Triton kernel 和对应的内存池，KV 每 token 字节数降到 BF16 的 50.8%。⏳ 实测容量比和吞吐数字待补。
- 用三层测试保证正确性：kernel 与 PyTorch 参考逐比特或在 1e-2 误差内一致；不开启时与原版 kernel 逐比特相同；模型级 top-1 一致率 0.989，与离线参考实现持平。
- 定位并解释了 FP8 KV 在 Qwen 上失效的原因（K 的离群通道与浮点格式的相对精度不匹配），并用独立于框架的 PyTorch 模拟交叉验证。
- 发现并量化了分块 prefill 读量化前缀的检索代价。

## 10. 复现

```bash
# 服务器（AutoDL 4090 D）
bash scripts/stage4/setup.sh                       # 环境、补丁、冒烟
RUN_ALL_SMOKE=1 bash scripts/stage4/run_all.sh     # 约 20 到 30 分钟的小规模试跑
nohup bash scripts/stage4/run_all.sh > run_all.log 2>&1 &   # 正式运行，预计 5 到 6 小时

# Mac：结果解压到 results/ 之后
python experiments/stage4/make_report.py results/stage4/Qwen2.5-7B-Instruct
```
