# 阶段 4 报告：在 SGLang 中实现 per-token-head 动态 INT8 KV Cache

> **状态：正式版（2026-11-20）。** 7B 上的正式运行数据来自 `results/stage4/Qwen2.5-7B-Instruct/`（服务器代码版本 `009ac1b`，结果包 `stage4_results_20261120_085337.tar.gz`），汇总表由 `experiments/stage4/make_report.py` 生成在同目录的 `report_tables.md`。本文所有数字都来自仓库里的结果文件或记录在 [工程底稿](stage4-engineering-draft.md)、[决策记录](stage4-decision-log.md) 中的服务器输出，没有估计值；凡是推理而非测量的地方会写明。

- 基线：SGLang v0.5.12（commit `127b9e32`），补丁 6 个，见 `sglang_patch/`
- 硬件：AutoDL RTX 4090 D（24 GB），PyTorch 2.11.0+cu128。Task 0–8 的检查在驱动 570 上做，正式运行在驱动 595.71.05 上做，软件栈相同
- 模型：正确性检查用 Qwen2.5-1.5B-Instruct，评测用 Qwen2.5-7B-Instruct
- 前置：[阶段 0-3 报告](stage0-3-report.md)、[设计文档](superpowers/specs/2026-11-08-stage4-sglang-int8-kv-design.md)
- 过程记录：[工程底稿](stage4-engineering-draft.md)、[决策记录](stage4-decision-log.md)

## 1. 结论

1. **INT8 KV 已经接进 SGLang 的 Triton 后端，并通过了 kernel 级和模型级的正确性检查。** `--kv-cache-dtype int8_per_token_head` 在写入、decode、extend 三处都用融合的 Triton kernel，直接读写 int8 和每个 (token, head) 的 scale。正式运行的代码版本上 L0 共 93 项全部通过。
2. **容量翻倍，并且确实转化成了吞吐。** 同样 0.85 的显存比例下，INT8 的 KV 容量是 196,433 token，BF16 是 99,750，比值 1.969，与理论值相同。128 个 8192+512 的长请求同时发出时，实际同时运行的请求数从 11 个变成 23 个。三种输入输出长度下的输出吞吐比 BF16 高 21%–23%，长请求测试高 28%。
3. **融合 decode kernel 在 4k 以上上下文比 BF16 快。** batch 32 时，4096 上下文 0.233 ms 对 0.355 ms，32768 上下文 1.380 ms 对 2.377 ms（快 1.72 倍）；1024 上下文两者都是 0.105 ms。FP8 kernel 仍然更快一些（32768 时 1.207 ms，比 INT8 快 13%）。
4. **长上下文检索没有损失。** needle150 和 needle600（约 1.3 万 token，按 2048 分块 prefill）各 100 题，INT8 融合版都是 96/100，BF16 都是 95/100。
5. **GSM8K 上融合版没有达到设计文档的精度标准。** 标准是 McNemar p ≥ 0.05；融合版 89.16% 对 BF16 90.30%，差 1.14 个百分点，p = 0.020。简单版（同一种量化，只是先反量化成 bf16 再算）差 0.38 个百分点，p = 0.51。作为对照，只把 BF16 的 attention 后端从 Triton 换成 FlashInfer（两者都无损）就有 67 道题对错改变、准确率差 0.53 个百分点（p = 0.46）。融合版和简单版彼此之间不显著（p = 0.12），融合版相对 FlashInfer 版 BF16 也不显著（p = 0.42）。现有数据不能把这 1 个百分点左右的差距判定为噪声，也不能确定它来自融合 kernel，详见 5.3。
6. **融合 kernel 是必要的。** 简单版每次前向都把整层缓存反量化：GSM8K 用时 263 秒（融合版 76 秒）；1024×256 压测吞吐只有融合版的 39%；在 0.85 的显存比例下 256 个并发请求会显存不足而崩溃，只能降到 0.75 才跑完。
7. **SGLang 自带的 FP8 KV 在 Qwen2.5-7B 上不可用，原因在格式本身。** GSM8K 约 2%、needle 0%–7%，换 e5m2、换 FlashInfer 都一样。纯 PyTorch 模拟（不经过 SGLang）复现了同样的崩溃：7B 的 K 最大值 422，没有超过 e4m3 的 448，但离群通道的幅值是中位通道的 108 倍，浮点格式在这些通道上的绝对误差压过了 token 之间真正有区分度的部分；每层校准一个静态 scale 也没有改善。

## 2. 成功标准逐项

| 维度 | 标准（设计文档第 1 节） | 结果 | 状态 |
|---|---|---|---|
| 正确性 | kernel 级和模型级检查全部通过 | L0 93/93（`009ac1b`）；L1 四个 INT8 配置均通过（第 4 节） | ✅ |
| 容量 | INT8 的最大 KV token 数 ≥ BF16 的 1.9 倍 | 196,433 / 99,750 = 1.969 | ✅ |
| 性能 | 上下文 ≥ 4k 时融合 decode kernel 的延迟不高于 BF16；KV 容量成为瓶颈时整体吞吐高于 BF16 | kernel 在 4k/16k/32k 分别是 BF16 的 66%/59%/58%；吞吐 +21% 到 +28% | ✅ |
| 精度 | 完整 GSM8K 上 INT8 对 BF16 的 McNemar p ≥ 0.05 | 融合版 p = 0.020（−1.14 pp）；简单版 p = 0.51（−0.38 pp） | ❌ 融合版未达标 |

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

`tests_stage4/` 共 93 项，正式运行（`009ac1b`，`results/stage4/run_all/20261120_014111/l0.log`）全部通过，用时 194 秒。

| 对象 | 检查 |
|---|---|
| 写入 kernel | scale 与 PyTorch 参考逐比特相等（N 取 1、7、4096，bf16 和 fp16 输入）；非连续输入；只写指定槽位；全 0 行；空 batch |
| decode kernel | 与 FP32 参考的相对 L2 < 1e-2（batch 1 和 4，上下文 1 到 4097，GQA 分组 6 和 7）；与简单版的余弦相似度 > 0.9999 |
| extend kernel | 前缀长度 {0, 5, 300} × 新输入长度 {1, 64, 257} 对 FP32 参考；没有前缀时与 BF16 kernel 逐比特相同 |
| 不影响原有路径 | 不传 scale 时，BF16 和 FP8 都与未打补丁的 kernel 逐比特相同 |
| 后端接线（1.5B，`sglang.Engine`） | 融合版能生成，前 8 个 token 与简单版相同；命中前缀缓存前后输出相同；512 分块时，简单版检索到的 needle 融合版也检索到 |

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
- 简单版在 CUDA Graph 开和关时的贪心生成有 3/22 条不同。这 3 处分叉点上 top-2 的差距都恰好是 0.125，也就是 logit 在 16 到 32 之间时 bf16 的最小间隔；分叉之前的 logprob 漂移是 0.0033，和 BF16 自身的 0.0032 相同。所以判为数值平局，不是 Graph 路径的错误。融合版的同一项检查也通过，但那几行输出当时没有保存；`run_all.sh` 不含 L1，所以这次也没有补上。

## 5. 精度（Qwen2.5-7B-Instruct，SGLang 默认设置）

评测用 SGLang 的默认设置：chunked prefill 2048，CUDA Graph 开，贪心解码，GSM8K 最多生成 1024 token。所有配置逐题配对，与 BF16（Triton 后端）做 McNemar 精确检验。

### 5.1 正式结果

| 配置 | GSM8K（1319 题） | needle150（100 题，约 3300 token） | needle600（100 题，约 1.3 万 token） | GSM8K 用时 |
|---|---|---|---|---|
| BF16 | 90.30%（1191） | 95/100 | 95/100 | 84 秒 |
| BF16，FlashInfer 后端（对照） | 89.76%（1184），p = 0.46 | — | — | 88 秒 |
| **INT8 融合版** | **89.16%（1176），p = 0.020** | 96/100，p = 1 | 96/100，p = 1 | 76 秒 |
| INT8 简单版 | 89.92%（1186），p = 0.51 | 95/100，p = 1 | 95/100，p = 1 | 263 秒 |
| FP8 e4m3 | 1.97%（26），p ≈ 0 | 2/100 | 3/100 | 164 秒 |
| FP8 e5m2 | 1.90%（25），p ≈ 0 | 7/100 | 0/100 | 233 秒 |

needle 上 INT8 和 BF16 的差别只有 0 到 1 题。FP8 见第 7 节。

### 5.2 GSM8K 逐题配对

四个无损或接近无损的配置两两比较（"A 独对"是 A 答对、B 答错的题数）：

| A | B | A 独对 | B 独对 | 对错不同 | 答案不同 | McNemar p |
|---|---|---|---|---|---|---|
| BF16 | FlashInfer BF16 | 37 | 30 | 67 | 94 | 0.46 |
| BF16 | INT8 简单版 | 21 | 16 | 37 | 50 | 0.51 |
| BF16 | INT8 融合版 | 26 | 11 | 37 | 58 | 0.020 |
| FlashInfer BF16 | INT8 简单版 | 35 | 37 | 72 | 102 | 0.91 |
| FlashInfer BF16 | INT8 融合版 | 41 | 33 | 74 | 108 | 0.42 |
| INT8 简单版 | INT8 融合版 | 22 | 12 | 34 | 56 | 0.12 |

只有融合版答错、其余三个配置都答对的题有 9 道；反过来只有融合版答对的有 2 道。

### 5.3 怎么理解融合版的 −1.14 pp

- **按设计文档的标准，融合版没有通过。** 本报告不改标准。
- **对错改变的题数不比无损变化多。** 融合版对 BF16 改变了 37 道题，和简单版相同，只有换后端（67 道）的一半左右。三个 Triton 后端的配置彼此接近，FlashInfer 和它们都差得更远，所以 GSM8K 对 attention 实现的任何数值变化都很敏感。
- **但融合版的改变方向偏向答错**（26 对 11），而且在四个配置里它最常"独自答错"（9 对 2）。这是 p = 0.020 的来源。
- **融合版和简单版的数学几乎相同**：两者存同样的 int8 和 scale，区别只是融合版在 kernel 里用 fp32 精确计算 `int8 × scale`，简单版先舍入成 bf16。L1 上融合版的 |Δlogp| 反而略小（0.0199 对 0.0208）。两者在 GSM8K 上的差别不显著（p = 0.12）。
- **结论**：数据能支持的是"融合版在 GSM8K 上可能有 1 个百分点左右的损失，量级与换一个 attention 后端带来的变化相当"；不能支持"无损"，也不能把原因归到融合 kernel 的某个具体环节。要区分噪声和真实损失，需要多次运行（例如不同的 batch 组合或采样种子）估计同一配置自身的波动，这次没有做。

### 5.4 冒烟结果（每个任务前 20 题，2026-11-10 和 2026-11-19 两次一致）

| 配置 | GSM8K | needle150 | needle600 |
|---|---|---|---|
| BF16 | 19/20 | 20/20 | 20/20 |
| INT8 融合版 | 19/20 | 20/20 | 20/20 |
| INT8 简单版 | 18/20 | 20/20 | 20/20 |
| FP8 e4m3 | 0/20 | 1/20 | 1/20 |
| FP8 e5m2 | 0/20 | 1/20 | 0/20 |

## 6. 已知代价和限制

### 6.1 分块 prefill 读 INT8 前缀

长提示词被切成多块时，后面的块要从 INT8 缓存读前面块的 K/V；不分块时整段都用原始精度。在 1.5B 上用 4 个 150 条事实的 needle、512 分块测（`chunk_diag`）：

| | 不分块 | 512 分块 |
|---|---|---|
| BF16 | 3/4 | 3/4 |
| INT8 简单版 | 3/4 | 2/4 |
| INT8 融合版 | 3/4 | 2/4 |

- 1.5B 上有一个 needle，INT8 只要分块就把 6 位数字的最后一位答错，简单版和融合版错得一样，所以是"分块后读 INT8 前缀"这个设计的代价，不是融合 kernel 引入的。样本只有 4 个。
- **7B 上没有看到这个代价**：按 SGLang 默认的 2048 分块，needle600 每题约分成 7 块，INT8 两种实现都是 96/100 或 95/100，BF16 是 95/100（5.1）。

### 6.2 支持范围

只支持 MHA 结构、Triton 后端、page_size 为 1、单卡。其余组合启动时报错。KV 搬移（`move_kv_cache`，推测解码和 sliding window 会用到）、CPU 卸载、PD 分离没有实现。

### 6.3 简单版的开销

简单版每层每次前向都把整个缓存池反量化成 fp32 再转 bf16，这部分临时显存不在 SGLang 为 KV 池预留的预算里：

- 0.85 的显存比例下，256 个 1024+256 的请求同时运行时显存不足（反量化一层需要 384 MiB，只剩 169 MiB），服务器崩溃。降到 0.75 后跑完，此时 KV 池只有 111,210 token，所以 8.2 里简单版的数字来自更小的池子。
- GSM8K 用时是融合版的 3.5 倍；1024×256 吞吐 609 对 1575 token/s；decode kernel 在 32768 上下文慢 17.8 倍。

## 7. FP8 对比

设计时计划做 BF16、FP8、INT8 三方对比。结果是 SGLang 自带的 FP8 KV 在 Qwen 上不能用，所以这一节说明原因。

### 7.1 现象

- 1.5B：top-1 一致率 0.8644，贪心生成没有一条和 BF16 完全相同（4.2）。
- 7B：GSM8K 约 2%，needle 0%–7%（5.1）。冒烟里换 FlashInfer 后端也一样（e4m3 1/20、0/20、0/20）。

### 7.2 SGLang 做了什么（读源码，未执行验证）

- 没有 checkpoint scale 时，K/V 直接转成 FP8，scale 是 1.0（`mem_cache/memory_pool.py:1061-1067`）。PyTorch 把超过 448 的值转成 e4m3 时得到 NaN（本地验证过）。但 7B 探针里 K 的最大值是 422，没有溢出，所以 7B 的崩溃不是溢出造成的（7.3）。
- `--quantization-param-path` 可以给每层一个静态 scale，但对未量化的 Qwen2 走不通：加载函数只设置 `k_scale`，没有设置两个后端实际读取的 `k_scale_float`；FlashInfer 读前缀的一条路径不传 scale；Triton kernel 把 Q 也转成 FP8。
- 所以没有在 SGLang 里做"带校准 scale 的 FP8"，改为在纯 PyTorch 里测 FP8 的最好情况。

### 7.3 Qwen 的 K 分布（探针）

一个 3353 token 的提示词，逐层取真实的 K/V。softmax 对所有 token 共有的分量不敏感，所以看误差相对"去掉通道均值后的张量"有多大（去均值相对 L2）。

| | 1.5B（Mac，`local_mac/fp8_probe.json`） | 7B（服务器，`fp8_probe.json`） |
|---|---|---|
| K 最大值 | 321.3 | 422.0（28 层都未超过 448） |
| K 离群通道 / 中位通道 | 77.7 倍 | 108.5 倍 |
| V 最大值 | 25.4 | 69.5 |
| V 离群通道 / 中位通道 | 4.8 倍 | 4.7 倍 |

| 格式 | 1.5B K 平均 | 1.5B K 最差层 | 7B K 平均 | 7B K 最差层 | 7B V 平均 |
|---|---|---|---|---|---|
| FP8 e4m3（scale 1.0） | 0.077 | 0.83 | 0.094 | 0.90 | 0.029 |
| FP8 e5m2（scale 1.0） | 0.169 | 2.18 | 0.174 | 1.48 | 0.058 |
| FP8 e4m3 + 每层校准 scale | 0.090 | 1.19 | 0.085 | 0.63 | 0.029 |
| INT8 per-token-head | 0.036 | 0.39 | 0.042 | 0.34 | 0.009 |

FP8 是浮点格式，相对精度固定，数值越大绝对误差越大。INT8 per-token-head 按每行的最大值均匀切 127 档。每层一个静态 scale 只能防溢出，不改变相对精度。

### 7.4 模型层面的模拟（纯 PyTorch，不经过 SGLang）

在未修改的 HF 模型上，把 K/V 存成"量化再反量化"的结果，prefill 语义和 SGLang 相同。提示词和 4.2 相同。1.5B 在 Mac 上跑（MPS、fp16，`local_mac/format_sim.json`），7B 在服务器上跑（CUDA、bf16，`format_sim.json`，GSM8K 取前 100 题）。

| 格式 | 1.5B top-1 | 1.5B needle | 7B top-1 | 7B KL | 7B needle | 7B GSM8K（100 题） |
|---|---|---|---|---|---|---|
| 浮点基线 | — | 7/10 | — | — | 10/10 | 90% |
| INT8 per-token-head | 0.985 | 7/10 | 0.992 | 0.0014 | 10/10 | 89%（p = 1） |
| FP8 e4m3（scale 1.0） | 0.812 | 0/10 | 0.664 | 1.97 | 1/10 | 2% |
| FP8 e5m2（scale 1.0） | 0.862 | 7/10 | 0.610 | 2.07 | 0/10 | 3% |
| FP8 e4m3 + 每层校准 scale | 0.693 | 0/10 | 0.679 | 2.15 | 1/10 | 4% |

- INT8 的模拟结果与 SGLang 实测吻合（1.5B：0.985 对 SGLang 0.989、阶段 3 0.988；7B：GSM8K 前 100 题 89% 对浮点 90%），说明这个模拟可信。
- 7B 上 FP8 在模拟里也崩溃，而且 22 个提示词没有一个出现 NaN，所以问题在格式的精度，不是溢出，也不是 SGLang 的实现。
- 校准 scale 在 7B 上 top-1 略好（0.664 → 0.679），但 needle 和 GSM8K 仍然崩溃；1.5B 上反而更差。

### 7.5 没有解释清楚的

- 1.5B 上 e5m2 的 K 误差最大，needle 却答对 7/10，比 e4m3 好；7B 上这个反常没有出现（e5m2 0/10，e4m3 1/10）。7.3 的误差指标能区分 INT8 和 FP8，但不能预测 FP8 各变体之间的排序。
- 探针只用了一个提示词；7B 的其他输入是否会让 K 超过 448 没有测。模拟里没有 NaN，说明至少这 22 个提示词没有溢出。

## 8. 容量和性能

所有配置用 Triton 后端（另加 BF16 FlashInfer 作参考），`--mem-fraction-static 0.85`，INT8 简单版例外（0.75，见 6.3）。

### 8.1 KV 容量

| 配置 | 容量（token） | 相对 BF16 | KV 显存 |
|---|---|---|---|
| BF16 | 99,750 | 1.000 | 5.32 GB |
| FP8 e4m3 | 199,503 | 2.000 | 5.32 GB |
| **INT8 融合版** | **196,433** | **1.969** | 5.32 GB |
| INT8 简单版（0.75） | 111,210 | 1.115 | 3.02 GB |
| BF16，FlashInfer | 99,751 | 1.000 | 5.32 GB |

INT8 比 FP8 少 1.5%，是每个 (token, head) 两个 bf16 scale 的开销（260 对 256 字节），与 3.1 的公式一致。

### 8.2 吞吐

`bench_serving`，random-ids 数据，请求一次性全部发出，每组 3 次取中位数，所有请求都完成。

| 输入×输出（请求数） | BF16 | FP8 e4m3 | **INT8 融合版** | INT8 简单版（0.75） | BF16 FlashInfer |
|---|---|---|---|---|---|
| 1024×256（256） | 1285.6 | 1564.2 | **1575.2（+22.5%）** | 608.9 | 1265.2 |
| 4096×512（64） | 502.3 | 630.9 | **618.5（+23.1%）** | — | 508.4 |
| 8192×512（32） | 252.7 | 311.2 | **306.9（+21.4%）** | — | 260.4 |

单位是输出 token/s，括号里是相对 BF16。

| 输入×输出 | 指标 | BF16 | FP8 e4m3 | INT8 融合版 |
|---|---|---|---|---|
| 1024×256 | TTFT 中位 | 20,176 ms | 14,181 ms | 14,277 ms |
| 1024×256 | TPOT 中位 | 38.7 ms | 56.4 ms | 55.7 ms |
| 8192×512 | TTFT 中位 | 28,399 ms | 16,449 ms | 17,222 ms |
| 8192×512 | TPOT 中位 | 30.3 ms | 35.9 ms | 34.9 ms |

- 请求一次性发出，TTFT 主要是排队时间。容量大的配置能更早接纳请求，所以 TTFT 降了约 30%–40%。
- TPOT 变长是因为同时运行的请求更多、每步 decode 的 batch 更大（推理，没有直接测每步 batch；按容量算，1280 token 的请求 BF16 池最多容纳 77 个，INT8 池最多 153 个）。总吞吐仍然更高。
- INT8 融合版与 FP8 的吞吐差在 −2% 到 +1% 之间。

### 8.3 长请求并发

128 个 8192+512 的请求一次发出，看服务器实际同时运行的请求数：

| 配置 | 峰值同时运行 | 输出 token/s |
|---|---|---|
| BF16 | 11 | 251.8 |
| FP8 e4m3 | 23 | 336.6 |
| **INT8 融合版** | **23** | **323.1（+28.3%）** |
| BF16，FlashInfer | 11 | 259.9 |

### 8.4 decode kernel 微基准

batch 32，Q 头 28、KV 头 4、head_dim 128，kv_splits 8，中位耗时（ms）：

| 上下文 | BF16 | FP8 | **INT8 融合** | INT8 简单 | INT8 融合 / BF16 |
|---|---|---|---|---|---|
| 1024 | 0.105 | 0.100 | 0.105 | 0.715 | 1.00 |
| 4096 | 0.355 | 0.205 | 0.233 | 3.114 | 0.66 |
| 16384 | 1.225 | 0.637 | 0.725 | 12.333 | 0.59 |
| 32768 | 2.377 | 1.207 | 1.380 | 24.625 | 0.58 |

- 上下文越长越受显存带宽限制，INT8 读的字节数约为 BF16 的一半，耗时也接近一半。
- 4k 以上 INT8 比 FP8 慢约 14%。可能的原因是要多读每行的 scale、多做两次乘法（`qk × s_k`、`p × s_v`），这一点没有单独测量。

## 9. 简历要点

- 在 SGLang（v0.5.12）中实现 per-token-head 动态 INT8 KV cache：新增量化写入、decode、extend 三个融合 Triton kernel 和对应的内存池。Qwen2.5-7B 上 KV 容量提升到 BF16 的 1.97 倍，长请求并发从 11 提升到 23，服务吞吐提升 21%–28%，32k 上下文的 decode attention 快 1.72 倍。
- 用三层测试保证正确性：kernel 与 PyTorch 参考逐比特或在 1e-2 误差内一致；不开启时与原版 kernel 逐比特相同；模型级 top-1 一致率 0.989，与离线参考实现持平。长上下文检索（1.3 万 token needle）无损。
- 用逐题配对的 McNemar 检验和"换 attention 后端"的噪声底评估精度，如实报告 GSM8K 上 1.1 个百分点、p = 0.02 的差距。
- 定位 SGLang FP8 KV 在 Qwen2.5-7B 上失效的原因（K 的离群通道与浮点格式的相对精度不匹配，非溢出），并用独立于框架的 PyTorch 模拟交叉验证。

## 10. 复现

```bash
# 服务器（AutoDL 4090 D）
bash scripts/stage4/setup.sh                       # 环境、补丁、冒烟
RUN_ALL_SMOKE=1 bash scripts/stage4/run_all.sh     # 约 30 分钟的小规模试跑
nohup bash scripts/stage4/run_all.sh > run_all.log 2>&1 < /dev/null &   # 正式运行，约 3.5 小时
# 这次另外补跑的两项
MEM_FRACTION=0.75 bash scripts/stage4/bench.sh int8_per_token_head:plain:triton:light
bash scripts/stage4/eval.sh auto:plain:flashinfer gsm8k

# Mac：结果解压到 results/ 之后
python experiments/stage4/make_report.py results/stage4/Qwen2.5-7B-Instruct
```

网络慢时的安装方式见 [工程底稿](stage4-engineering-draft.md) 9.3 节。
