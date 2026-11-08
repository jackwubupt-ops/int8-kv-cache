# 阶段 4 设计：在 SGLang 中实现 per_token_head 动态 INT8 KV Cache

- 日期：2026-11-08
- 前置：[阶段 0-3 分析报告](../../stage0-3-report.md)
- 状态：设计已确认，待审阅设计文档

## 1. 目标与成功标准

**目标**：在 SGLang 中加入 `--kv-cache-dtype int8_per_token_head`，在 Qwen2.5-7B-Instruct 上和 BF16、FP8 KV 做对比，给出显存、并发、吞吐和精度四方面有数据支撑的结论。这是一个简历项目，要能体现三方面能力：改框架源码、做量化、做性能分析。

**成功标准**（结果不管好坏都如实报告）：

| 维度 | 标准 |
|---|---|
| 正确性 | 第 6 节的 kernel 级和模型级检查全部通过 |
| 容量 | INT8 的最大 KV token 数 ≥ BF16 的 1.9 倍（理论值 1/0.508 ≈ 1.97） |
| 性能 | 融合 kernel 在上下文 ≥ 4k 时，decode attention 延迟不高于 BF16；在 KV 容量成为瓶颈的负载下，整体吞吐高于 BF16 |
| 精度 | 完整 GSM8K 上 INT8 对 BF16 的 McNemar 检验 p ≥ 0.05；如果显著下降，报告下降幅度 |

## 2. 现状（调研结论）

- SGLang 最新正式版 v0.5.23 仍然没有 INT8 KV。
- SGLang 有一套 KV 量化插件接口（`srt/layers/quantization/fp4_kv_cache_quant_method.py`）：
  - `KVCacheQuantMethodBase` 负责分配缓存、量化写入、恢复精度。
  - `KV_CACHE_ATTENTION_ACCESS_REGISTRY` 声明每种格式在各个后端、各个阶段该怎么读。
  - `KV_CACHE_QUANT_REGISTRY` 把 `--kv-cache-dtype` 的取值映射到对应的实现类。
- `fp4_mx_block16` 就是照这套接口实现的：写缓存时量化；用 `PLAIN` 方式读取，也就是 `get_key_buffer` 先把整层恢复成 BF16，再交给 Triton 后端。
- Triton extend kernel 对本次新输入的 token 直接用原始精度的 K/V，前缀部分才从 `K_Buffer` 读。这和阶段 3 的 `float_prefill` 语义一致。
- Triton decode grouped kernel 按 token 块读 K、V，FP8 用的是整层一个 `k_scale`，默认值是 1.0。
- SGLang 0.5.23 依赖 `torch==2.14.1` 和 CUDA 13 版的 FlashInfer。服务器当前是 `torch 2.11.0+cu128`，驱动是否支持 CUDA 13 还不知道。

## 3. 范围

**做**：
- 只支持 MHA 结构（Qwen2 系列）；
- 只改 Triton 后端；
- page_size 为 1；
- 开启 CUDA Graph 和 chunked prefill（SGLang 默认）。

**不做**：
- MLA、推测解码、hicache、多卡并行（TP 大于 1）、sliding window。遇到这些组合直接报错，不静默回退。
- vLLM 对比。
- K 按通道量化（列为后续优化）。

## 4. 版本与代码组织

### 4.1 第 0 步：先确定基线版本

开发之前，在服务器上执行 `nvidia-smi` 查看驱动版本：

- **驱动支持 CUDA 13**（≥ 580）：以 SGLang **v0.5.23** 为基线。
- **否则**：以最后一个依赖 CUDA 12.8 PyTorch 的 SGLang 正式版为基线，具体版本在查询 PyPI 后写回本节。

补丁只针对选定的那个版本，一旦定下来就不再改动。

### 4.2 仓库结构

```
sglang_patch/
  BASE_VERSION                         # 基线版本号和 commit
  0001-int8-kv-plain.patch             # 第一步：简单版
  0002-int8-kv-fused-triton.patch      # 第二步：融合 kernel
  apply.sh                             # 克隆基线版本并依次 git am
tests_stage4/                          # 只在 GPU 上运行，Mac 上自动跳过
experiments/stage4/                    # 精度、性能脚本
scripts/stage4/                        # 服务器上的一键脚本
results/stage4/<模型名>/
```

- 在 Mac 上，把基线版本克隆到 `third_party/sglang`（加入 `.gitignore`），在上面逐个提交修改，再用 `git format-patch` 导出到 `sglang_patch/`。
- 服务器上的安装方式：
  1. 新建 venv，执行 `pip install sglang==<基线版本>`，装好所有编译好的依赖；
  2. 运行 `apply.sh`，克隆同一版本并打上补丁；
  3. 执行 `pip install -e third_party/sglang/python --no-deps`，用打过补丁的源码覆盖。
- 不在 GitHub 上建 fork，服务器只需要拉我们这一个仓库。

## 5. 实现

### 5.1 缓存格式（两步共用）

每一层：

| 缓冲区 | 形状 | 类型 |
|---|---|---|
| `k_buffer`、`v_buffer` | `[size + page_size, head_num, head_dim]` | `torch.int8` |
| `k_scale_buffer`、`v_scale_buffer` | `[size + page_size, head_num]` | `torch.bfloat16` |

量化规则和阶段 1-3 的 `int8kv` 完全相同：

- `scale = absmax / 127`，按 (token, head) 计算；
- scale 向上取整到 BF16 能表示的下一个值，保证量化后不会越界；
- `q = clamp(round(x / scale), -127, 127)`。

每个 token 每层占用 `head_num × (2 × head_dim + 4)` 字节，是 BF16 的 (128 + 2) / 256 = 50.8%。容量估算用的 `compute_cell_size` 按这个公式实现。

### 5.2 第一步：简单版

1. 新增 `Int8PerTokenHeadKVCacheMethod(KVCacheQuantMethodBase)`，名字是 `int8_per_token_head`，注册到 `KV_CACHE_QUANT_REGISTRY`：
   - `create_buffers`：按 5.1 分配缓冲区；
   - `quantize_and_store`：用 PyTorch 实现 5.1 的量化规则，按 `loc` 写入；
   - `dequantize_kv_tensor`：`int8 × scale`，返回 BF16。
2. 在 `KV_CACHE_ATTENTION_ACCESS_REGISTRY` 里给它登记 `PLAIN` 读取方式：只允许 Triton 和 torch_native 后端，存储类型为 `torch.int8`，attention 用 BF16。如果现有的辅助函数把存储类型写死成 `uint8`，就加一个参数让它可以指定。
3. 把 `int8_per_token_head` 加进 `--kv-cache-dtype` 的可选值，让 `configure_kv_cache_dtype` 和 `resolve_kv_cache_quant` 能解析它；遇到第 3 节列出的不支持组合时直接报错退出。

**已知代价**：每层每一步都要把整个缓存池恢复成 BF16，所以吞吐会变差。第一步的意义在于拿到显存收益，并为第二步提供参照。

### 5.3 第二步：融合 kernel

1. 新增读取方式 `NATIVE_INT8`：Triton 后端的 prefill 和 decode 都直接读 int8 缓冲区；内存池加一个 `get_kv_scale_buffer(layer_id)`，返回该层的 K、V scale。
2. **decode**：`_fwd_grouped_kernel_stage1` 加一个编译期开关 `KV_INT8` 和两个 scale 指针：
   - 每读一块 K，先把 int8 转成 BF16 再和 Q 做点积，然后每一列乘上对应 token 的 `s_k`；
   - 读 V 之前，把概率 `p` 乘上每个 token 的 `s_v`，再和 BF16 的 V 做点积。
   - 这和阶段 2 中 integer 模式的公式是同一套数学，只是 Q 保持 BF16，不量化。
3. **extend**：`extend_attention.py` 中读缓存前缀的循环做同样的修改；本次新输入 token 的那部分循环不变，继续用原始精度。
4. **写入**：新增一个 Triton kernel 完成"量化并写缓存"，替换第一步的 PyTorch 实现。
5. 开关关闭时，kernel 编译出来和原来完全一样，BF16 和 FP8 路径不受影响。

## 6. 正确性验证

| 层级 | 内容 | 通过标准 |
|---|---|---|
| L0 量化 | Triton 写入和 PyTorch 写入，对比 `int8kv.quant` 参考实现 | scale 逐比特一致；int8 值不一致的比例 < 1e-4（只允许 round-half 边界的差异） |
| L0 decode | 融合 decode 对比"恢复精度后用 FP32 计算的参考 attention"，覆盖多种 batch、上下文长度和 GQA 分组 | 输出相对 L2 < 1e-2 |
| L0 extend | 同上，覆盖"有前缀 + 新输入 token"的情况 | 输出相对 L2 < 1e-2 |
| L0 一致性 | 融合版对比简单版 | 余弦相似度 > 0.9999 |
| L0 不回归 | 开关关闭时 BF16、FP8 的输出 | 和打补丁前逐比特一致 |
| L1 模型 | Qwen2.5-1.5B 在 SGLang 中，INT8 对 BF16：先用 BF16 生成，再把生成内容作为输入，比较每个位置的 Top-1 和 logprob | Top-1 一致率 ≥ 99%（阶段 3 为 99.4%） |
| L1 长上下文 | 同上，用阶段 3 的 needle 提示词 | 中位数和"超过阈值的比例"和阶段 3 在同一量级 |

## 7. 精度评测（Qwen2.5-7B-Instruct）

四种配置：BF16、FP8 E4M3（SGLang 默认 scale 为 1.0）、INT8 简单版、INT8 融合版。

| 任务 | 设置 |
|---|---|
| GSM8K | 完整 1319 题，提示词和答案提取与阶段 3 相同，贪心解码，最多 1024 个 token，批量推理；每题答案都保存，对 BF16 做 McNemar 检验 |
| needle（标准） | 100 题，150 条事实（约 3300 token），和阶段 3 一致 |
| needle（长） | 100 题，600 条事实（约 13k token），考察更长的上下文 |

## 8. 性能评测（7B，都用 Triton 后端）

所有配置统一：`mem_fraction_static=0.85`、开启 CUDA Graph、page_size 为 1。每组先预热，再跑 3 次取中位数。

| 测试 | 方法 | 记录 |
|---|---|---|
| KV 容量 | 启动服务 | 日志里的 `max_total_num_tokens` |
| 吞吐 | `bench_serving`，随机数据，不限请求速率；输入/输出长度取 (1024, 256)、(4096, 512)、(8192, 512) | 输出 token/s、TTFT、TPOT |
| 最大并发 | 输入 8192、输出 512 的长请求一次性提交 256 个 | 同时在跑的请求数峰值、饱和时的吞吐 |
| kernel 延迟 | 直接调用 decode attention：batch 32，上下文 1k、4k、16k、32k | 每次调用的耗时（CUDA event 计时） |
| 生产参考 | BF16 用默认的 FlashInfer 后端跑一遍吞吐测试 | 和 Triton BF16 的差距 |

## 9. 服务器运行方式

每个环节一条命令，都在后台用 `nohup` 运行，日志和结果写到 `results/stage4/`：

| 脚本 | 作用 |
|---|---|
| `scripts/stage4/setup.sh` | 检查驱动 → 建 venv → 安装基线版本 → 打补丁 → 用 ModelScope 下载模型 |
| `scripts/stage4/test.sh` | 运行第 6 节的全部检查，输出一份汇总（通过或失败，以及关键数字） |
| `scripts/stage4/eval.sh` | 第 7 节的精度评测 |
| `scripts/stage4/bench.sh` | 第 8 节的性能评测 |

- 模型放到数据盘：`/root/autodl-tmp/models/`。7B 约 15GB，系统盘空间不够。
- 脚本运行前检查当前 venv 和补丁版本，不对就直接退出，避免用错环境跑出无效数据。
- 结果的传回方式不变：打包后下载到 Mac，由我来提交。

## 10. 风险与应对

| 风险 | 应对 |
|---|---|
| 驱动不支持 CUDA 13 | 第 0 步先查驱动，必要时换成 CUDA 12.8 的基线版本 |
| 新 kernel 在 CUDA Graph 下出错 | kernel 内不做 host 同步；L1 检查在开、关 CUDA Graph 两种情况下都跑 |
| 没法 SSH，调试来回成本高 | L0 测试一次跑完全部用例，失败时输出最小复现参数，减少来回轮数 |
| 第一步简单版吞吐太差，测试时间过长 | 性能测试中简单版只测容量和一组吞吐 |
| Triton 版本差异导致 kernel 编译失败 | 用基线版本自带的 Triton，不单独升级 |
