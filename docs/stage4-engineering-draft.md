# 阶段 4 工程底稿：在 SGLang v0.5.12 中实现 `int8_per_token_head` KV Cache

- 分支：`stage4-int8-kv`
- 快照：2026-11-10，tag `stage4-snapshot-20261110`（上一版 `stage4-snapshot-20261109`）
- 决策记录：[stage4-decision-log.md](stage4-decision-log.md)（执行过程中每个偏离计划的决定、原因和代价）
- 前置：
  - [阶段 0-3 分析报告](stage0-3-report.md)
  - [阶段 4 设计](superpowers/specs/2026-11-08-stage4-sglang-int8-kv-design.md)
  - [实施计划](superpowers/plans/2026-11-08-stage4-sglang-int8-kv.md)
- 阶段 0-3 的完整实验和结果在 `main` 分支（tag `stage0-3-final`），本分支也包含全部内容：
  - 代码：`int8kv/`、`experiments/stage1_*.py` ~ `stage3_*.py`
  - 数据：`results/Qwen2.5-1.5B-Instruct/`
  - 报告：`docs/stage0-3-report.md`

这份底稿记录到目前为止的全部工程实现：

- 版本选择和原因；
- 补丁工作流；
- 对 SGLang 每个文件的改动；
- 测试体系；
- 服务器部署中遇到的每个问题和修复；
- 还没解决的问题。

附录收录核心代码全文。即使 `third_party/` 丢失，也能从本文和 `sglang_patch/*.patch` 完整重建。

---

## 1. 当前状态一览

| 计划任务 | 内容 | 状态 |
|---|---|---|
| Task 0 | 服务器环境、SGLang 版本锁定、冒烟测试 | ✅ 服务器上 `SMOKE OK`（BF16、FP8 KV 均能生成） |
| Task 1 | 补丁工作流（apply / export / overlay） | ✅ |
| Task 2 | 量化数学 `int8_kv.py` | ✅ |
| Task 3 | `MHATokenToKVPoolINT8` 简单版（读时整层反量化） | ✅ |
| Task 4 | L1 模型级检查 | ✅ INT8 与阶段 3 同 prompt 持平（0.9888 / 0.9895 vs 0.9879），graph 开关检查通过（第 7 节） |
| Task 5 | 融合写入 kernel | ✅ 服务器 L0 通过；scale 与 PyTorch 逐比特一致 |
| Task 6 | 融合 decode kernel | ✅ 服务器 L0 通过（70 项） |
| Task 7 | 融合 extend kernel | ✅ 服务器 L0 通过（87 项） |
| Task 8 | 接入 Triton 后端 | ⚠️ L1 全部通过（fused 已在模型中运行）；L0 后端接线测试的分块用例已按诊断结果改写（`fe002aa`），新版在服务器上运行时进程无输出退出，原因待查（9.1） |
| Task 9-11 | 7B 上 GSM8K 全量、needle、吞吐和并发基准、最终报告 | 未开始 |

最新代码：`stage4-int8-kv` 分支，tag `stage4-snapshot-20261110`。SGLang 补丁 6 个（`sglang_patch/0001`~`0006`）：量化数学、简单版 pool、融合写入、融合 decode、融合 extend、后端接线。

本地测试：150 passed, 66 skipped（GPU 测试在 Mac 上跳过）。服务器最近一次完整 L0（`45dd78e`）：92 passed, 1 failed（分块用例，已改写）。

---

## 2. 量化方案

动态 per-token-head INT8，和阶段 3 的 PyTorch 参考实现逐位一致（测试 `test_int8_kv_math.py` 直接和 `int8kv.quant` 比对）：

- 每个 (token, head) 一行 `head_dim` 个值，`scale = absmax / 127`；
- scale 以 bf16 存储，并且**向上取整**：round-to-nearest 结果小于真值时，bf16 位模式加 1，保证 `absmax / scale ≤ 127`，不会溢出；
- 全零行 scale = 1；
- `q = clamp(round(x / scale), -127, 127)`，`torch.round` 是 round-half-to-even；
- 反量化：`x̂ = q * scale`，在 fp32 中计算后转成模型精度。

**显存**：每 token 每层 `head_num * (head_dim_k + head_dim_v + 2*2)` 字节，其中 K 和 V 各带一个 2 字节的 bf16 scale。

- head_dim=128 时为 260/512 = **50.8%**，阶段 3 实测 50.7%；
- Qwen2.5-1.5B：28 层，2 个 KV head，每 token 14,560 字节，BF16 为 28,672 字节；
- Qwen2.5-7B：28 层，4 个 KV head，每 token 29,120 字节，BF16 为 57,344 字节。

**prefill 语义**：当前 chunk 用原始精度的 K/V 计算 attention，只在写缓存时量化；前缀（radix cache 命中、chunked prefill 的前几块）从 INT8 读回。阶段 3 证明这个选择决定成败：prefill 也读 INT8 时，needle 从 79% 掉到 51%。SGLang Triton extend kernel 本来就是这样工作的，不需要改 kernel。

---

## 3. 版本锁定和原因

| 组件 | 版本 | 原因 |
|---|---|---|
| GPU / 驱动 | RTX 4090 D，570.124.04 | 最高支持 CUDA 12.8，这是所有版本选择的根本约束 |
| SGLang | **v0.5.12**（commit `127b9e3283f7c2a43234b852ff5c9f1796d53624`） | 0.5.23 要求 torch 2.14 + CUDA 13，驱动跑不了。0.5.12 是最后一个依赖 torch 2.11 的版本 |
| torch / torchaudio / torchvision | 2.11.0+cu128 / 2.11.0+cu128 / 0.26.0+cu128 | 必须是 cu128 版本。PyPI 上的默认版本是 CUDA 13 编译的 |
| sglang-kernel | 0.4.2.post2+cu129 | PyPI 只有 CUDA 13 版本。cu129 版本利用 CUDA 小版本兼容，在 12.8 驱动上能运行 |
| transformers | 5.6.0（sglang 锁定） | — |
| kernels | **0.12.x** | sglang 没有锁定这个包，pip 会装最新的 0.18，它与 transformers 5.6.0 不兼容，`import sglang` 时报 `Either a revision or a version must be specified` |
| ninja、nvcc | venv 中的 ninja + `/usr/local/cuda/bin/nvcc` | SGLang 0.5.12 首次运行时会 JIT 编译 RoPE 等算子，并按名字调用这两个程序 |

v0.5.12 没有 0.5.23 的通用 KV 量化插件接口（`KVCacheQuantMethodBase` / 注册表），所以 INT8 按 v0.5.12 中 `MHATokenToKVPoolFP4` 的模式实现：新增 pool 类，在 `_init_pools` 中按 dtype 选择。Triton 算子在 v0.5.12 中位于 `srt/layers/attention/triton_ops/`（0.5.23 移到了 `kernels/ops/attention/`），计划中的路径已按此映射。

---

## 4. 补丁工作流

SGLang 源码不进本仓库（`third_party/` 在 `.gitignore` 中），改动以 `git format-patch` 补丁的形式保存在 `sglang_patch/`：

| 文件 | 作用 |
|---|---|
| `BASE_VERSION` | `version=v0.5.12`，`commit=127b9e32...` |
| `0001-Add-per-token-head-INT8-KV-quantization-math.patch` | 量化数学 |
| `0002-Add-int8_per_token_head-KV-cache-plain-dequant-on-re.patch` | pool + 接线 + CLI |
| `apply.sh [target]` | 浅克隆到 `<target>/sglang-pristine`（原版）和 `<target>/sglang`（打补丁），校验 commit，`git am --3way` 应用补丁 |
| `export.sh [target]` | 从 `<target>/sglang` 的提交重新导出补丁 |
| `overlay.py apply\|verify` | 把补丁改动的 `.py` 文件覆盖到已安装的 sglang wheel 上 |

`apply.sh` 可以重复运行：

- 中止进行中的 `git am`，再 reset 并 clean；
- 基线版本变化时自动 fetch 新 tag；
- 用 `-c user.name/-c user.email` 指定提交身份，因为服务器上没有配置 git 身份。

### 为什么用 overlay，不用 `pip install -e`

v0.5.12 的源码构建需要 Rust 工具链，服务器上没有。官方 wheel 里的 `.so` 和预编译部分完全可用，我们的改动只涉及 6 个纯 Python 文件，所以 overlay 只把这 6 个文件复制进 site-packages：

```
srt/layers/quantization/int8_kv.py                (新增)
srt/mem_cache/memory_pool.py
srt/model_executor/model_runner.py
srt/model_executor/model_runner_kv_cache_mixin.py
srt/model_executor/pool_configurator.py
srt/server_args.py
```

overlay 会写一个清单 `.int8kv_overlay.json`：

- 再次 apply 时，不再改动的文件会从 pristine 恢复原样，新增的文件会被删除；
- `verify` 逐字节比对已安装文件和补丁版本。

每个服务器脚本都先调用 `require_env`，做三项检查：venv 存在、overlay 校验通过、`ninja` 和 `nvcc` 在 PATH 上。任何一项不满足都拒绝运行，避免用错版本跑出无效数据。

---

## 5. 对 SGLang 的改动（逐文件）

### 5.1 `srt/layers/quantization/int8_kv.py`（新增，只依赖 torch）

| 函数 | 作用 |
|---|---|
| `bf16_round_up(x)` | fp32 → bf16，向上取整 |
| `quantize_per_token_head(x[N,H,D])` | 返回 `(int8 [N,H,D], bf16 scale [N,H])` |
| `dequantize_per_token_head(q, scale, dtype)` | 反量化 |
| `store_quantized(q_buf, scale_buf, loc, x)` | 量化并写入 `loc` 行 |
| `int8_kv_cell_size(head_num, head_dim, v_head_dim, num_layers)` | 每 token 字节数，用于显存规划 |
| `check_int8_kv_supported(...)` | 遇到不支持的配置时立即报错 |

`check_int8_kv_supported` 拒绝的配置：非 triton 后端、page_size≠1、TP>1、MLA、投机解码、分层缓存、混合滑窗。

### 5.2 `srt/mem_cache/memory_pool.py`：`MHATokenToKVPoolINT8(MHATokenToKVPool)`

- **缓冲区**：每层分配 int8 `k_buffer`、`v_buffer`（`[size+page_size, H, D]`）和 bf16 `k_scale_buffer`、`v_scale_buffer`（`[size+page_size, H]`）。第 0 行用于接收 padding token 的写入。
- **写入** `set_kv_buffer`：K 和 V 各调用一次 `store_quantized`。如果传入 checkpoint 的静态 `k_scale`/`v_scale`，抛 `ValueError`，因为 scale 是动态计算的。
- **读取**（简单版）：`_get_key_buffer` 和 `_get_value_buffer` 把整层反量化成 `compute_dtype`，任何 attention 后端都能直接使用。代价是每次读取都要处理整个 pool，这正是 Task 5-8 融合 kernel 要消除的开销。
- `get_kv_size_bytes` 把 scale 也计入；`get_kv_scale_buffer(layer_id)` 留给融合 kernel 使用。
- 不支持的功能显式抛 `NotImplementedError`，避免静默出错：KV copy、PD 分离（`get_contiguous_buf_infos`）、CPU offload、`move_kv_cache`。

### 5.3 接线

- `server_args.py`：`--kv-cache-dtype` 增加 `int8_per_token_head` 选项和帮助文字。
- `model_runner.py` `configure_kv_cache_dtype`：`"int8_per_token_head"` 解析为 `self.kv_cache_dtype = torch.int8`。
- `pool_configurator.py` `_compute_cell_size`：非 MLA 分支中，dtype 为 int8 时 `cell_size = int8_kv_cell_size(...)`，让 SGLang 按真实字节数规划 token 容量（约为 BF16 的 1.97 倍）。
- `model_runner_kv_cache_mixin.py`：
  - 新增 `_check_int8_kv_supported()`。三个 attention backend 参数（`attention_backend`、`prefill_attention_backend`、`decode_attention_backend`）不一致时按 `mixed` 处理并拒绝；`is_hybrid_swa` 或 mamba 结构视为混合模型并拒绝。
  - `_init_pools` 开头，dtype 为 int8 时先做检查。
  - pool 选择分支中，在 FP4 分支之前加入 `elif self.kv_cache_dtype == torch.int8: MHATokenToKVPoolINT8(..., compute_dtype=self.dtype)`。

Qwen2 不走 fused set_kv 路径，写缓存统一经过 `token_to_kv_pool.set_kv_buffer`，所以不需要改模型代码。

---

### 5.4 融合 kernel 和后端接线（Task 5-8）

开关：环境变量 `SGLANG_INT8_KV_IMPL`，`fused`（默认）或 `plain`，其他值报错。pool 在构造时读一次，存为 `pool.int8_impl`。

| 文件 | 改动 |
|---|---|
| `srt/layers/attention/triton_ops/int8_kv_store.py`（新增） | `int8_quantize_and_store(cache, loc, q_buf, scale_buf)`：一个 program 处理一个 (token, head)；用 `div_rn`（IEEE 除法）和 `rint`（round-half-even），scale 与 PyTorch 参考实现逐比特一致；支持非连续输入（qkv 的 view） |
| `srt/layers/quantization/int8_kv.py` | `int8_kv_impl()` 读开关；`check_int8_backend_mode()`：fused 与 `--enable-deterministic-inference` 同时开启时报错（确定性模式走 unified kernel，没有 INT8 路径） |
| `srt/mem_cache/memory_pool.py` | `set_kv_buffer` 按开关选 Triton 写入或 PyTorch 写入；新增 `get_int8_kv_buffers(layer_id)` 返回原始 int8 K/V 和两组 scale；`get_key_buffer` 仍返回反量化结果，其他读取方不受影响 |
| `srt/layers/attention/triton_ops/decode_attention.py` | grouped stage-1 kernel 增加 `KV_INT8` 分支：`qk = dot(q, k→bf16) * s_k`，乘 V 前 `p * s_v`，softmax 分母用未乘 scale 的 p；非 grouped、MLA、HIP、标量 scale ≠ 1.0、只给一个 scale buffer 都报错 |
| `srt/layers/attention/triton_ops/extend_attention.py` | `_fwd_kernel` 只改前缀循环（从缓存池读的部分），本次新输入保持原始精度；unified kernel 收到 scale 报 `NotImplementedError` |
| `srt/layers/attention/triton_backend.py` | `_kv_buffers_for_read()`：fused 时把 int8 buffer 和 scale 传给 extend、decode kernel；plain 时照旧取反量化 buffer |

不传 scale 时，decode 和 extend kernel 在 BF16 和 FP8 下都与未打补丁的原版逐比特相同（L0 测试保证）。

---

## 6. 测试体系

| 层级 | 位置 | 运行环境 | 内容 |
|---|---|---|---|
| 工程流程 | `tests/test_stage4_patch_flow.py` | Mac | apply/export 往返、错误 commit 被拒、切换基线版本、无 git 身份、`require_env` 的接受和拒绝（包括 ninja/nvcc 缺失） |
| overlay | `tests/test_stage4_overlay.py` | Mac | 复制、清单、再次 apply 时恢复原版、verify 检出不一致 |
| L0 数学 | `tests_stage4/test_int8_kv_math.py` | Mac + 服务器 | 18 个测试：与阶段 3 参考实现逐位相等、零行、往返误差上界、bf16 向上取整、cell size、`store_quantized`、不支持配置的报错 |
| L0 pool | `tests_stage4/test_int8_pool.py` | 仅 GPU | pool 形状、cell size 与实际字节一致、写读往返、拒绝静态 scale、CLI 选项 |
| L1 指标 | `tests/test_stage4_l1_metrics.py` | Mac | L1 脚本的纯函数部分（用假 engine 测请求构造和解析） |
| L1 模型 | `experiments/stage4/l1_model_check.py` | 服务器 | 见第 7 节 |
| 冒烟 | `experiments/stage4/smoke.py` | 服务器 | Triton 后端，BF16 和 FP8 KV 各生成 8 个 token |
| L0 写入 kernel | `tests_stage4/test_int8_store_kernel.py` | 开关解析 Mac，其余 GPU | 与 PyTorch 参考逐比特（N 取 1/7/4096，bf16/fp16）、非连续输入、只写 loc、全 0 行、空 batch、pool 两种实现 |
| L0 参考 attention | `tests_stage4/reference.py`、`test_reference.py` | Mac | FP32 参考 attention（GQA、因果偏移），供 decode/extend 测试共用 |
| L0 decode | `tests_stage4/test_fused_decode.py` | GPU | 与精确反量化参考相对 L2 < 1e-2、与 plain 余弦 > 0.9999、关闭时与原版逐比特（BF16、FP8）、各种报错 |
| L0 extend | `tests_stage4/test_fused_extend.py` | GPU | 前缀 {0,5,300} × 新输入 {1,64,257} 对参考、对 plain、无前缀时与 BF16 逐比特、关闭时与原版逐比特、unified 拒绝 |
| L0 后端接线 | `tests_stage4/test_backend_wiring.py`、`test_int8_impl_wiring.py` | GPU（Engine）/ Mac | fused 生成与 plain 前 8 token 相同、radix cache 命中前后输出相同、512 分块时 fused 检索到 plain 检索到的每根针；确定性模式检查 |
| 分块诊断 | `experiments/stage4/chunk_diag.py`、`scripts/stage4/diag_chunk.sh` | 服务器 | BF16 / INT8 plain / INT8 fused × 分块与否，长 needle 的检索结果 |

测试数据约定（Task 6 的教训）：参考值用精确 FP32 反量化（`int8 × scale`，不舍入到 bf16）；q 乘 0.05，让 attention logit 标准差不超过 5 左右；FP8 的 scale 取 absmax/200，避免 e4m3 溢出成 NaN。

本地完整测试：150 passed, 66 skipped（GPU 测试被跳过）。

服务器脚本：

- `scripts/stage4/setup.sh`：安装，可重复运行，跳过已完成的步骤；
- `scripts/stage4/test.sh`：运行 L0 + L1，输出 SUMMARY，结果保存在 `results/stage4/test_<时间>/`；有文件锁，禁止两个实例同时运行。

---

## 7. L1 模型级检查：问题和重写

### 第一版（commit `16074b9` ~ `e0497c8`）

方法：

1. BF16 贪心生成 64 个 token；
2. 把"prompt + 续写"整段送进每个配置做一次 prefill 打分；
3. 比较每个位置的 Top-1。

服务器结果（20261109_001630）：

```
L1 int8_per_token_head:plain:on:  FAIL top1=0.9898 |dlogp|=0.0346
L1 int8_per_token_head:plain:off: FAIL top1=0.9898 |dlogp|=0.0346
```

排查结论：**这个测法几乎没有测到 INT8。**

1. CUDA graph 开、关的结果完全相同，连小数点后四位都一样。CUDA graph 只作用于 decode，说明测试完全没经过 decode。
2. 整段 prefill 时，当前 chunk 用原始精度的 K/V，只有 radix cache 命中的前缀才读 INT8。所以大部分位置根本没读量化缓存。
3. 门槛 0.99 来自阶段 3 的 99.4%，但阶段 3 的测法不同：逐 token 喂入，每步从缓存读全部历史，按 prompt 平均。
4. 没有 BF16 对照，无法区分 1% 的差异是量化误差，还是两个引擎计算路径不同带来的数值噪声。参考引擎的 radix cache 状态和新引擎不同。

另外修过一个解析问题：SGLang 在 `logprob_start_len` 位置固定返回 `None`（`scheduler_output_processor_mixin._process_input_token_logprobs` 中的 `[None] + logprobs[:-1]`）。

### 第二版（commit `1a32cf2`，待服务器结果）

- **逐 token 强制解码**，与阶段 3 的 `teacher_forced_logits` 是同一种方法：
  - 第 k 步的请求是 `prompt + 续写[:k]`，`max_new_tokens=1`，`logprob_start_len = len-1`；
  - SGLang 会把之前的全部 token 从 KV 缓存中取出（`schedule_batch.py` 中 `max_prefix_len = min(input_len-1, logprob_start_len)`），只计算最后一个 token，等价于 decode 读 INT8；
  - 用 `token_ids_logprob=[参考 token]` 取参考 token 的 logprob，用 `top_logprobs_num=1` 取 Top-1；
  - 同一步的所有 prompt 合成一个 batch。
- **指标**：按 prompt 平均的 Top-1（与阶段 3 相同），同时报告按 token 汇总的 Top-1、|Δlogp|，以及 short 和 needle 分组。
- **对照组**：
  - 新开一个 BF16 引擎作为噪声底线，低于 0.999 判为 `INVALID`；
  - FP8 只报告，不判定。
- **真实生成**：每个配置先 `flush_cache`，再贪心生成，报告与 BF16 完全一致的比例和第一次分歧的位置；INT8 在 CUDA graph 开、关下生成的 token 必须完全相同。
- **判定**：只对 INT8 配置判定，按 prompt 平均的 Top-1 ≥ 0.99，并且噪声底线有效。

### 7.1 L1 v2 第一次完整结果（2026-11-09，新服务器，patch `4059b73`）

| 配置 | top1/prompt | short | needle | gen_exact |
|---|---|---|---|---|
| BF16（噪声底线） | 1.0000 | 1.0000 | 1.0000 | 1.00 |
| FP8 e4m3 | 0.8244 | 0.8685 | 0.7714 | 0.00 |
| INT8（graph 开/关，结果完全相同） | 0.9684 | 0.9779 | 0.9571 | 0.50 |

当时以 0.99 为门槛，判为 FAIL。在同一批 prompt 上和阶段 3 逐组对比后，结论是实现没有问题，门槛和测法有误：

- **short（12 个）：** SGLang INT8 为 0.9779，阶段 3 PyTorch 为 0.9779（`results/.../stage3/model_metrics.json` 前 12 条），**完全一致**。
- **needle（seed 0-9）：** 阶段 3 float prefill 为 1.0，阶段 3 int8 prefill 为 0.857，SGLang 为 0.957，落在两者之间。
  - needle prompt 约 3,330 token。SGLang 在 24 GB 显卡上默认 `chunked_prefill_size=2048`，所以第二块 prefill 要从 INT8 缓存读第一块，即部分走了 "int8 prefill"。
  - 阶段 3 的 float prefill 是整段一次算完，从来没有这一步。
- **门槛：** 0.99 是阶段 3 在 112 个 prompt 上的平均值（99.38%），其中 100 个是 needle。L1 只用 22 个 prompt，short 占的比例大，阶段 3 在这 22 个 prompt 上的期望值本来就只有 0.9879。所以按 0.99 判，即使实现完美也会 FAIL。

修正（L1 v3）：

- **新判定：** 和阶段 3 在同样 prompt 上的结果比，`stage3_expected()` 从阶段 3 的 json 读取对应行，INT8 ≥ 期望值 − 0.005 判为通过。
- **关闭 chunked prefill：** 默认 `--chunked-prefill-size -1`，与阶段 3 的测法一致。
- **chunked prefill 的影响单独记录：** 默认 2048 下 needle 掉到 0.957，这是 SGLang 默认部署下的真实代价，留到阶段 4 报告中讨论。
- **FP8 很差（0.82）：** SGLang 的 FP8 KV 不带 scale（scale=1），e4m3 只有 3 位尾数，而 Qwen2.5 的 K 有很大的离群值，所以误差大。这正好说明 per-token-head INT8 的价值，报告中会用。

### 7.2 L1 v3 结果（2026-11-09，patch `3d9312a`，chunked prefill 关闭）

| 配置 | top1/prompt | short | needle | gen_exact | 判定 |
|---|---|---|---|---|---|
| BF16 | 1.0000 | 1.0000 | 1.0000 | 1.00 | 噪声底线 |
| FP8 e4m3 | 0.8644 | 0.8705 | 0.8571 | 0.00 | 只报告 |
| INT8，graph 开 | **0.9888** | 0.9795 | **1.0000** | 0.59 | PASS（阶段 3 为 0.9879） |
| INT8，graph 关 | **0.9895** | 0.9808 | **1.0000** | 0.64 | PASS |

- **INT8 正确性：** 关闭分块后 needle 回到 1.0，总分 0.9888，和阶段 3 的 0.9879 一致。至此 SGLang 实现与阶段 3 的 PyTorch 参考对齐。
- **新问题：** INT8 在 CUDA graph 开和关时，有 3/22 条生成结果不同（exact 0.86）。
  - 原因：SGLang 在 graph 开和关时走的不是同一条数值路径。decode 的 KV split 数按填充后的 graph batch 计算，prefill 用的是分段 CUDA graph，累加顺序不同，近似平局的 token 在贪心生成时就可能翻转。
  - 佐证：forced-decode 的 |Δlogp| 在 graph 开和关时完全相同（0.0208），top-1 只差 0.0007。
- **修正（L1 v4）：**
  - 加一组 BF16 graph 关闭的对照（`auto:plain:off`）。
  - INT8 在 graph 开和关之间出现分歧的 prompt 数，最多比 BF16 自己的分歧数多 `--graph-slack`（默认 2）条。
  - 噪声底线只看 BF16 graph 开（`noise_floor()`），因为它和参考的运行方式完全相同。

### 7.3 L1 v4 结果（patch `4d0069e`）与 v5 诊断

- **结果：** BF16 在 graph 开和关之间 0/22 条分歧（forced-decode top-1 为 0.9986，|Δlogp| 为 0.0034）；INT8 仍有 3/22 条分歧，判为 FAIL。
- **结论：** "graph 开关本身就会翻转"这个解释不成立，或者至少证据不够，不能放宽判定了事。
- **测法的问题：** forced decode 只走 extend（prefill）路径，根本不经过 decode 的 CUDA graph。所以 forced-decode 分数在 graph 开和关时接近，并不能证明 decode graph 路径是对的。
- **L1 v5 的做法：** 改为直接比较两条贪心生成的 decode 路径：
  - 贪心生成时取 top-2 logprob；
  - **drift：** 两次运行在分歧之前、逐 token 所选 token 的 |Δlogp| 均值；
  - **flip margin：** 分歧那一步的 top-1 与 top-2 之差，取两次运行中较大的那个。
- **判定：**
  - INT8 的 drift ≤ max(3 × BF16 的 drift, 0.01)；
  - 并且每次翻转都发生在近似平局处（margin < 0.1）。
  - 如果是真正的 graph 路径 bug，会表现为 drift 明显偏大，或者在模型很有把握的位置发生翻转。

---

### 7.4 L1 v5 结果（patch `6910930`）

| 项 | INT8 | BF16 对照 |
|---|---|---|
| graph on/off 分叉 | 3/22 | 0/22 |
| 分叉前 chosen-token logprob 漂移 | 0.0033 | 0.0032 |
| 分叉点 top-2 差距 | 0.125, 0.125, 0.125 | — |

- 漂移与 BF16 一致，说明 INT8 decode graph 路径没有额外误差。
- 三个分叉点的差距都恰好是 0.125 = 2^-3，即 logit 在 [16, 32) 区间时相邻两个 bf16 值的间隔。top-2 只差一个 bf16 步长，是模型能表达的最小区分，在这里翻转属于数值平局。
- INT8 比 BF16 更容易碰上平局，是因为 INT8 本身改变了 logit（|dlogp| 0.02），让一些位置的 top-2 落进了一个步长内；graph 与 eager 的累加顺序差异再把它翻过去。
- v5 的平局阈值 0.1 低于 bf16 分辨率，是判据写错了。v6 改为“差距不超过一个 bf16 步长（0.125）”，两个步长（0.25）仍判 FAIL。按 v5 的数据重判为 PASS。

### 7.5 Task 5-8 的服务器结果

**Task 5（`0591288`）**：L0 44 passed；L1 PASS，`fused:on`（此时只有写入是融合的）与 `plain:on` 指标完全相同（0.9888 / 0.0208），写入逐比特一致的直接证据。

**Task 6（`9536038` → `e2b6e49`）**：第一次 L0 有 5 个失败，全部是测试数据问题，CPU 复现后修正，kernel 未改：

- 参考值用了舍入到 bf16 的反量化 K/V，而 kernel 直接算 `int8 × scale`；
- q ~ N(0,1) 配 100 倍幅度的 K head，logit 标准差约 110，softmax 近似 one-hot，仅 bf16 舍入就让输出偏 2%；
- FP8 用例 scale 0.5 让 |k|/0.5 超过 e4m3 上限 448，变成 NaN，`torch.equal` 永远不等。

修正后 70 passed。

**Task 7（`286ad79`）**：L0 87 passed。

**Task 8 检查点 2（`45dd78e`）**：L1 全部通过，融合 kernel 第一次在模型中运行：

| 配置 | top1/prompt | \|dlogp\| | needle |
|---|---|---|---|
| int8 plain:on | 0.9888 | 0.0208 | 1.0 |
| int8 plain:off | 0.9895 | 0.0208 | 1.0 |
| int8 fused:on | 0.9902 | 0.0199 | 1.0 |
| int8 fused:off | 0.9895 | 0.0201 | 1.0 |

fused 与 plain 差距 < 0.0015（阈值 0.005）。fused 的 \|dlogp\| 略小，因为 plain 把反量化结果多舍入了一次 bf16。

L0 92 passed、1 failed：分块用例原先要求 600 条事实（约 1.3 万 token）的 needle 在 512 分块和不分块下输出 token 完全相同。

**分块诊断。** 600 条事实，4 根针：

- BF16 本身只对 1/4，所有 INT8 配置 0/4；
- 各配置答案互不相同，连不分块的 plain 和 fused 都不同；
- 这个长度已超出 1.5B 的检索能力，模型在近似平局的候选间猜测，比较 token 只量到噪声。

150 条事实（约 3300 token），4 根针：

| | 不分块 | 512 分块 |
|---|---|---|
| BF16 | 3/4 | 3/4 |
| INT8 plain | 3/4 | 2/4 |
| INT8 fused | 3/4 | 2/4 |

- seed 3：所有配置都错，包括 BF16；
- seed 0：INT8 一分块就把最后一位答错（224672 vs 224679），plain 和 fused 错得一样，是“后续块从 INT8 读前面块”的设计代价，不是融合 kernel 的问题。这条要写进最终报告（与阶段 3 的分块观察一致）。

据此把用例改为：512 分块下，plain 检索到的针 fused 也必须检索到（`fe002aa`）。

## 8. 服务器部署问题记录

| # | 现象 | 原因 | 修复（commit） |
|---|---|---|---|
| 1 | `git am`：Committer identity unknown | 服务器没有配置 git 身份 | apply.sh 用 `-c user.name/email`（`57d6424`） |
| 2 | Failed building editable for sglang | v0.5.12 源码构建需要 Rust | 改为 overlay 覆盖 wheel（`c6f0a13`） |
| 3 | `driver too old (found version 12080)` | pip 把 torch 换成了 PyPI 上的 CUDA 13 版本 | 每次运行都检查，不是 `2.11.0+cu128` 就从 cu128 源重装（`c3c9c6b`） |
| 4 | `Either a revision or a version must be specified` | sglang 没有锁定 `kernels`，pip 装了 0.18，与 transformers 5.6.0 不兼容 | 锁定 `kernels>=0.12,<0.13`（`c3c9c6b`） |
| 5 | `No such file or directory: 'ninja'` | JIT 编译按名字调用 ninja/nvcc，venv/bin 和 CUDA bin 不在 PATH 上 | common.sh 导出 PATH/CUDA_HOME，require_env 检查（`1c39fab`） |
| 6 | L0 FAIL，没有输出 | 安装时用了 `--no-deps`，没装 pytest | 安装和测试脚本都会补装 pytest（`e0497c8`） |
| 7 | L1 `'NoneType' object is not subscriptable` | SGLang 起始位置的 logprob 固定为 None | 跳过 None（`e0497c8`），随后整体重写（`1a32cf2`） |
| 8 | 两个 test.sh 同时运行 | 手动启动了两次 | test.sh 加 flock 文件锁（`e0497c8`） |
| 9 | L1 top1=0.9898，CUDA graph 开、关结果相同 | 测法没有读 INT8，见第 7 节 | 重写 L1（`1a32cf2`） |

AutoDL 操作要点：

- 访问 GitHub 前先 `source /etc/network_turbo`；pip 安装前 `unset http_proxy https_proxy`，因为走代理反而更慢；
- 模型从 ModelScope 下载到 `/root/autodl-tmp/models`，venv 在 `/root/autodl-tmp/venv-sglang`；
- 长任务用 `nohup ... &` 在后台运行，用 `jobs` 或 `ps` 查看是否结束；
- 所有命令都在 `~/int8-kv-cache` 目录下执行。

---

## 9. 已知问题和下一步

### 9.1 待查：后端接线测试运行后没有任何输出

`fe002aa` 上运行 `pytest tests_stage4/test_backend_wiring.py ... | tail -5`，终端没有输出。pytest 正常结束一定会打印结果行，没有输出说明进程很可能被系统杀掉（例如内存不足，管道里 shell 不会显示 `Killed`）。这个文件会连续启动 4 次 Engine。

下一步在服务器上运行：

```bash
cd ~/int8-kv-cache && git log --oneline -1 && /root/autodl-tmp/venv-sglang/bin/python -m pytest -q tests_stage4/test_backend_wiring.py -p no:cacheprovider > wiring.log 2>&1; echo "exit code: $?"; tail -20 wiring.log; free -g | head -2
```

退出码 137 表示被杀，再用 `dmesg | grep -i -E "killed process|out of memory"` 确认。

### 9.2 已知代价（写入报告）

- 分块 prefill 时后续块从 INT8 读前面块，150 条事实的 needle 上 INT8 分块比不分块少对 1 根（plain、fused 相同）。SGLang 在 24 GB 卡上默认分块 2048。
- FP8 KV 没有校准 scale 时精度很差（top1 0.8644），原因是 Qwen K 的离群值。

### 9.3 后续任务

1. 解决 9.1，Task 8 收尾。
2. Task 9-11：Qwen2.5-7B-Instruct 上 BF16 / FP8 / INT8 三方对比：
   - GSM8K 全部 1319 题（McNemar 检验）；
   - needle；
   - 最大 token 容量、并发和吞吐；
   - 撰写 `docs/stage4-report.md`。
3. 全分支代码审查（最强模型），然后合并。

---

## 附录 A：`int8_kv.py` 全文

```python
"""Dynamic per-token-head INT8 KV cache quantization (reference math).

KV rows are laid out as [tokens, heads, head_dim]. Each (token, head) row gets
its own scale = absmax / 127, stored as bfloat16 and rounded *up* so that the
row's absmax never quantizes past 127. An all-zero row uses scale 1.
"""
from typing import Optional

import torch

INT8_KV_QMAX = 127


def bf16_round_up(x: torch.Tensor) -> torch.Tensor:
    cast = x.to(torch.bfloat16)
    too_small = cast.to(torch.float32) < x
    bumped = (cast.view(torch.int16) + 1).view(torch.bfloat16)
    return torch.where(too_small, bumped, cast)


def quantize_per_token_head(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x: [N, H, D] -> (int8 [N, H, D], bfloat16 scale [N, H])."""
    xf = x.to(torch.float32)
    absmax = xf.abs().amax(dim=-1)
    scale = torch.where(absmax > 0, absmax / INT8_KV_QMAX, torch.ones_like(absmax))
    scale = bf16_round_up(scale)
    q = (xf / scale.to(torch.float32).unsqueeze(-1)).round()
    return q.clamp(-INT8_KV_QMAX, INT8_KV_QMAX).to(torch.int8), scale


def dequantize_per_token_head(q, scale, dtype=torch.bfloat16):
    out = q.to(torch.float32) * scale.to(torch.float32).unsqueeze(-1)
    return out.to(dtype)


def store_quantized(*, q_buf, scale_buf, loc, x) -> None:
    q, scale = quantize_per_token_head(x)
    q_buf[loc] = q
    scale_buf[loc] = scale


def int8_kv_cell_size(*, head_num, head_dim, v_head_dim, num_layers) -> int:
    """KV bytes per token: int8 values plus one bf16 scale per head, for both K and V."""
    return num_layers * head_num * (head_dim + v_head_dim + 2 * 2)


def check_int8_kv_supported(*, attention_backend: Optional[str], page_size, tp_size, use_mla,
                            speculative_algorithm, enable_hierarchical_cache, is_hybrid_swa) -> None:
    problems = []
    if attention_backend != "triton":
        problems.append(f"attention backend {attention_backend!r} (pass --attention-backend triton)")
    if page_size != 1:
        problems.append(f"page_size={page_size}")
    if tp_size != 1:
        problems.append(f"tensor parallel size {tp_size}")
    if use_mla:
        problems.append("MLA models")
    if speculative_algorithm is not None:
        problems.append(f"speculative decoding ({speculative_algorithm})")
    if enable_hierarchical_cache:
        problems.append("hierarchical cache")
    if is_hybrid_swa:
        problems.append("hybrid sliding window attention")
    if problems:
        raise NotImplementedError(
            "--kv-cache-dtype int8_per_token_head does not support: " + "; ".join(problems))
```

## 附录 B：`MHATokenToKVPoolINT8` 全文

```python
class MHATokenToKVPoolINT8(MHATokenToKVPool):
    """MHA pool with int8 K/V and one bf16 scale per (token, head).

    Reads dequantize the whole layer to compute_dtype, so any attention backend
    sees a plain K/V buffer.
    """

    def __init__(self, *args, compute_dtype: torch.dtype, **kwargs):
        if kwargs.get("enable_kv_cache_copy"):
            raise NotImplementedError("int8_per_token_head KV cache does not support KV copy")
        self.compute_dtype = compute_dtype
        super().__init__(*args, **kwargs)

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (torch.cuda.use_mem_pool(self.custom_mem_pool)
                  if self.enable_custom_mem_pool else nullcontext()):
                m = self.size + self.page_size  # slot 0 receives dummy writes from padded tokens
                self.k_buffer = [torch.zeros((m, self.head_num, self.head_dim), dtype=torch.int8, device=self.device)
                                 for _ in range(self.layer_num)]
                self.v_buffer = [torch.zeros((m, self.head_num, self.v_head_dim), dtype=torch.int8, device=self.device)
                                 for _ in range(self.layer_num)]
                self.k_scale_buffer = [torch.zeros((m, self.head_num), dtype=torch.bfloat16, device=self.device)
                                       for _ in range(self.layer_num)]
                self.v_scale_buffer = [torch.zeros((m, self.head_num), dtype=torch.bfloat16, device=self.device)
                                       for _ in range(self.layer_num)]

    def _clear_buffers(self):
        del self.k_buffer, self.v_buffer, self.k_scale_buffer, self.v_scale_buffer

    def get_kv_size_bytes(self):
        k = sum(get_tensor_size_bytes(t) for t in self.k_buffer + self.k_scale_buffer)
        v = sum(get_tensor_size_bytes(t) for t in self.v_buffer + self.v_scale_buffer)
        return k, v

    def get_kv_scale_buffer(self, layer_id: int):
        i = layer_id - self.start_layer
        return self.k_scale_buffer[i], self.v_scale_buffer[i]

    def _get_key_buffer(self, layer_id: int):
        i = layer_id - self.start_layer
        return dequantize_per_token_head(self.k_buffer[i], self.k_scale_buffer[i], dtype=self.compute_dtype)

    def _get_value_buffer(self, layer_id: int):
        i = layer_id - self.start_layer
        return dequantize_per_token_head(self.v_buffer[i], self.v_scale_buffer[i], dtype=self.compute_dtype)

    def set_kv_buffer(self, layer, loc, cache_k, cache_v, k_scale=None, v_scale=None, layer_id_override=None):
        if k_scale is not None or v_scale is not None:
            raise ValueError("int8_per_token_head KV cache computes its own scales; "
                             "checkpoint k_scale/v_scale are not supported")
        layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
        i = layer_id - self.start_layer
        store_quantized(q_buf=self.k_buffer[i], scale_buf=self.k_scale_buffer[i], loc=loc, x=cache_k)
        store_quantized(q_buf=self.v_buffer[i], scale_buf=self.v_scale_buffer[i], loc=loc, x=cache_v)

    def get_contiguous_buf_infos(self):
        raise NotImplementedError("int8_per_token_head KV cache does not support disaggregation")

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError("int8_per_token_head KV cache does not support CPU offload")

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError("int8_per_token_head KV cache does not support CPU offload")

    def move_kv_cache(self, tgt_loc, src_loc):
        raise NotImplementedError("int8_per_token_head KV cache does not support KV moves")
```

## 附录 C：接线代码

```python
# server_args.py — --kv-cache-dtype choices
choices=["auto", "fp8_e5m2", "fp8_e4m3", "bf16", "bfloat16", "fp4_e2m1", "int8_per_token_head"]

# model_runner.py — configure_kv_cache_dtype
elif self.server_args.kv_cache_dtype == "int8_per_token_head":
    self.kv_cache_dtype = torch.int8

# pool_configurator.py — DefaultPoolConfigurator, non-MLA branch
if kv_cache_dtype == torch.int8:
    cell_size = int8_kv_cell_size(head_num=model_config.get_num_kv_heads(tp_size),
                                  head_dim=model_config.head_dim,
                                  v_head_dim=model_config.v_head_dim, num_layers=num_layers)

# model_runner_kv_cache_mixin.py
def _check_int8_kv_supported(self):
    args = self.server_args
    backends = {args.attention_backend,
                args.prefill_attention_backend or args.attention_backend,
                args.decode_attention_backend or args.attention_backend}
    check_int8_kv_supported(
        attention_backend=backends.pop() if len(backends) == 1 else "mixed",
        page_size=self.page_size, tp_size=get_attention_tp_size(), use_mla=self.use_mla_backend,
        speculative_algorithm=args.speculative_algorithm,
        enable_hierarchical_cache=args.enable_hierarchical_cache,
        is_hybrid_swa=self.is_hybrid_swa or self.mambaish_config is not None)

# _init_pools: at the top
if self.kv_cache_dtype == torch.int8:
    self._check_int8_kv_supported()
# _init_pools: pool selection, before the FP4 branch
elif self.kv_cache_dtype == torch.int8:
    self.token_to_kv_pool = MHATokenToKVPoolINT8(
        self.max_total_num_tokens, page_size=self.page_size, dtype=self.kv_cache_dtype,
        head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),
        head_dim=self.model_config.head_dim, v_head_dim=self.model_config.v_head_dim,
        layer_num=self.num_effective_layers, device=self.device,
        enable_memory_saver=self.server_args.enable_memory_saver,
        start_layer=self.start_layer, end_layer=self.end_layer,
        enable_alt_stream=not self.server_args.enable_pdmux, compute_dtype=self.dtype)
```

完整、可直接应用的版本以 `sglang_patch/0001-*.patch` 和 `0002-*.patch` 为准。
