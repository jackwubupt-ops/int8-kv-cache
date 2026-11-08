# 阶段 4：SGLang INT8 KV Cache 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 SGLang 中加入 `--kv-cache-dtype int8_per_token_head`（先做简单版，再做融合 Triton kernel 版），并在 Qwen2.5-7B-Instruct 上和 BF16、FP8 对比精度与性能。

**Architecture:** 修改以补丁形式保存在本仓库的 `sglang_patch/` 中，针对一个固定的 SGLang 基线版本。简单版通过 SGLang 的 KV 量化插件接口（`KVCacheQuantMethodBase`）注册一个新方法，读取时把整层恢复成 BF16。融合版在 Triton 的 decode grouped kernel 和 extend kernel 中直接读取 int8 和 scale。两种实现在同一份代码里，用环境变量 `SGLANG_INT8_KV_IMPL=plain|fused` 切换。

**Tech Stack:** SGLang（基线版本见 Task 0）、PyTorch、Triton、pytest、ModelScope；服务器为 AutoDL RTX 4090 D。

**Spec:** `docs/superpowers/specs/2026-11-08-stage4-sglang-int8-kv-design.md`

## Global Constraints

- 量化规则必须和 `int8kv` 一致：`scale = absmax / 127`，按 (token, head) 计算；absmax 为 0 时 scale 取 1；scale 用 BF16 存，并且只能向上取整；`q = clamp(round(x / scale), -127, 127)`，舍入方式为 round-half-even。
- 每层缓冲区：`k_buffer` 和 `v_buffer` 的形状是 `[size + page_size, head_num, head_dim]`，类型 `torch.int8`；`k_scale_buffer` 和 `v_scale_buffer` 的形状是 `[size + page_size, head_num]`，类型 `torch.bfloat16`。
- 每个 token 每层的字节数：`head_num * (2 * head_dim + 4)`。
- 只支持：MHA 结构、Triton 后端、page_size 为 1、TP 为 1。其余组合（MLA、推测解码、hicache、sliding window、非 grouped decode 路径）直接抛出 `NotImplementedError` 或 `ValueError`，不静默回退。
- 不打开开关时，BF16 和 FP8 的 kernel 输出必须和打补丁前逐比特一致。
- prefill 语义：本次新输入的 token 用原始精度，缓存中的前缀读 INT8。
- 服务器上的所有命令都通过 `scripts/stage4/*.sh` 执行，在后台运行，结果写到 `results/stage4/`。模型放在 `/root/autodl-tmp/models/`。
- Mac 上没有 CUDA，`tests_stage4/` 中依赖 GPU 或 sglang 的测试在 Mac 上自动跳过；只依赖 PyTorch 的数学测试在 Mac 上也要运行。

## Review Focus

1. **radix cache 命中前缀**：第二个请求和第一个共享前缀时，extend kernel 会从 INT8 缓存读前缀，结果应和没有命中时一致（在误差范围内）。→ Task 8 的 `test_prefix_cache_hit_matches_cold_request`。
2. **长提示词被 chunked prefill 切成多段**：后面的段要从 INT8 缓存读前面段的 K/V。→ Task 8 的 `test_chunked_prefill_long_prompt`。
3. **某个 (token, head) 的 K 或 V 全是 0**（比如 padding、CUDA Graph 占位写入）：不能出现除以 0 或 NaN。→ Task 2 和 Task 5 的 `*_zero_rows` 测试。
4. **容量估算和实际分配不一致**：`compute_cell_size` 算错会导致运行时 OOM，或者容量数字虚高。→ Task 3 的 `test_cell_size_matches_allocated_bytes`。
5. **CUDA Graph 下 decode 的结果和不用 CUDA Graph 时不同**（captured kernel 拿到的 scale 指针不对）。→ Task 8 的 L1 检查在开、关 CUDA Graph 下各跑一次。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `sglang_patch/BASE_VERSION` | 基线版本号和 commit |
| `sglang_patch/apply.sh` | 克隆基线版本到 `third_party/sglang` 并依次 `git am` 补丁；再克隆一份不打补丁的到 `third_party/sglang-pristine` |
| `sglang_patch/export.sh` | 把 `third_party/sglang` 上基线之后的提交导出成补丁 |
| `sglang_patch/000N-*.patch` | 补丁本身 |
| SGLang 内：`srt/layers/quantization/int8_kv.py`（新建） | 纯 PyTorch 的量化数学，不 import sglang 的其它模块 |
| SGLang 内：`kernels/ops/quantization/int8_kv_store.py`（新建） | 融合写入的 Triton kernel |
| SGLang 内：`srt/layers/quantization/fp4_kv_cache_quant_method.py` | 注册 INT8 方法和读取方式 |
| SGLang 内：`srt/mem_cache/kv_cache_dtype.py`、`srt/mem_cache/memory_pool.py` | dtype 解析、scale 缓冲区访问 |
| SGLang 内：`kernels/ops/attention/decode_attention.py`、`extend_attention.py` | 融合 kernel |
| SGLang 内：`srt/layers/attention/triton_backend.py` | 把 scale 传给 kernel，并做组合检查 |
| `tests_stage4/` | L0 测试和公共夹具 |
| `experiments/stage4/` | L1 检查、精度评测、性能测试、汇总 |
| `scripts/stage4/` | `setup.sh`、`test.sh`、`eval.sh`、`bench.sh` |

以下所有 SGLang 路径都相对于 `third_party/sglang/python/sglang/`，以 v0.5.23 为准。如果 Task 0 选了别的版本，Task 1 中会核对这些路径。

---

### Task 0：确定基线版本（需要你在服务器上执行）

**Files:**
- Create: `sglang_patch/BASE_VERSION`
- Modify: `docs/superpowers/specs/2026-11-08-stage4-sglang-int8-kv-design.md`（4.1 节）

- [ ] **Step 1：在服务器上运行，把输出发回来**

Run: `nvidia-smi --query-gpu=driver_version,name --format=csv,noheader && nvidia-smi | head -4`
Expected: 一行驱动版本号，以及 `CUDA Version: xx.x`

- [ ] **Step 2：确定版本**

驱动 ≥ 580（支持 CUDA 13）时用 `v0.5.23`。否则查询 PyPI，找到最后一个依赖的 `torch` 为 cu12x 版本的 sglang 正式版。

- [ ] **Step 3：把结果写进 `sglang_patch/BASE_VERSION`**（两行：`version=v0.5.23`、`commit=<sha>`），并回填设计文档 4.1 节

- [ ] **Step 4：Commit**

```bash
git add sglang_patch/BASE_VERSION docs/superpowers/specs/2026-11-08-stage4-sglang-int8-kv-design.md
git commit -m "chore: pin SGLang base version for stage 4"
```

### Task 1：补丁流程和服务器安装脚本

**Files:**
- Create: `sglang_patch/apply.sh`、`sglang_patch/export.sh`、`scripts/stage4/setup.sh`、`scripts/stage4/common.sh`
- Modify: `.gitignore`（加 `third_party/`）
- Test: `tests/test_stage4_patch_flow.py`（Mac 上运行）

**Interfaces:**
- Produces:
  - `sglang_patch/apply.sh [target_dir]`：默认目标是 `third_party/`；结果为 `third_party/sglang`（打了补丁）和 `third_party/sglang-pristine`（未打补丁），两者都位于 `BASE_VERSION` 指定的 commit。
  - `scripts/stage4/common.sh` 导出 `STAGE4_VENV=/root/autodl-tmp/venv-sglang`、`MODELS=/root/autodl-tmp/models` 和函数 `require_env`：检查 venv 存在，且 `python -c "import sglang"` 加载的是 `third_party/sglang`；不满足时 `exit 1`。

- [ ] **Step 1：写失败测试** `test_apply_then_export_round_trip`：在临时目录里用 `git init` 建一个假的"基线仓库"，打 tag，加一个提交；`export.sh` 导出补丁，`apply.sh` 克隆后再打上补丁，断言得到的文件内容和原提交一致。基线仓库地址用环境变量 `SGLANG_REPO_URL` 覆盖，默认是 `https://github.com/sgl-project/sglang.git`。

- [ ] **Step 2：运行，确认失败** `.venv/bin/python -m pytest tests/test_stage4_patch_flow.py -q`，Expected：FAIL（脚本不存在）

- [ ] **Step 3：实现 `apply.sh` 和 `export.sh`**（`set -euo pipefail`；用 `git clone --depth 1 --branch <version>`；`git am --3way`；`export.sh` 执行 `git format-patch <base>..HEAD -o sglang_patch/`，导出前先删除旧的 `0*.patch`）

- [ ] **Step 4：运行，确认通过**

- [ ] **Step 5：写 `scripts/stage4/setup.sh`**，依次执行：
  1. 打印驱动版本；
  2. 建 venv：`python3 -m venv $STAGE4_VENV`；
  3. `pip install "sglang==<version>"`；
  4. `apply.sh`；
  5. `pip install -e third_party/sglang/python --no-deps`；
  6. 安装本项目：`pip install -e ".[dev]" --no-deps`，再单独装 `modelscope`；
  7. 如果模型目录不存在，用 `modelscope download` 下载 `Qwen/Qwen2.5-1.5B-Instruct` 和 `Qwen/Qwen2.5-7B-Instruct`；
  8. 最后打印 `python -c "import sglang, torch; print(sglang.__file__, torch.__version__, torch.cuda.get_device_name(0))"`。

  用 `bash -n` 检查语法；再用假的 python（同 `run_stages.sh` 的测试方法）验证 `require_env` 在环境不对时 `exit 1`。

- [ ] **Step 6：Commit** `feat(stage4): patch flow and server setup script`

### Task 2：量化数学模块（纯 PyTorch）

**Files:**
- Create（SGLang 内）：`srt/layers/quantization/int8_kv.py`
- Test: `tests_stage4/test_int8_kv_math.py`、`tests_stage4/conftest.py`

**Interfaces:**
- Produces（`int8_kv.py` 只 import torch）：
  - `INT8_KV_QMAX = 127`
  - `quantize_per_token_head(x: Tensor[N, H, D]) -> tuple[Tensor[N, H, D] int8, Tensor[N, H] bfloat16]`
  - `dequantize_per_token_head(q: Tensor[N, H, D] int8, scale: Tensor[N, H] bf16, dtype=torch.bfloat16) -> Tensor[N, H, D]`
  - `bf16_round_up(x: Tensor fp32) -> Tensor bf16`：转成 BF16，结果小于原值时取下一个可表示的值；后面 Triton 实现照这个规则写。
- `conftest.py` 提供：
  - `load_sglang_file(relpath)`：用 `importlib.util.spec_from_file_location` 按文件路径加载 `third_party/sglang/python/sglang/<relpath>`，Mac 上无需 import sglang 也能测；
  - 标记 `needs_gpu`：没有 CUDA 或 import sglang 失败时跳过。

- [ ] **Step 1：写失败测试**
  - `test_matches_int8kv_reference`：随机 `x[N=37, H=4, D=128]`，数值乘以 `[1, 50, 300]` 制造大范围；把 `int8kv.quant` 的结果转置到 `[B, H, T, D]` 布局后对比，断言 `q` 和 `scale` 都 `torch.equal`。
  - `test_zero_rows`：某个 (token, head) 全为 0 时，scale 等于 1，`q` 全为 0，没有 NaN。
  - `test_round_trip_error_bound`：`|dequant - x| <= scale / 2 + 1e-6`，逐元素检查。
  - `test_bf16_round_up_never_rounds_down`：一万个随机正 fp32 值，`bf16_round_up(x).float() >= x`，并且结果和 `int8kv.quant.round_scale_up(x, torch.bfloat16)` 完全一致。

- [ ] **Step 2：运行，确认失败** `.venv/bin/python -m pytest tests_stage4/test_int8_kv_math.py -q`

- [ ] **Step 3：实现 `int8_kv.py`**（`bf16_round_up` 的做法：先按就近舍入转成 BF16；如果小于原值，把 BF16 的位模式当作 int16 加 1；只处理正数）

- [ ] **Step 4：运行，确认通过**

- [ ] **Step 5：在 `third_party/sglang` 中提交，执行 `sglang_patch/export.sh`，在本仓库提交** `feat(stage4): int8 per-token-head quant math`

### Task 3：简单版（插件接口注册）

**Files:**
- Modify（SGLang 内）：`srt/layers/quantization/fp4_kv_cache_quant_method.py`、`srt/mem_cache/kv_cache_dtype.py`、`srt/mem_cache/kv_cache_configurator.py`（`_build_mha_quant_method`）、`srt/arg_groups/fields/model.py`（`--kv-cache-dtype` 的可选值）；如果读取方式需要，也改 `srt/layers/attention/triton_backend.py`
- Test: `tests_stage4/test_plain_method.py`（`needs_gpu`）

**Interfaces:**
- Consumes：Task 2 的 `quantize_per_token_head` 和 `dequantize_per_token_head`。
- Produces：
  - `Int8PerTokenHeadKVCacheMethod(KVCacheQuantMethodBase)`，`name = "int8_per_token_head"`，已注册到 `KV_CACHE_QUANT_REGISTRY`；
  - 在 `KV_CACHE_ATTENTION_ACCESS_REGISTRY` 中登记 PLAIN 读取（Triton 和 torch_native，prefill 和 decode），storage 为 `torch.int8`，attention 用 `torch.bfloat16`；
  - `configure_kv_cache_dtype("int8_per_token_head", ...)` 返回 `("int8_per_token_head", torch.int8)`。

- [ ] **Step 1：写失败测试**
  - `test_create_buffers_shapes`：`size=10, head_num=4, head_dim=128, layer_num=2`；K/V 为 `(10, 4, 128)` int8，scale 为 `(10, 4)` bf16，各有 2 层。
  - `test_cell_size_matches_allocated_bytes`：`compute_cell_size(4, 128, 2, 10)` 等于 `create_buffers(...)` 返回的所有缓冲区实际字节数之和，也等于 `10 * 2 * 4 * (2 * 128 + 4)`。
  - `test_quantize_and_store_round_trip`：在 `loc=[3, 7]` 写入，读回并恢复精度，误差 ≤ scale/2；其它位置保持为 0。
  - `test_unsupported_backend_rejected`：后端为 `flashinfer` 时，解析读取方式会报错，错误信息里包含 `int8_per_token_head`。
  - `test_dtype_parse`：`configure_kv_cache_dtype` 对 `"int8_per_token_head"` 返回 `torch.int8`；`resolve_kv_cache_quant("int8_per_token_head")` 返回这个名字。

- [ ] **Step 2：在 Mac 上确认 collect 时被跳过且没有 import 错误**

- [ ] **Step 3：实现。** 照着 `FP4MXBlock16KVCacheMethod` 的写法；存储类型如果被写死成 `uint8`，就给 `_plain` 加一个 `storage_dtype` 参数。在 `triton_backend` 初始化时检查：如果是 INT8 且（MLA、推测解码、hicache、sliding window、page_size ≠ 1、TP > 1）中任一成立，就抛出 `NotImplementedError("int8_per_token_head KV cache does not support <feature>")`。

- [ ] **Step 4：导出补丁，提交** `feat(stage4): plain int8_per_token_head KV cache method`

### Task 4：L1 模型级检查脚本和第一次服务器测试

**Files:**
- Create: `experiments/stage4/l1_model_check.py`、`scripts/stage4/test.sh`
- Test: `tests/test_stage4_l1_metrics.py`（Mac 上运行，测纯函数）

**Interfaces:**
- Produces（`experiments/stage4/l1_model_check.py`）：
  - `compare_token_logprobs(ref: list[list[tuple[float, int]]], other: list[list[tuple[float, int]]]) -> dict`，返回 `{"top1_agreement", "mean_abs_logprob_diff", "n_tokens"}`。输入是每个提示词、每个位置的 (chosen_logprob, top1_token_id)。
  - 命令行参数：`--model PATH --kv-dtype {auto,fp8_e4m3,int8_per_token_head} --impl {plain,fused} --cuda-graph {on,off} --needles N --out PATH`。
  - 流程：用 BF16 Engine（`attention_backend="triton"`）对阶段 3 的 `SHORT_PROMPTS` 和 needle 提示词贪心生成 64 个 token；然后对每种待测配置分别新建 Engine，把"提示词 + 生成内容"作为输入，设置 `return_logprob=True`、`logprob_start_len=提示词长度`、`top_logprobs_num=1`，得到每个位置的 logprob；和 BF16 自己做同样的计算，进行对比。每个 Engine 用完后 `shutdown()`，再建下一个。
- `scripts/stage4/test.sh`：先 `require_env`，然后依次运行 `pytest tests_stage4 -q` 和 L1 检查（模型 1.5B；配置为 INT8 plain，CUDA Graph 开和关各一次）；全部输出写进 `results/stage4/test_<时间戳>.log`，最后打印一段汇总：每项 PASS/FAIL 和关键数字。

- [ ] **Step 1：写失败测试**
  - `test_compare_identical_is_perfect`：自己和自己比，`top1_agreement == 1.0`，`mean_abs_logprob_diff == 0.0`。
  - `test_compare_counts_mismatch`：4 个 token 中有 1 个 top1 不同，结果为 `0.75`。

- [ ] **Step 2：运行，确认失败；实现；运行，确认通过**

- [ ] **Step 3：提交并推送** `feat(stage4): L1 model check and server test script`

- [ ] **Step 4：服务器检查点 1（需要你执行）**：`git pull`，然后在后台运行 `bash scripts/stage4/setup.sh`；完成后在后台运行 `bash scripts/stage4/test.sh`，把汇总部分发回来。
  Expected：L0 全部 PASS；L1 中 INT8 plain 的 `top1_agreement ≥ 0.99`，CUDA Graph 开和关结果一致。没有达到就回到对应的 Task 排查，不进入 Task 5。

### Task 5：融合写入 kernel

**Files:**
- Create（SGLang 内）：`kernels/ops/quantization/int8_kv_store.py`
- Test: `tests_stage4/test_int8_store_kernel.py`（`needs_gpu`）

**Interfaces:**
- Produces：`int8_quantize_and_store(cache: Tensor[N, H, D] bf16|fp16, loc: Tensor[N] int64, q_buf: Tensor[S, H, D] int8, scale_buf: Tensor[S, H] bf16) -> None`。一个 program 处理一个 (token, head)，在 kernel 内按 Task 2 的规则算 scale（包括向上取整到 BF16）和量化值。

- [ ] **Step 1：写失败测试**
  - `test_store_matches_torch_reference`：N 取 {1, 7, 4096}，H 取 {2, 4}，D=128，数值乘以 `[1, 300]`；scale 和 `quantize_per_token_head` 的结果 `torch.equal`；int8 值不一致的比例小于 1e-4。
  - `test_store_only_touches_loc`：没写入的位置保持原值。
  - `test_store_zero_rows`：全 0 的行，scale 为 1，int8 为 0。

- [ ] **Step 2：实现。** 在 `Int8PerTokenHeadKVCacheMethod.quantize_and_store` 中：`SGLANG_INT8_KV_IMPL=fused`（默认）时调用这个 kernel，`plain` 时用 PyTorch 实现。

- [ ] **Step 3：导出补丁，提交** `feat(stage4): fused int8 quantize-and-store kernel`

### Task 6：融合 decode kernel

**Files:**
- Modify（SGLang 内）：`kernels/ops/attention/decode_attention.py`（`_fwd_grouped_kernel_stage1`、`decode_attention_fwd_grouped`、`decode_attention_fwd`）
- Test: `tests_stage4/test_fused_decode.py`（`needs_gpu`）、`tests_stage4/reference.py`

**Interfaces:**
- Produces：`decode_attention_fwd(..., k_scale_buffer: Optional[Tensor[S, H_kv] bf16] = None, v_scale_buffer: Optional[Tensor[S, H_kv] bf16] = None)`。两者都不为 None 时，走 `KV_INT8=True` 的 kernel：
  - 每读一块 K：`qk = dot(q, k.to(q.dtype)) * s_k[None, :]`；
  - 读 V 前：`p = p * s_v[None, :]`，再和 `v.to(q.dtype)` 做点积。
  - `k_scale` 和 `v_scale` 这两个标量参数在 INT8 时必须为 1.0，否则抛 `ValueError`。
  - 如果 `kv_group_num == 1`（不走 grouped 路径），或者是 MLA、HIP 的 lean 路径，就抛出 `NotImplementedError`。
- `tests_stage4/reference.py` 提供 `reference_attention(q, k, v, causal_offset=None) -> Tensor`：用 FP32 计算，供 Task 6 和 Task 7 共用。

- [ ] **Step 1：写失败测试**
  - `test_fused_decode_matches_reference`：参数化 batch {1, 4}、上下文 {1, 17, 1000, 4097}、GQA 分组 {6, 7}（kv_head 取 2 或 4），head_dim 128；K/V 用 Task 5 的 kernel 量化，参考值用恢复精度后的 K/V 算 `reference_attention`；断言输出相对 L2 < 1e-2。
  - `test_fused_decode_matches_plain`：和"恢复精度后调用原 BF16 kernel"的结果比较，余弦相似度 > 0.9999。
  - `test_flag_off_bitwise_equals_pristine`：用 `load_sglang_file` 从 `third_party/sglang-pristine` 加载原始的 `decode_attention.py`，BF16 和 FP8 各跑一次，输出 `torch.equal`。
  - `test_non_grouped_rejected`：`kv_group_num == 1` 时传入 scale，抛出 `NotImplementedError`。

- [ ] **Step 2：实现**

- [ ] **Step 3：导出补丁，提交** `feat(stage4): fused int8 decode attention kernel`

### Task 7：融合 extend kernel

**Files:**
- Modify（SGLang 内）：`kernels/ops/attention/extend_attention.py`（`_fwd_kernel` 的前缀循环、`extend_attention_fwd`）
- Test: `tests_stage4/test_fused_extend.py`（`needs_gpu`）

**Interfaces:**
- Produces：`extend_attention_fwd(..., k_scale_buffer=None, v_scale_buffer=None)`，规则和 Task 6 相同：只有前缀部分（从 `K_Buffer` 读的那段）乘 scale，本次新输入的部分不变。`extend_attention_fwd_unified` 和 dense FP8 等其它路径收到 scale 时抛 `NotImplementedError`。

- [ ] **Step 1：写失败测试**
  - `test_fused_extend_matches_reference`：参数化前缀长度 {0, 5, 300}、新输入长度 {1, 64, 257}，GQA 分组 7；参考值 = 前缀用恢复精度的 K/V、新输入用原始 K/V，加因果 mask，算 `reference_attention`；断言相对 L2 < 1e-2。
  - `test_fused_extend_matches_plain`：余弦相似度 > 0.9999。
  - `test_extend_flag_off_bitwise_equals_pristine`：BF16 和 FP8 下都和 pristine 版本 `torch.equal`。

- [ ] **Step 2：实现**

- [ ] **Step 3：导出补丁，提交** `feat(stage4): fused int8 extend attention kernel`

### Task 8：接入 Triton 后端，第二次服务器测试

**Files:**
- Modify（SGLang 内）：`fp4_kv_cache_quant_method.py`（新增 `KVCacheAttentionAccessKind.NATIVE_INT8`，按 `SGLANG_INT8_KV_IMPL` 选择 PLAIN 或 NATIVE_INT8 读取方式）、`srt/mem_cache/memory_pool.py`（`MHATokenToKVPool.get_kv_scale_buffer(layer_id) -> tuple[Tensor, Tensor]`；NATIVE_INT8 时 `get_key_buffer` 返回原始 int8）、`srt/layers/attention/triton_backend.py`（`forward_decode`、`forward_extend` 在 NATIVE_INT8 时取 scale 传给 kernel；INT8 时强制走标准 extend 路径）
- Modify: `experiments/stage4/l1_model_check.py`（加两项检查）、`scripts/stage4/test.sh`（L1 增加 fused 配置）
- Test: `tests_stage4/test_backend_wiring.py`（`needs_gpu`）

**Interfaces:**
- Consumes：Task 5、6、7 的 kernel 签名。
- Produces：`SGLANG_INT8_KV_IMPL` 取 `fused`（默认）或 `plain`，两种都能通过 `--kv-cache-dtype int8_per_token_head` 启动。

- [ ] **Step 1：写失败测试**（`tests_stage4/test_backend_wiring.py`，用 1.5B 和 `sglang.Engine`）
  - `test_fused_engine_generates`：fused 实现下生成 16 个 token，结果非空，并且和 plain 实现的生成结果前 8 个 token 相同（贪心解码）。
  - `test_prefix_cache_hit_matches_cold_request`：同一个长提示词（needle，150 条事实）连续请求两次，第二次命中 radix cache；两次的输出 token 相同。
  - `test_chunked_prefill_long_prompt`：`chunked_prefill_size=512`，600 条事实的 needle 提示词；和不分块时的输出前 16 个 token 相同。

- [ ] **Step 2：实现，导出补丁，提交并推送** `feat(stage4): wire fused int8 KV into triton backend`

- [ ] **Step 3：服务器检查点 2（需要你执行）**：`git pull`，重新运行 `setup.sh`（只会重新打补丁），再运行 `test.sh`。
  Expected：L0 全部 PASS；L1 中 plain 和 fused 在 CUDA Graph 开、关下都满足 `top1_agreement ≥ 0.99`，并且 fused 和 plain 的指标差距小于 0.005。

### Task 9：精度评测脚本

**Files:**
- Create: `experiments/stage4/eval_accuracy.py`、`scripts/stage4/eval.sh`
- Modify: `int8kv/prompts.py`（如需，加 `GSM8K_INSTRUCTION` 常量，和 `experiments/stage3_gsm8k.py` 共用）
- Test: `tests/test_stage4_eval.py`（Mac 上运行，用假 Engine）

**Interfaces:**
- Produces：`run_eval(engine, tokenizer, task: str, n: int, max_new_tokens: int) -> dict`。`task` 取 `gsm8k`、`needle150`、`needle600`。返回 `{"accuracy", "correct", "total", "predictions"}`。`engine` 只需要有 `generate(input_ids: list[list[int]], sampling_params: dict) -> list[dict]` 方法。
- 命令行：`--model --kv-dtype --impl --task --out`；结果 JSON 写到 `results/stage4/<模型名>/accuracy/<task>_<配置名>.json`。
- `eval.sh`：在 7B 上依次跑 4 种配置（BF16、FP8、INT8 plain、INT8 fused）× 3 个任务，GSM8K 跑完整 1319 题、`max_new_tokens=1024`，needle 各 100 题、`max_new_tokens=64`；最后调用 `summarize_accuracy(dir) -> dict`，对每个 INT8/FP8 配置和 BF16 做 McNemar 检验（复用 `int8kv.metrics.mcnemar_exact`），写入 `accuracy_summary.json`。

- [ ] **Step 1：写失败测试**
  - `test_run_eval_scores_with_fake_engine`：假 Engine 对所有输入返回 `"\\boxed{18}"`；两道 GSM8K 题中答案为 18 的那道算对，`accuracy == 0.5`。
  - `test_summarize_uses_paired_counts`：手写两个 predictions 文件，断言 `only_baseline`、`only_quantized`、`mcnemar_p` 和手算的结果一致。

- [ ] **Step 2：运行，确认失败；实现；运行，确认通过；提交** `feat(stage4): accuracy evaluation on SGLang`

### Task 10：性能测试脚本

**Files:**
- Create: `experiments/stage4/bench_decode_kernel.py`、`experiments/stage4/parse_logs.py`、`scripts/stage4/bench.sh`
- Test: `tests/test_stage4_parse_logs.py`（Mac 上运行，用固定的日志样例）

**Interfaces:**
- Produces（`parse_logs.py`）：
  - `parse_max_total_tokens(server_log: str) -> int`
  - `parse_peak_running_reqs(server_log: str) -> int`（取日志中 `#running-req: N` 的最大值）
  - `parse_bench_serving(output_json: Path) -> dict`（output token/s、TTFT 和 TPOT 的中位数）
  - `summarize_bench(dir: Path) -> dict`：同一配置跑 3 次取中位数，写入 `bench_summary.json`。
- `bench_decode_kernel.py --dtype {bf16,fp8,int8} --batch 32 --contexts 1024,4096,16384,32768`：用 CUDA event 计时，预热 10 次，测 100 次取中位数，输出 JSON。
- `bench.sh`：对每种配置启动 server（`--attention-backend triton --mem-fraction-static 0.85`），从日志读容量；跑 `bench_serving --dataset-name random`，输入/输出长度取 (1024, 256)、(4096, 512)、(8192, 512)，各 3 次；再跑长请求并发测试（输入 8192、输出 512，256 个请求，`--request-rate inf`）；最后加一组 BF16 + FlashInfer 的参考。INT8 plain 只测容量和 (1024, 256) 这一组吞吐。

- [ ] **Step 1：写失败测试**：在 `tests/fixtures/stage4/` 放一段真实格式的 server 日志和一个 bench_serving 输出 JSON。日志格式在检查点 2 时从服务器取一段真实日志；拿到之前，先按 SGLang 源码中的打印格式写。断言能解析出正确的数字。

- [ ] **Step 2：运行，确认失败；实现；运行，确认通过；提交并推送** `feat(stage4): benchmark scripts`

### Task 11：服务器正式运行和阶段 4 报告

**Files:**
- Create: `docs/stage4-report.md`
- Add: `results/stage4/Qwen2.5-7B-Instruct/`

- [ ] **Step 1：服务器检查点 3（需要你执行）**：`git pull`，然后在后台依次运行 `eval.sh` 和 `bench.sh`；跑完后把 `results/stage4` 打包下载到 Mac。

- [ ] **Step 2：解压、核对。** 每个结果文件里的环境信息、补丁 commit、配置名要和预期一致；容量比例要接近理论值 1.97。

- [ ] **Step 3：写报告。** 按设计文档第 1 节的成功标准逐项给出结论，没达到的如实写明，并标出未解释的现象；补充简历要点。

- [ ] **Step 4：提交并推送** `results: stage 4 on Qwen2.5-7B-Instruct and report`
