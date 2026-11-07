# 阶段 0：框架现状调研（2026-11-07）

| 框架 | INT8 KV Cache | 依据 |
|---|---|---|
| vLLM | 主干有 `KVQuantMode.INT8_PER_TOKEN_HEAD`（动态 per-token-head scale） | `vllm/v1/kv_cache_interface.py`（main） |
| SGLang（commit 8920463） | `--kv-cache-dtype` 可选值没有 INT8；在 `srt`、`kernels` 中未搜到 per-token-head 实现（DSA 目录除外） | `srt/arg_groups/fields/model.py:195` |

## 待确认
- vLLM 的 INT8 在哪个正式版本可用，以及命令行参数名。阶段 4 开始前核实。

## SGLang 接入点
| 环节 | 位置 |
|---|---|
| 参数入口 | `srt/arg_groups/fields/model.py:195` |
| Cache 分配 | `srt/model_executor/pool_configurator.py`（`resolve_kv_cache_quant`） |
| Cache 写入 | `srt/mem_cache/memory_pool.py:164`（`_set_kv_buffer_impl`） |
| Decode 读取 | `srt/layers/attention/torch_native_backend.py:239-245` |
| Qwen2 调用 | `srt/models/qwen2.py:244-245`（K 带 RoPE 后写入） |
| 高性能路径 | `srt/layers/attention/triton_backend.py`、`kernels/ops/attention/` |
