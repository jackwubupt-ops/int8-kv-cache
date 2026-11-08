# Windows 运行手册（阶段 0-3）

阶段 0-3 只用 PyTorch 和 Transformers，**不需要 WSL2 或 SGLang**，直接在 Windows 上运行。

## 1. 准备
- Python 3.11（python.org 安装版，带 `py` 启动器）、Git、最新 NVIDIA 驱动。
- 插上电源并选择高性能模式。笔记本 GPU 限功耗只影响速度，不影响精度结果。
- 仓库是私有的，首次克隆时按提示登录 GitHub。

## 2. 安装
```powershell
git clone https://github.com/jackwubupt-ops/int8-kv-cache.git
cd int8-kv-cache
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
```
成功时会打印显卡名称、显存大小，单元测试全部通过，最后显示 `Setup complete.`。

如果访问 Hugging Face 很慢，可以在运行前设置镜像（由你自行选择可信的镜像）：
```powershell
$env:HF_ENDPOINT = "https://<镜像地址>"
```

## 必须使用 Instruct 模型
所有实验都把输入包装成对话模板。基座模型（如 `Qwen/Qwen2.5-1.5B`）没有学过这种格式，会输出乱码，
导致 GSM8K 和长上下文结果全部失真，所以脚本会直接拒绝基座模型名。

## 3. 先用小模型冒烟
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_stages.ps1 -Model Qwen/Qwen2.5-0.5B-Instruct -Stage 1
```

## 4. 跑完整实验
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_stages.ps1 -Model Qwen/Qwen2.5-1.5B-Instruct
```
也可以用 `-Stage 1`、`2`、`3`、`gsm8k` 分开跑。`stage3_model.py` 和 GSM8K 每完成一组配置就会保存一次。

## 5. 提交结果
```powershell
git add results
git commit -m "results: Qwen2.5-1.5B-Instruct stages 1-3"
git push
```
结果推送后告诉我，我会基于数据写分析报告，并确定阶段 4 的量化粒度。

## 结果文件
| 文件 | 内容 |
|---|---|
| `stage1/kv_stats.json` | K/V 分布、各粒度误差和 scale 开销 |
| `stage1/*.png` | 分布图和误差对比图 |
| `stage2/operator_metrics.json` | 算子级误差（分 Prefill / Decode） |
| `stage3/model_metrics.json` | 模型级对比、噪声底、KV 字节比、长上下文准确率 |
| `stage3/gsm8k.json` | GSM8K 准确率及答案变化数 |

## 注意
- 已经用 `.venv`（基于 `C:\Python311` 的 `--system-site-packages`）的话，`git pull` 后无需重装。
- 笔记本结果**只用于精度结论**，不要作为延迟或吞吐数字。性能数据在阶段 4 的服务器上测。
- 显存不足时，先关闭其他占用 GPU 的程序，再减小 `--needle-facts`。
