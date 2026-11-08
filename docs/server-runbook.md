# Linux 服务器运行手册（阶段 0-3，并为阶段 4-5 做准备）

## 1. 选什么机器
| 需求 | 建议 |
|---|---|
| 阶段 0-3（1.5B） | 任意 ≥16GB 显存的 NVIDIA 卡都够 |
| 阶段 4-5（7B + SGLang，和 FP8 对比） | **RTX 4090 24GB 或 L40S 48GB**。这两张卡原生支持 FP8，对比更公平；A100 没有原生 FP8，FP8 那一组可能跑不了或不公平 |

系统选 Ubuntu 22.04，镜像选带 PyTorch ≥ 2.6、CUDA 12.x 的。为了阶段 4 不换机器，建议直接租 4090。
**按小时计费的平台，跑完记得关机。**

## 2. 安装
```bash
git clone https://github.com/jackwubupt-ops/int8-kv-cache.git
cd int8-kv-cache
bash scripts/setup_linux.sh
```
- 镜像里已有满足要求的 PyTorch 时会直接复用，不重复下载；否则从 PyTorch 官方源安装（可用 `CUDA_INDEX=cu126` 指定）。
- 成功时会打印 PyTorch、transformers 版本和显卡名称，测试全部通过，最后显示 `Setup complete.`。
- 仓库是私有的：克隆时用 GitHub 账号和一个**只授权这个仓库**的 fine-grained token 登录。
- 访问 Hugging Face 慢时，可先设置你信任的镜像：`export HF_ENDPOINT=https://<镜像地址>`；有些平台自带学术加速，按平台说明开启。

## 3. 跑实验
建议放在 `tmux` 里跑，断开 SSH 也不会中断：
```bash
tmux new -s exp
bash scripts/run_stages.sh                                  # 默认 Qwen2.5-1.5B-Instruct，全部阶段
```
分开跑：`bash scripts/run_stages.sh Qwen/Qwen2.5-1.5B-Instruct 1`（阶段可选 `all`、`1`、`2`、`3`、`gsm8k`，第三个参数是 GSM8K 样本数）。

## 4. 提交结果
```bash
git add results
git commit -m "results: Qwen2.5-1.5B-Instruct stages 1-3"
git push
```
不想在服务器上登录 GitHub，也可以把 `results/` 用 `scp` 拷回本机再提交。

## 注意
- 必须使用 Instruct 模型，原因见 [windows-runbook.md](windows-runbook.md)。
- 服务器结果同样以精度结论为主；性能数字到阶段 4 用 SGLang 统一测量。
