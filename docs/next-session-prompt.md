# 下次会话的起始提示（阶段 4，2026-11-17 写）

把下面整段复制给下一个会话：

---

继续我 GitHub 上 `jackwubupt-ops/int8-kv-cache` 的阶段 4（SGLang INT8 KV cache），分支 `stage4-int8-kv`。本地 clone 在 `~/int8-kv-cache`，Mac 上的测试环境是 `.venv`（Python 3.9），运行 `.venv/bin/python -m pytest -q`，上次结果是 198 passed、52 skipped。

先读这几份文件：`docs/superpowers/plans/2026-11-08-stage4-sglang-int8-kv.md`（计划）、`docs/stage4-decision-log.md`（进度记录，最后一段是 2026-11-17 那次失败的安装）、`docs/stage4-engineering-draft.md` 第 9.3 节（服务器步骤）、`docs/stage4-report.md`（报告草稿）。

**当前状态：** Task 0–8 已在服务器上确认。Task 9–11 的代码已写完，并在 Mac 上测过。剩下的只有服务器运行（`scripts/stage4/run_all.sh`）和根据结果写报告。2026-11-17 在一台新的 AutoDL 容器（Python 3.12）上安装环境，没有装完，原因是网络。代码没有改动，也没有产生结果。

**这次要做的事：**

1. 先把 `scripts/stage4/setup.sh` 改成能绕开慢速网络（先在 Mac 上写测试，再提交）：
   - torch 的下载源可以配置。目前 `ensure_cu128_torch` 写死了 `--index-url https://download.pytorch.org/whl/cu128`，这会覆盖 `PIP_INDEX_URL`，导致 torch 的所有依赖都从国外下载，一直断线。国内可用的源：`-i https://mirrors.aliyun.com/pypi/simple --extra-index-url https://mirrors.nju.edu.cn/pytorch/whl/cu128`（NJU 有 torch 2.11.0+cu128；阿里云的 pytorch-wheels 没有）。
   - sglang-kernel 的 wheel（`sglang_kernel-0.4.2.post2+cu129-cp310-abi3-manylinux2014_x86_64.whl`，348223055 字节）从 GitHub 下载只有约 10 KB/s。改为在 Mac 上下载，scp 到服务器的 `/root/autodl-tmp/wheels/`；`setup.sh` 已经会直接使用这个位置的现成文件。给出 Mac 端的下载和 scp 步骤。
   - `apply.sh` 要从 GitHub 克隆 sglang v0.5.12（commit 127b9e32）。它支持 `SGLANG_REPO_URL`，可以改为在 Mac 上生成一个包含 tag `v0.5.12` 的 git bundle，传到服务器后用这个 bundle 克隆。
2. 一步一步带我在服务器上装完 setup，直到输出 `Setup complete.`。pip 安装时不走代理（`env -u http_proxy -u https_proxy ...`）。
3. 运行 `RUN_ALL_SMOKE=1` 试跑，然后运行正式的 `run_all.sh`。后台运行的那条命令要单独一行，加上 `< /dev/null`；git 命令加 `--no-pager`。
4. 拿到 `stage4_results_*.tar.gz` 后：运行 `make_report.py`，补完 `docs/stage4-report.md` 中标 ⏳ 的部分，做一次全分支代码审查，再合并到 main。

**规则：** 服务器上只通过 `scripts/stage4/*.sh` 运行 GPU 任务。你在本机工作，服务器命令由我来执行；你给我的每条服务器命令都要带检查或确认步骤。不要编造任何数字，报告里的数字都必须来自结果文件。每个阶段结束后，在 `docs/stage4-decision-log.md` 里追加记录。
