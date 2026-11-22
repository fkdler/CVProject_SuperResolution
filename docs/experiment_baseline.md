# DRCT-L 架构实验的对照基线

原版 DRCT-L 已通过结构、权重、Set14 推理与训练能力验收。现在补测原版在一套固定短微调协议下的画质、延迟与内存，供随后比较 RDG、注意力、FFN 或重建模块的改动。入口和本轮配置见 `tools/experiment.py` 与 `configs/experiment_baseline.json`。

2026-11-22 在既有 Notebook 完成原版基线测量，端到端 8 项检查全部通过。Stage3 的 Set14 成绩和 Stage4 的三步训练结果见[推理验收](verification/stage3.md)和[训练验收](verification/stage4.md)。

## 固定实验条件

| 项目 | 设置 |
|---|---|
| 网络与初始化 | 官方 DRCT-L ×4，12 RDG、60 Swin，严格加载已对齐的官方转换 EMA |
| 云端环境 | 既有 ModelArts，MindSpore 2.7.2，Ascend 910B4，PYNATIVE，FP32、HF32 关闭、matmul 等价卷积 |
| 数据 | DIV2K 官方 bicubic ×4，32 张训练、8 张验证。ID 按官方拆分分别等间隔取样，清单见 `configs/div2k_baseline_selection.json` |
| 训练 | 种子 0，batch 1，LR 64×64 / HR 256×256，L1、ReferenceAdam、LR 1e-5、EMA 0.999 |
| 总步数与恢复 | 32 步保存后重启进程，从第 33 步继续到第 64 步，使用最终 EMA |
| 验证 | 8 张 DIV2K 验证图全图推理，按补齐后面积从大到小执行，同图比较官方转换 EMA 与第 64 步 EMA。按发布 YAML 量化 uint8、BT.601 Y、HR 每边裁 4 像素 |
| 效率 | LR 64×64 与 256×256，各开一个新进程，batch 1。单独记录首跑，之后预热 5 次、计时 20 次 |

64 步是首轮工程对照规模。模型训练能力、EMA 可评测性、稳态资源量级是这次的验收目标；质量差值只描述这个短跑设置。后续正式对照需给原版、简单替代方案和候选架构相同的训练数据、采样预算及验证协议，并记录改动引起的权重迁移差异。若使用多个随机种子，各组使用相同种子集合和恢复安排。Set14 用于方案冻结后的测试。

FFN 缩宽适合作为单因素消融，也可以用来估计不同 Swin 位置的容量需求。结构研究同样可以替换 RDG 中逐层扩展的融合路径，或调整局部混合与窗口注意力的分工。建议先用逐模块测量定位成本，再选一个结构假设和一个简单对照。测速表记录整网规模和资源，模块成本需由后续测量确认。

数据读取时核对官方拆分、清单和逐图 SHA-256。图像按需解码，缓存最多四对。若 HR 尺寸仅在底部或右侧多出 1–3 像素，按 LR 的 4 倍尺寸裁掉余数并逐图记录。训练按全局样本序号确定配对裁剪与增强，恢复后接续相同样本。完整训练状态含模型、Adam 与 EMA。DropPath 的跨进程随机流仍有 Stage4 已知限制。

## 复跑命令

本地准备源码与原始文件。以下命令只做静态检查、网络文件下载和归档。已有可信数据时，下载命令会核对原始文件哈希。

```powershell
python -S tools/static_check.py
python -S tools/fetch_div2k_smoke.py --selection configs/div2k_baseline_selection.json --output data/DIV2K_baseline --workers 3
python -S tools/build_cloud_bundle.py --output experiments/experiment_baseline_source.tar
tar.exe -cf experiments/experiment_baseline_data.tar -C data DIV2K_baseline
```

等用户启动既有 Notebook 并确认就绪后再 SSH。沿用任务专用 known_hosts 和 `accept-new`，并核对传输归档 SHA-256。源码包在 `/home/ma-user/work` 解包到项目目录，数据包解到项目的 `data`。权重默认使用 Stage2 已有的 `weights/drct_l_x4_params_ema.ckpt` 和同名 JSON。

```bash
set -e
. /usr/local/Ascend/cann-8.5.2/set_env.sh
cd /home/ma-user/work/drct_ms
export PYTHONPATH="$PWD/src:$PWD:$PYTHONPATH"
run=experiments/baseline_64_run1   # 重跑时取一个新名字
python -m tools.probe_cloud --run-check --output "$run/environment.json"
timeout 600 python -u -m tools.experiment train --until-step 32 --output "$run/first"
timeout 600 python -u -m tools.experiment train --resume "$run/first/step_000032.ckpt" --output "$run/resumed"
timeout 300 python -u -m tools.experiment evaluate --output "$run/div2k_official"
timeout 300 python -u -m tools.experiment evaluate --checkpoint "$run/resumed/ema.ckpt" --output "$run/div2k_step64"
timeout 300 python -u -m tools.experiment benchmark --shape 64 64 --output "$run/bench_64"
timeout 300 python -u -m tools.experiment benchmark --shape 256 256 --output "$run/bench_256"
python -m tools.experiment audit --run-dir "$run"
```

另可用 `evaluate --split set14` 加载原版或当前架构的实验 EMA，输出同样口径的 14 图逐图结果。Stage3 官方转换与 PyTorch 的对齐验收仍可用原入口复跑。候选架构加入后，应让各组共用全图评测与测速代码，并绑定各自的网络配置及权重来源。

## 读懂报告和验收

- `first/` 与 `resumed/` 各有逐步 `train.jsonl`、完整状态 `step_*.ckpt`、可独立评测的 `ema.ckpt`、权重元数据和汇总报告。检查每步 loss 和全部梯度有限，恢复计数为 32，第一个恢复批次的 SHA-256 等于第一个进程记录的下一批次，最终计数为 64。
- 两份 DIV2K 报告应各有同样的 8 个图像 ID、清单哈希及 Y/裁4 协议，且 EMA 评测报告的权重 SHA-256 与第 64 步导出一致。逐图 CSV 用于定位退化图像。训练时用验证拆分分析质量，方案冻结后再汇报正式测试集。
- 测速用已构造输入，每次前向前后同步设备，计时包含 Python 调用和设备等待。首跑、预热和正式样本分开保存。正式计时窗口不含主机图像解码、输出回传和指标计算。报告含所有样本、p50/p95、参数数、权重文件大小。
- 内存报告记录 MindSpore 分配器与内存池的预热后基线和测量窗口峰值，并保留 `npu-smi` 的独立观察。驱动观察是时点取样，持续的物理 HBM 峰值需要另行采样。后续候选模型在同一台设备、相同尺寸、模式、精度及测量流程下对照。 [MindSpore 同步接口](https://www.mindspore.cn/docs/en/r2.7.2/api_python/runtime/mindspore.runtime.synchronize.html) · [MindSpore 峰值重置接口](https://www.mindspore.cn/docs/en/r2.7.2/api_python/runtime/mindspore.runtime.reset_peak_memory_stats.html)
- `audit` 写入 `acceptance.json`，检查恢复、训练步、划分、权重链路、画质和测速记录。小型验收摘要可进入 Git。数据、权重和大日志保留在工作盘及本地忽略目录。

## 已测效率与入口回归

两档输入均用官方 EMA、同一台 Ascend 910B4、FP32 PYNATIVE、五次预热及二十次正式采样。参数量为 27,580,719。

| LR 尺寸 | p50 | p95 | MindSpore 张量分配峰值 | 内存池保留峰值 |
|---|---:|---:|---:|---:|
| 64×64 | 265.24 ms | 279.23 ms | 563,998,720 B | 687,865,856 B |
| 256×256 | 1,233.42 ms | 1,239.89 ms | 5,513,742,336 B | 6,887,047,168 B |

原版 Set14 使用新入口全图重跑，得到 29.57982569 dB / 0.8008819660，数据清单和指标与 Stage3 一致。该次回归检查只用已冻结的官方 EMA。 [逐图报告](verification/experiment_baseline/set14_official.json) · [64×64 测速](verification/experiment_baseline/bench_64.json) · [256×256 测速](verification/experiment_baseline/bench_256.json)

## 短微调实测与全图评测

| 权重 | DIV2K 验证 8 图平均 PSNR | 平均 SSIM |
|---|---:|---:|
| 官方转换 EMA | 32.07353679 dB | 0.90059943 |
| 原版第 64 步 EMA | 32.07347715 dB | 0.90061232 |

两段的训练循环累计 364.81 秒，含首步编译、读图和参数审计。去除每段首步后，更新阶段的中位耗时分别为 1.727 秒和 1.659 秒，各取 31 步。64 步的 916 个梯度张量均有限，恢复时确认了 3,667 个状态张量、优化器步数和下一批次哈希。评测进程独立严格加载导出的 EMA。两份成绩接近，可作为相同步数与数据设计下的参照。 [验收摘要](verification/experiment_baseline/acceptance.json) · [官方 EMA 逐图](verification/experiment_baseline/div2k_official.json) · [第 64 步 EMA 逐图](verification/experiment_baseline/div2k_step64.json)

首次按 ID 评测时，第 7 张 `0879` 的 LR 为 468×510，`Stack-op0` 申请 9,059,696,640 字节输出失败。该设备的 Best Fit 池提示不支持 `empty_cache`。正式运行改为先处理最大补齐形状，两个权重都完成 8 张全图，且官方权重在两次顺序下已完成的 6 张图指标一致。输入、填充、网络、权重与指标保持一致，输出逐图表按 ID 排序。尺寸继续增大时仍需实测显存。 [MindSpore 缓存接口](https://www.mindspore.cn/docs/en/r2.7.2/api_python/runtime/mindspore.runtime.empty_cache.html)

本次云端记录位于 `/home/ma-user/work/drct_ms/experiments/baseline_20261122`。本地完整日志归档为 `experiments/baseline_20261122_records.tar`，SHA-256 为 `1ab1b675ea4b6e19b844c689b2987a3ae71158387d484d36d428c2aec7428bfb`，含两次失败详情、每次延迟样本、训练日志和 NPU 快照。Git 副本省略重复的官方参数映射。 [数据清单](verification/experiment_baseline/dataset_manifest.json)

最终 EMA 与完整训练状态已备份在 `experiments/baseline_20261122_checkpoints/` 并核对 SHA-256。训练和测速先执行，评测的排序修复后上传，因此报告各自记录源码身份。恢复此次历史训练须用与保存时匹配的源码包 `experiments/experiment_baseline_source.tar`，也保存在云端 work 根目录。用当前提交复跑时，两段训练使用同一源码版本。此次所有数值工作在云端完成，本地静态检查覆盖 39 个 Python 文件。
