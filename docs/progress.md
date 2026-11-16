# 开发进度

更新：2026-11-16。当前阶段：**Stage3完整Set14推理与跨框架一致性验证已完成；Stage4尚未开始**。

## Stage3 完成情况

- 已固定SelfExSR作者仓库commit及14对原始HR/LR PNG，记录文件哈希与尺寸头信息，不重采样。
- 完整推理评测、BasicSR指标/量化/padding对照、逐图浮点输出和指标比较工具均已在云端验证。
- 主协议EMA/Y4：MindSpore **29.57982569dB / 0.8008819660**；官方PT **29.57982653 / 0.8008818045**。
- 全部14图raw输出通过原atol=rtol=1e-4；最大绝对误差1.03414e-5，平均PSNR差−8.4271e-7dB。
- RGB8诊断单列；论文表格29.54/0.8025与发布checkpoint结果有小差异，未宣称逐项完全一致。
- 官方`params`分支也作了固定协议诊断（29.54578749/0.8001310172），未替换事先固定的EMA主基线。
- 本地仅文件准备/传输与31个Python文件静态检查；所有框架导入、图片解码、数值工作都在云端。
- 新SSH指纹已由用户控制台核验；沿用预置环境，无框架安装/升级，无训练。
- 实际窗口约23分钟（标称约8元，非账单），结束时验证进程和NPU进程已退出，已通知可停止Notebook。
- 逐图指标、报告、哈希和原始输出已备份。方案见`docs/stage3_evaluation.md`，结果见`docs/verification/stage3.md`。

## Stage2 历史记录（2026-11-14）

- 用户明确要求所有模型与数值计算只在云端运行；本地仅源码编辑、纯文本静态检查和文件整理/传输。
- 不再使用第一阶段的本地参考环境；本轮未安装、导入、运行任何本地推理框架，也未运行本地pytest。
- 已补齐WindowAttention、SwinTransformerBlock、RDG、Upsample和完整12组DRCT-L。
- 已补齐DropPath，保留五层宽度/heads/MLP ratio、adjust5无激活、残差0.2及重建slope0.01。
- slice/concat实现循环移位；动态mask用设备张量生成，construct内不调用NumPy。
- 已写云端PyTorch参考导出、MindSpore严格转换、序列化回读、完整输出和首个RDG逐层对齐工具。
- 模型模块、数值工具和pytest收集入口均增加Linux ARM64及work目录检查，防止误在本地执行。
- 云端执行前固定FP32策略，禁用HF32；PYNATIVE与GRAPH分开验证。
- PYNATIVE首轮14项测试通过；修复梯度测试内联lambda的GRAPH解析问题后，GRAPH O0的16项测试通过。
- 官方EMA权重转换覆盖916个参数张量、27,580,719元素；84个几何buffer核对通过，ckpt严格回读通过。
- 原生Ascend卷积即使禁用HF32仍有首层约3.23e-4偏差，完整输出最大误差约1.20e-3，未通过原门槛。
- 增加同一OIHW权重与零padding的slice/stack/GEMM等价卷积作为默认，保留native用于诊断。
  两种输入完整PYNATIVE对齐通过，最大绝对误差3.16e-6/2.92e-6；首个RDG等关键层也通过原门槛。
- 完整GRAPH O0对齐通过，两种输入最大绝对误差均3.43e-6，未放宽1e-4的原allclose门槛。
- 转换后ckpt与小型证据已备份本地（只做文件传输，未在本地加载）。本轮没有安装框架或运行训练。
- 验证窗口约18:02—18:32，约30分钟；标称实例费约10.4元，非实际账单。
- 本项目验证进程已退出，npu-smi确认无NPU运行进程；Notebook是否停止由用户控制。
- 已通知用户需要启动已有Notebook并SSH连接；首次计划约30分钟/标称10.41元，不启动训练。
- 用户通过ModelArts控制台核验了新指纹；使用项目忽略目录内的专用known_hosts恢复严格SSH，未改全局记录。
- 非交互会话通过加载现有CANN8.5.2环境并保留原PYTHONPATH修复库路径；预置框架与依赖均未安装/升级。
- 操作顺序、命令与依赖隔离见 [云端操作单](cloud_validation.md)。

## 已完成

- [x] 检查工作目录，初始为空；初始化本项目 Git 仓库。
- [x] 克隆官方 DRCT 并锁定 `c0a8b51e69241cf7f8ae598053235ff498517935`。
- [x] 克隆对应 BasicSR `v1.3.4.9`，锁定 `5668ba75eb8a77e8d2dd46746a36fee0fbb0fdcd`。
- [x] 阅读论文 v5、README、模型、训练/测试配置、数据及评测实现。
- [x] 核对 L/Base 实际结构、参数量与官方权重链接；记录论文/代码/推理脚本差异。
- [x] 下载官方DRCT-L_X4.pth（约463MiB），记录SHA-256，两个参数分支均严格加载通过。
- [x] 建立独立参考源码、MS代码、配置、工具、测试、实验与文档目录；忽略大文件与凭据。
- [x] 编写 `README.md`、`AGENTS.md`、源码分析、迁移风险及本进度文件。
- [x] 本地独立 CPU PyTorch 环境实例化两种官方网络，保存完整参数形状审计。
- [x] 首批 MS Cells：PatchEmbed/UnEmbed、MLP（erf GELU）、PixelShuffle；窗口划分/合并。
- [x] 生成可离线用于目标 MS 环境的官方 PyTorch 基础模块参考输出及 MLP 输入梯度。
- [x] NumPy 几何与官方 PyTorch 索引/mask 对照测试通过。

## 实际验证

| 验证 | 结果 | 证据 |
|---|---|---|
| 官方源码版本与规范化哈希 | 通过 | `reference/manifest.json`、`tools.fetch_reference` |
| DRCT Base 参数 | 14,139,579；6 RDG / 30 Swin | `verification/reference_audit.json` |
| DRCT-L 参数 | 27,580,719；12 RDG / 60 Swin | 同上 |
| 官方checkpoint | params/params_ema各1000项；strict load通过，finite，buffer一致 | `verification/checkpoint_audit.json` |
| Ascend PYNATIVE模块测试 | 初版14项通过 | `verification/ascend_stage2/pytest_pynative.log` |
| Ascend GRAPH O0模块测试 | 最终16项通过，含等价卷积输出/梯度 | `verification/ascend_stage2/pytest_graph.log` |
| MindSpore转换 | 916张量/27,580,719元素100%映射，84buffer一致，保存回读通过 | `verification/ascend_stage2/drct_l_x4_params_ema.json` |
| PYNATIVE完整网络对齐 | 32×48与64×64通过，max_abs≤3.16e-6 | `verification/ascend_stage2/alignment_pynative_matmul.json` |
| GRAPH O0完整网络对齐 | 同两输入通过，max_abs≤3.43e-6 | `verification/ascend_stage2/alignment_graph_matmul.json` |
| Python 语法检查 | 通过 | `python -m compileall -q src tools tests` |
| 第一阶段历史本地测试 | 14 passed / 1 module skipped，本轮未重跑 | `verification/local_pytest.txt` |
| 第一阶段本地MindSpore测试 | 历史缺少运行时而skip；本轮云端结果见上 | 不将历史skip计为通过 |
| Set14推理/指标/14图对齐 | 已完成；MS主协议29.57982569 / 0.8008819660 | `verification/stage3.md`及`verification/stage3/comparison.json` |

官方权重SHA-256和原始来源已固定在manifest，主分支选择params_ema。
仅结构审计不能替代跨框架数值对齐。完整阶段记录见
[stage1](verification/stage1.md)。

## 关键发现

1. depths 的6不等于组内6层；每组固定5层，L共60层Swin。
2. 五层通道180/212/244/276/308，heads=6/4/2/6/4，最后两层MLP ratio=1。
3. adjust5后无LeakyReLU，RDG残差系数0.2；重建端LeakyReLU slope=0.01。
4. 测试 YAML 是Y/裁4，论文文本是RGB/裁8；官方standalone与BasicSR padding也不同。
5. 官方训练YAML与论文在步数、beta2和最终L2阶段说明上不完全一致，逐项保留证据。
6. 官方测试YAML请求params_ema，standalone读取params，必须核查真实checkpoint。

## 下一步

1. 等待用户进入Stage4后，实现训练数据、损失、优化器、checkpoint恢复与日志。
2. 先用训练集少量样本做云端梯度/参数更新/恢复smoke，禁止用Set14训练或选参。
3. 实测训练step时间和HBM后估算预算，长训练待用户批准；当前不启动Stage4。
4. 后续若要求论文表格逐项精确追溯，需作者数据/最终快照证据；不改现有已冻结基线凑数。
5. 改进和消融安排在训练基线可用后，保持一致预算和协议。

## 算力与环境

本地：Windows，仅编辑、静态检查与文件整理；第一阶段已有本地参考环境不再使用。
目标MS2.7.2/Ascend910B4只在云端执行；已由用户授权SSH完成本轮小输入验证。
未使用其他项目的Conda环境执行本项目实验。云端预算约600元，标称20.828元/小时；
不对用户已有实例当前状态、已产生费用或预算余量作推断。
