# DRCT-L ×4 的 MindSpore 复现

CV 课程大作业，目标运行环境为 **MindSpore 2.7.2 / Ascend 910B4**。
当前已编写完整MindSpore DRCT-L、严格权重转换和逐层/完整输出对齐工具。
Ascend PYNATIVE和GRAPH O0合成输入对齐均通过；Stage3的Set14完整推理和14图跨框架对齐已完成。
训练能力属于尚未开始的Stage4。

**计算约束：所有模型、参考生成、权重转换和数值测试仅在ModelArts执行。**
本地仅编辑源码、静态检查、整理和传输文件，不安装或调用任何推理框架。

## 已确认的基线

- 官方源码：`c0a8b51e69241cf7f8ae598053235ff498517935`。
- DRCT-L ×4：**27,580,719** 参数，12 个 RDG、每组 5 个 Swin Block。
- Base ×4：14,139,579 参数，6 个 RDG；主宽度同为180。
- 官方权重：[DRCT-L_X4.pth](https://drive.google.com/file/d/1bVxvA6QFbne2se0CQJ-jyHFy94UOi3h5/view)。
- 已下载并验证 `params` / `params_ema` 均严格加载通过，主分支选用 `params_ema`；
  来源、SHA-256与核验结果见[权重审计](docs/verification/checkpoint_audit.json)。
- 作者报告的Set14×4参考值为29.54dB / 0.8025；本项目如实保留实测差异。
- 实际主协议（Y/裁4，EMA）为 **29.57982569dB / 0.8008819660 SSIM**。
  官方PyTorch同数据/协议为29.57982653 / 0.8008818045，14图raw输出全部通过原门槛。
  论文SSIM0.8025未逐项完全对应，不将论文值当作实测或把协议差异说成模型改进。
  数据、逐图指标和排查见[Stage3结果](docs/verification/stage3.md)。

来源：[官方仓库](https://github.com/ming053l/DRCT)、
[论文 v5](https://arxiv.org/abs/2404.00722v5)。详细证据与差异见
[源码分析](docs/source_analysis.md)、[迁移清单](docs/migration.md)、
[进度](docs/progress.md)。

**评测有未消除的来源冲突：** 发布 YAML 使用 Y 通道、每边裁4个HR像素，
论文 v5 文本写 RGB、裁8。项目以发布配置作为主协议，并将 RGB/裁8 单列诊断。

## 目录

```text
reference/          官方只读参考仓库、版本 manifest
src/drct_ms/        MindSpore 模块与常量几何
tools/              源码获取、结构/权重审计、参考数据生成
configs/            复现架构与评测协议规格
tests/              几何测试、官方 PyTorch 对照、MindSpore 离线对照
tests/fixtures/     官方 CPU PyTorch 生成的小型输入/输出/梯度参考
requirements/       本地与目标环境依赖说明
scripts/            预留正式推理、评测与训练入口
data/ weights/      本地数据与权重，内容忽略 Git
experiments/        大日志、图像、检查点，内容忽略 Git
docs/               分析、迁移、进度及小型验证结果
```

## 本地源码操作（Windows）

```powershell
python -S tools/static_check.py
python -S tools/build_cloud_bundle.py
```

以上只使用标准库处理文本与文件，不导入框架或执行数值测试。
源码包忽略凭据、环境和权重；现有官方权重另行传输。
第一阶段已有的本地CPU结果和 [环境快照](docs/verification/reference_environment.txt)
保留为历史记录，原本地参考环境不再调用。所有数值工具和pytest收集入口都有云端保护。

## MindSpore 验证入口（待运行）

已编写窗口划分/合并、PatchEmbed/UnEmbed、MLP、Attention、Swin、RDG、
PixelShuffle、DropPath与完整12组L网络。目标MindSpore环境无需安装PyTorch；
参考输出与NumPy参数包在另一个云端参考环境生成。
先核对现有ModelArts环境，不覆盖预置MindSpore/CANN依赖。
详细连接、环境隔离、转换与验证顺序见 [云端操作单](docs/cloud_validation.md)。

在已具备 NumPy 与 pytest 的 **MindSpore 2.7.2** 环境，项目根目录运行：

```bash
. /usr/local/Ascend/cann-8.5.2/set_env.sh
export PYTHONPATH="$PWD/src:$PWD:$PYTHONPATH"
python -m tools.probe_cloud --run-check
python -m pytest tests/test_mindspore_primitives.py tests/test_mindspore_network.py -q -ra --ms-mode pynative
```

转换与对齐入口分别为`tools.export_reference_bundle`、`tools.convert_checkpoint`和
`tools.verify_alignment`。强制检查完整权重，记录来源/分支/产物哈希，拒绝不完整映射。
目前仅预训练加载路径，不宣称从零初始化与官方训练随机数流相同。

默认卷积使用等价的FP32 slice/stack/GEMM计算，保持原OIHW权重、bias和零padding。
原生Ascend卷积在本环境的完整输出对齐未过原门槛，`conv_backend='native'`仅保留诊断用途；
等价后端在两个合成输入上的最大输出绝对误差为PYNATIVE约3.16e-6、GRAPH O0约3.43e-6，
均通过原门槛。转换后的权重已备份至`weights/drct_l_x4_params_ema.ckpt`（不进入Git）。
合成输入证据见[适配验证记录](docs/verification/stage2.md)，真实14图证据见
[Stage3结果](docs/verification/stage3.md)，重现命令见[评测操作单](docs/stage3_evaluation.md)。

Stage4已通过完整L网络的2+1步云端训练与跨进程恢复：DIV2K配对裁剪/增强、L1/Adam/EMA、
严格训练状态恢复及日志均可用。长训练尚未启动，须由实际速度与预算决定；
见[训练操作单](docs/stage4_training.md)及[验收记录](docs/verification/stage4.md)。

改动模型前使用 [对照实验操作单](docs/experiment_baseline.md) 复跑原版短微调、
DIV2K 全图 Y/裁4 评测，以及 Ascend 固定尺寸延迟和内存基线。
