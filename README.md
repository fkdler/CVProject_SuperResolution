# DRCT-L ×4 的 MindSpore 复现

CV 课程大作业，目标运行环境为 **MindSpore 2.7.2 / Ascend 910B4**。
当前完成官方源码分析、工程初始化、CPU 结构审计与首批基础模块编写。
完整 MindSpore DRCT-L、权重转换、Set14 评测与训练仍在后续阶段。

## 已确认的基线

- 官方源码：`c0a8b51e69241cf7f8ae598053235ff498517935`。
- DRCT-L ×4：**27,580,719** 参数，12 个 RDG、每组 5 个 Swin Block。
- Base ×4：14,139,579 参数，6 个 RDG；主宽度同为180。
- 官方权重：[DRCT-L_X4.pth](https://drive.google.com/file/d/1bVxvA6QFbne2se0CQJ-jyHFy94UOi3h5/view)。
- 已下载并验证 `params` / `params_ema` 均严格加载通过，主分支选用 `params_ema`；
  来源、SHA-256与核验结果见[权重审计](docs/verification/checkpoint_audit.json)。
- 作者报告的 Set14 ×4 为29.54 dB；本项目尚未测得该结果。

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

## 本地开发验证（Windows）

当前已创建 `.venv-reference`，与全局及其他实验环境分离。已有环境直接运行：

```powershell
.\.venv-reference\Scripts\python.exe -m tools.fetch_reference
.\.venv-reference\Scripts\python.exe -m tools.inspect_reference
.\.venv-reference\Scripts\python.exe -m pytest -q -ra
```

新环境从项目根目录创建：

```powershell
python -m venv .venv-reference
.\.venv-reference\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv-reference\Scripts\python.exe -m pip install -r requirements/reference.txt
.\.venv-reference\Scripts\python.exe -m tools.fetch_reference
```

本阶段实测依赖见 [环境快照](docs/verification/reference_environment.txt)。CPU PyTorch
用于源码与迁移对照，和目标 MindSpore 分开。审计工具在内存中加载原始网络定义，
只去掉注册装饰器并载入 BasicSR 所需的两类辅助函数，避免安装完整 CUDA 训练栈；
这不是官方 BasicSR 端到端评测。上游文件保持原样，并核查源码哈希。

下载官方权重到 `weights/DRCT-L_X4.pth` 后执行：

```powershell
.\.venv-reference\Scripts\python.exe -m tools.inspect_checkpoint weights/DRCT-L_X4.pth
```

审计输出在 `docs/verification/`。`params` 与 `params_ema` 分别严格检查，
不静默回退。权重不进入 Git。

需要重建基础模块参考输入、输出与输入梯度时：

```powershell
.\.venv-reference\Scripts\python.exe -m tools.generate_primitive_fixtures
```

## MindSpore 验证入口（待运行）

已编写窗口划分/合并、PatchEmbed/UnEmbed、MLP 和 PixelShuffle；
MS 测试使用已生成的官方 PyTorch 参考数据，目标环境无需安装 PyTorch。
先在现有 ModelArts 环境核对 Python/MS/CANN/NPU。安装缺失的 pytest 前应确认
使用项目隔离环境或预置环境认可的方式，勿直接覆盖镜像依赖。

在已具备 NumPy 与 pytest 的 **MindSpore 2.7.2** 环境，项目根目录运行：

```bash
python -c "import mindspore as ms; assert ms.__version__ == '2.7.2'; ms.set_device('Ascend'); import pytest; raise SystemExit(pytest.main(['-q', 'tests/test_mindspore_primitives.py']))"
```

当前本地没有 MindSpore，目标运行结果尚未验证。后续先补齐 WindowAttention、Swin、
RDG 与完整 L 网络，完成 FP32 数值一致性与 Set14 baseline 后，再实现训练及改进。
本阶段没有连接云端、创建实例或启动训练。
