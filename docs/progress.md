# 开发进度

更新：2026-11-14。当前阶段：**第一阶段完成；第二阶段基础模块已开始，待MS运行验证**。

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
| Python 语法检查 | 通过 | `python -m compileall -q src tools tests` |
| 本地测试 | 14 passed / 1 module skipped | `verification/local_pytest.txt` |
| MindSpore 模块运行与梯度 | 未运行，本地缺少 MindSpore | skip 明确记录，不计为通过 |
| Ascend / GRAPH模式 | 未运行 | 未连接云端 |
| Set14 PSNR/SSIM | 未运行 | 不将论文29.54当作实测值 |

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

1. 在目标MS 2.7.2环境运行现有基础模块对照测试，记录误差与设备信息。
2. 按固定源码实现WindowAttention、SwinTransformerBlock、RDG及完整DRCT-L；逐层FP32对齐。
3. 编写完整转换脚本，训练参数100%映射、buffer逐项核查、strict load与转换哈希报告。
4. 固定Set14数据来源/清单/哈希，先跑原始PyTorch，再跑MindSpore全图同协议评测。
5. baseline成功后实现训练smoke，实测step时间/HBM并给出预算方案；长训练待用户批准。
6. 依据真实瓶颈确定改进和消融；当前不改变baseline模型结构。

## 算力与环境

本地：Windows，项目专用Python3.13 CPU PyTorch；MS目标2.7.2未安装/未运行。
未使用其他项目的Conda环境执行本项目实验。云端预算约600元，标称20.828元/小时；
本阶段无云端操作和训练任务，不对用户已有实例是否持续计费作推断。
