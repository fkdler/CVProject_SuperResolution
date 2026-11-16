# Stage3：DRCT-L ×4 发布模型的推理复现

日期：2026-11-16。**已完成完整Set14推理、指标核验和14图跨框架数值一致性验证。**
主基线固定官方`params_ema`、MindSpore2.7.2/Ascend910B4、PYNATIVE、FP32等价卷积。
论文表格数值仍有小差异，未声称逐项完全一致；训练能力属于尚未开始的Stage4。

## 1. 主要结果

| 协议 | 官方PyTorch PSNR / SSIM | MindSpore PSNR / SSIM |
|---|---|---|
| 发布YAML：BT.601 Y，每边裁4 HR像素 | 29.57982653 / 0.8008818045 | **29.57982569 / 0.8008819660** |
| 论文文本诊断：RGB，每边裁8 HR像素 | 27.77205648 / 0.7644960041 | 27.77205476 / 0.7644961155 |

主协议平均MS−PT差：PSNR **−8.4271e-7 dB**，SSIM **+1.6152e-7**。
所有14图的raw输出均通过`atol=rtol=1e-4`，全数据集最大绝对误差 **1.03414e-5**。
量化后逐图PSNR/SSIM也均通过预先声明的0.01dB / 1e-4差异限制，未事后放宽门槛。
PNG少量像素可能因半整数附近的微小浮点差异相差1级，逐图变化数见对照报告。

证据：[完整对照JSON](stage3/comparison.json)、[逐图差异CSV](stage3/comparison.csv)、
[MS逐图指标](stage3/mindspore_per_image.csv)、[官方PyTorch逐图指标](stage3/torch_per_image.csv)。

## 2. 数据与来源

使用[SelfExSR作者仓库的Set14预生成×4配对图](https://github.com/jbhuang0604/SelfExSR/tree/8f6dd8c1d20cb7e8792a7177b4f6fd677633f598/data/Set14/image_SRF_4)，
commit固定为`8f6dd8c1d20cb7e8792a7177b4f6fd677633f598`。
原始28个PNG共5,338,861字节，不改像素、不重新bicubic，14对HR/LR均精确4倍。
灰度PNG按官方OpenCV IMREAD_COLOR路径扩展成三通道。

数据清单SHA-256：`dcb9719a0395f617dcb83a08d6ffd11d8bbb6b7577ebbe1b68d1c77cff595e05`。
[逐文件URL/哈希/尺寸](stage3/dataset_manifest.json)和[来源配置](../../configs/set14_source.json)已进入Git，图片本身忽略Git。
该来源提供现成benchmark配对图；尚无DRCT作者当年实际测试文件清单，不能证明两者逐字节相同。
没有用Set14训练、调学习参数或选择改进模型。

## 3. 评测实现的独立核验

在云端直接加载固定BasicSR1.3.4.9的原始PSNR、SSIM、BT.601转换和tensor2img函数体，
只在内存中移除注册装饰器；原文件不改并新增四个文件的LF规范化SHA-256锁定。

- 两种尺寸×两种协议：本项目PSNR/SSIM与BasicSR的实测差值均为0。
- 四组量化检查（含半整数边界）：逐元素相等。
- 三组reflect padding（含已经整除尺寸）：与Torch F.pad逐元素相等。
- 28张HR/LR颜色/数值往返逐元素相等，解码和14对比例检查通过。

[核验报告](stage3/evaluation_parity.json)。改写的BasicSR指标保留Apache-2.0说明和许可证。

实际流程为BGR解码→RGB NCHW float32/255→右侧/底部reflect到16倍数→完整模型→
去padding→clamp/×255/round/uint8→逐图指标→算术平均。没有tile、TTA或self-ensemble。
主协议和RGB8诊断从同一模型输出计算，分别报告，不混合解释。

## 4. 权重、数值与资源

完整L仍为12 RDG / 60 Swin、27,580,719参数。默认`matmul`等价卷积保持原OIHW权重、
bias与零padding；原生卷积的Stage2偏差证据保留，未静默切回native。

- 官方文件SHA-256：`a99044c0275699d1a296ae21b8f322fa8c65d7b9be2213ee2a3dcc280ab8d64b`。
- 主分支固定`params_ema`。
- MS ckpt SHA-256：`146cfca01fd10398fc0ce1e52534b64f76af16dcebcfd9ae03235bcebbbb716b`。
- 全量14图NPU tensor分配峰值：2,399,461,376字节；内存池保留峰值：3,388,997,632字节。
- MS累计forward+主机取回约17.19秒，包含第一个新尺寸的约11.20秒编译/首跑开销。

内存统计为MindSpore runtime内存池统计，不是驱动总HBM。时间不是稳态吞吐基准，
也不把云端CPU参考与NPU的时间当作同硬件性能对比。较大图img_002的预先资源检查
仅确认能全图运行，不用于选模型或调整精度参数。

[MS完整报告](stage3/mindspore_report.json)、[PT完整报告](stage3/torch_report.json)、
[云环境](stage3/environment.json)记录配置、文件/代码哈希、逐图资源与计时。

## 5. 与论文表格的差异排查

[论文v5表1](https://arxiv.org/html/2404.00722v5)报告Set14×4为29.54dB / 0.8025。
本次主协议EMA结果与其相差约+0.03983dB / −0.001618，**不是完全相同的表格数值**。

已核实：官方checkpoint来源和规模正确；转换完整；指标与发布BasicSR一致；
官方PyTorch在相同数据和协议下得到同样差异，因此这部分差异不能归因于MS迁移误差。
论文文本的RGB/裁8与发布YAML的Y/裁4矛盾，独立RGB8结果也已如实记录。

另核验standalone脚本使用的`params`分支：先严格转换并通过两种合成输入的原始PT对齐，
再用同一固定数据/协议作分支诊断，Y4得到 **29.54578749 / 0.8001310172**。
它的PSNR更接近论文29.54，但SSIM仍未对应0.8025；本次没有因此改选主分支。
这个诊断只改变权重分支，仍用reflect包装器；没有声称完整复现standalone的flip拼接padding。
该分支未额外执行第二组14图PyTorch全量对照，不与主基线14图配对证据混为一谈。

[分支诊断](stage3/params_diagnostic_report.json)、[分支小输入对齐](stage3/params_alignment.json)、
[分支转换](stage3/params_conversion.json)。作者实际数据版本及表格对应的最终训练快照
缺少独立清单，残余差异来源未完全确认，不通过更换数据、混用协议或挑选分支凑论文数字。

## 6. 运行边界与交付物

本地只编辑、静态检查、文件下载/打包/传输/哈希核验；未导入或运行本地推理框架，
未解码图片或计算本地指标。所有数值工作在现有ModelArts完成，没有新增框架安装或付费实例。
SSH新指纹先由用户控制台独立核验，再通过严格校验连接，未关闭主机验证。

窗口约10:24:45—10:48（北京时间），约23分钟，含指纹核验和文件传输；
按用户提供的20.828元/小时标称费率约8元，非实际账单，不包含此前/之后的空闲计费。
末尾进程检查和npu-smi确认本项目评测进程、NPU进程均已退出，已通知用户可以停止Notebook。

云端原始结果：`/home/ma-user/work/drct_ms/experiments/stage3/`。
本地完整主基线输出备份：`experiments/stage3_primary_outputs.tar`（两框架raw.npy、SR PNG、报告、CSV）。
可直接查看的14张MS超分图：`experiments/stage3_sr_images/`。这些大产物与数据不进入Git。
Stage3工程交付与数值证据已完成，Stage4训练实现/训练smoke尚未开始。
