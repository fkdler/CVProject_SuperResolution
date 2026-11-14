# DRCT-L ×4 官方源码分析

分析日期：2026-11-14。所有结构结论以本地固定源码为准。

## 1. 版本与证据

| 来源 | 固定版本 / 位置 |
|---|---|
| 官方 DRCT | `c0a8b51e69241cf7f8ae598053235ff498517935` |
| 官方 BasicSR | `v1.3.4.9`，`5668ba75eb8a77e8d2dd46746a36fee0fbb0fdcd` |
| 论文 | [arXiv:2404.00722v5](https://arxiv.org/html/2404.00722v5)，2024-11-23 修订 |
| 网络 | `reference/DRCT/drct/archs/DRCT_arch.py` |
| L ×4 测试配置 | `reference/DRCT/options/test/DRCT-L_SRx4_ImageNet-pretrain.yml` |
| Base ×4 测试配置 | `reference/DRCT/options/test/DRCT_SRx4.yml` |
| 推理与指标入口 | `reference/DRCT/drct/models/drct_model.py` |
| 指标与量化 | `reference/BasicSR/basicsr/metrics/psnr_ssim.py`、`metric_util.py`、`utils/img_util.py` |

完整来源和关键文件 SHA-256 见 [manifest](../reference/manifest.json)。参考仓库独立保留，
不直接修改；根 Git 仓库通过 manifest 和获取脚本重建参考源码。

## 2. Base 与 L 的实际区别

两者使用同一个 `DRCT` 类。L 的主要变化是 RDG 数由 6 增加到 12，主宽度仍为 180。

| 项目 | DRCT Base ×4 | DRCT-L ×4 |
|---|---|---|
| `depths` | `[6] * 6` | `[6] * 12` |
| RDG 数 | 6 | 12 |
| 实际 Swin Block 总数 | 30 | 60 |
| 主干宽度 / growth channels | 180 / 32 | 180 / 32 |
| 窗口 / 初始 attention heads | 16 / 6 | 16 / 6 |
| 主 `mlp_ratio` | 2 | 2 |
| patch / 训练 LR 尺寸 | 1 / 64×64 | 1 / 64×64 |
| 上采样 | 两级 Conv + PixelShuffle(2) | 同左 |
| 作者报告参数量 | 14.13M | 27.58M |
| 本地源码实例化精确参数量 | 14,139,579 | 27,580,719 |

**`depths` 不是每组实际 Swin 层数。** 构造器用 `len(depths)` 确定 RDG 数，
每组传入 `depth=0`，而 RDG 内部固定构造 `swin1` 至 `swin5`。
`sum(depths)` 用于生成 DropPath 调度。不能将 L 实现成 72 层 Swin。
精确参数量由实际实例化工具输出，见 [CPU 审计](verification/reference_audit.json)。

## 3. 前向计算与 RDG

输入为 RGB、NCHW、FP32、[0,1]。先减 RGB 均值 `(0.4488, 0.4371, 0.4040)`，
乘 `img_range=1`，经 3×3 Conv(3→180) 得浅层特征。
PatchEmbed 仅展平空间维并转成 BLC，主干入口有 LayerNorm。
之后串联 12 个 RDG，最终 LayerNorm、PatchUnEmbed，再经 3×3 Conv(180→180)，
与浅层特征相加。重建端：3×3 Conv(180→64) + LeakyReLU，
两次 3×3 Conv(64→256) + PixelShuffle(2)，3×3 Conv(64→3)，最后加回均值。

每个 RDG 的五层实际配置如下。heads 来自源码的
`6 - ((180 + i*32) % 6)`，不能全部写为 6。

| 子层 | Swin 输入通道 | heads | shift | MLP ratio | 1×1 transition 输出 |
|---|---:|---:|---:|---:|---:|
| swin1 / adjust1 | 180 | 6 | 0 | 2 | 32 + LeakyReLU(0.2) |
| swin2 / adjust2 | 212 | 4 | 8 | 2 | 32 + LeakyReLU(0.2) |
| swin3 / adjust3 | 244 | 2 | 0 | 2 | 32 + LeakyReLU(0.2) |
| swin4 / adjust4 | 276 | 6 | 8 | **1** | 32 + LeakyReLU(0.2) |
| swin5 / adjust5 | 308 | 4 | 0 | **1** | 180，**无激活** |

第 i 层沿 token 的最后一个维度拼接原输入和此前 transition 输出；transition 在
NCHW 下进行，之后重新展平为 BLC。组输出为 `x + 0.2*x5`。
组内 LeakyReLU 负半轴斜率是 0.2；重建端 `nn.LeakyReLU()` 使用 PyTorch 默认 **0.01**。

Swin Block 使用两个 pre-LayerNorm 残差分支：
`x += DropPath(W/SW-MSA(LN(x)))`，再 `x += DropPath(MLP(LN(x)))`。
QKV 共用一个 Linear，按 Q/K/V、head、token、head_dim 的顺序拆分。
相对位置偏置表为 `(31*31, heads)`，索引按 ij 坐标、行优先构造。
shift 通过空间轴循环移位实现，mask 使用 0 / -100，而非负无穷。

## 4. 必须忠实保留的实现细节

- `compress_ratio`、`squeeze_factor`、`conv_scale`、`overlap_ratio` 等出现在配置中，
  但当前 DRCT 前向路径没有调用 CAB / ChannelAttention / overlap attention。
  文件中定义了这些类，不代表 DRCT-L 使用它们。
- 还存在重复定义的 `PatchMerging`，主 DRCT-L 路径不使用下采样或 PatchMerging。
- 所有五个 Swin 都使用该 RDG 的 `drop_path[0]`，并非五个独立递增概率。
  概率序列由长度 `sum(depths)` 的 linspace 切片而来，L 最后一组小于 0.1。
  推理需 `eval` / `set_train(False)`；训练阶段不能擅自修正该行为。
- 动态输入尺寸不等于构造时 64×64 时，源码对未 shift 的块也重算 mask；
  shift=0 时这个 mask 是全零。等价省略只可在验证后采用。
- `mean` 是普通 tensor，不在 PyTorch state_dict 中；相对位置索引和 attention mask
  则是 buffer。参数计数不能把这些 buffer 算成可训练参数。
- 网络本体不 pad、不裁剪输出、不 clamp。padding 与输出量化发生在模型包装器。

## 5. 预训练权重身份

官方 README 的 DRCT-L benchmark 行链接到
[DRCT-L_X4.pth](https://drive.google.com/file/d/1bVxvA6QFbne2se0CQJ-jyHFy94UOi3h5/view)。
这与 Real-DRCT-GAN、DRCT-XL ImageNet-only 权重不同。
测试 YAML 的本地文件名是 `DRCT-L_SRx4_ImageNet-pretrain.pth`，不能仅凭名称判定权重阶段。
官方 standalone `inference.py` 读取 `params`，YAML 请求 `params_ema`；
应读取真实顶层键，并逐分支验证，结果见 [权重审计](verification/checkpoint_audit.json)。
本地已下载485,570,697字节，SHA-256为
`a99044c0275699d1a296ae21b8f322fa8c65d7b9be2213ee2a3dcc280ab8d64b`。
`params`和`params_ema`均存在，各1000项，均严格加载通过；无缺失、多余、形状或几何
buffer不匹配，浮点张量均finite。后续主评测明确采用`params_ema`。
这是对官方链接所得文件的本地内容校验，作者未在已检查资料中提供独立的文件校验值。
结构匹配不等于证明该文件复现 29.54 dB，仍需实际 Set14 评测。

## 6. 数据、训练配置与论文差异

官方 L ImageNet 预训练 YAML：GT crop=256（LR=64），水平翻转和旋转增强，
每卡 batch=4、worker=6、seed=0；Adam，LR=2e-4，beta=(0.9,0.99)，
weight_decay=0；L1 均值损失；800k 步，在 300k/500k/650k/700k/750k 处减半，
EMA=0.999，无 warmup。

官方 L DF2K 微调 YAML：预生成 HR/LR 配对图，256 HR crop，每卡 batch=4；
Adam LR=1e-5，beta=(0.9,0.99)，L1，250k 步，
milestones=125k/200k/225k/240k，gamma=0.5，EMA=0.999。

DF2K = DIV2K 800 张训练图 + Flickr2K 2650 张图。论文描述了 ImageNet 预训练、
DF2K L1 优化、L2 收尾，但发布的这两个 L ×4 YAML 都是 L1；不能把 Real-DRCT-GAN
的 MSE 配置作为经典 SR 的 L2 最终阶段。论文的 beta2=0.999、batch=32、
前两阶段各 800k 步，与发布 YAML 也不完全一致；每卡 batch 不能直接当作全局 batch。

ImageNet 数据实现会先 modcrop，再用 OpenCV resize 将不足 patch 的图放大，
随后调用 BasicSR MATLAB 兼容 `imresize(..., 1/scale)`（默认 antialiasing=True）。
DF2K `PairedImageDataset` 则直接读现有配对 LR，不在线重新 bicubic。
读取 BGR float32 / 255 后转 RGB CHW；默认不额外标准化，模型内部减均值。
测试数据目录名为 GTmod4 / LRbicx4，但仓库未提供一套完整可核验的数据生成清单；
后续必须固定测试集来源及哈希，不能只凭目录名断言降采样完全一致。

## 7. 评测协议：发布配置优先，论文文本另列

| 项目 | 发布 BasicSR 测试路径 | 论文 v5 §6.3 文本 |
|---|---|---|
| 通道 | BT.601 Y | full RGB |
| 每边 HR 裁剪 | 4 | 2×scale = 8 |
| 量化 | `tensor2img` clamp [0,1]，×255，round，uint8 | 文本未充分规定 |

因此目前**不能断言 29.54 的评测口径已经由论文文字与代码共同确认**。
主 baseline 采用发布 YAML 的 Y / shave4；同时输出 RGB / shave8 诊断结果，分别命名。
项目诊断配置选择同样 uint8 量化，这是项目定义，并非从论文中确认的事实。

发布路径的细节：

1. OpenCV 解码 BGR→RGB，NCHW FP32 [0,1]。
2. 在右侧、底部做 **reflect padding**，补齐到 16 的倍数；已整除则不额外补。
   太小的图片可能不满足 PyTorch reflect 的 pad < 输入尺寸约束，应显式报错。
3. 全图 FP32 推理，默认不 tile、不自集成。按 pad×4 去除输出 padding。
4. SR 与 GT 都经过 `tensor2img`，转 HWC BGR uint8。先量化再计算指标。
5. PSNR：裁四边 4 个 HR 像素，再按 BasicSR BT.601 转浮点 Y，peak=255。
   BGR 系数顺序是 24.966 / 128.553 / 65.481，带 +16 偏移；不能用 OpenCV JPEG YCrCb 替代。
6. SSIM：同一 crop/Y 流程，11×11 Gaussian、sigma=1.5，valid 区域，
   C1=(0.01×255)^2，C2=(0.03×255)^2。
7. 各图指标算术平均，不将全数据集像素 MSE 先合并再计算 PSNR。

standalone `inference.py` 使用拼接 flip 的 padding，且已整除仍多补一个窗口，
与 `DRCTModel.pre_process` 的 reflect **不等价**。项目采用发布 YAML 对应的包装器作为
主评测依据，任何替代 padding、tile 或 TTA 都应单独标注，不能静默混用。

## 8. 本阶段验证范围

源码分析、CPU 参数/权重审计与几何对照用于锁定迁移基线。
后续已补齐MindSpore完整网络和严格转换/对齐工具，并在Ascend通过PYNATIVE/GRAPH O0
两个合成空间尺寸的数值对齐，见[阶段二记录](verification/stage2.md)。
Set14数据与PSNR/SSIM仍待实际评测。
后续按 [迁移清单](migration.md) 逐级推进。
