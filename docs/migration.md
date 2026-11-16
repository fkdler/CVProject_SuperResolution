# PyTorch → MindSpore 迁移与验证清单

目标：MindSpore **2.7.2**，Ascend **910B4**，先 FP32。主结构见
[源码分析](source_analysis.md)，架构规格见 [配置](../configs/drct_l_x4.json)。

## 1. 模块与风险

| PyTorch 模块 | MindSpore 实现 | 核验要点 | 当前状态 |
|---|---|---|---|
| window_partition / reverse | reshape + ops.transpose | BHWC 窗口顺序，非方形、多 batch、整数 batch 推导 | Ascend两模式对照通过 |
| PatchEmbed / UnEmbed | nn.Cell，展平和转置 | 不做 patch convolution；动态推导 RDG 通道 | Ascend两模式对照通过 |
| LayerNorm | nn.LayerNorm，epsilon=1e-5 | 最后通道维；weight→gamma，bias→beta；默认 epsilon 不同 | Ascend两模式近常量输入对照通过 |
| MLP | Dense + erf GELU + Dropout + Dense | PyTorch 默认精确 GELU；最后两层 ratio=1 | Ascend两模式输出/输入梯度对照通过 |
| relative_position_index | NumPy 预计算 → MS Tensor | ij 网格，坐标差方向，展平顺序，int32 | 已编写预计算，CPU 对照测试 |
| shifted-window mask | NumPy 预计算 → MS Tensor | 0/-100，shift=0，全动态输入尺寸，缓存按 H/W/shift 区分 | 已编写预计算，CPU 对照测试 |
| WindowAttention | nn.Cell，Dense / matmul / softmax / gather | QKV 排列；scale 先乘 Q；bias 广播；softmax 最后一维 | Ascend人工可解析测试和关键层对齐通过 |
| SwinTransformerBlock | pre-LN、slice/concat循环移位、窗口注意力、两次残差 | 0/8 shifts，设备动态mask，DropPath eval/train语义 | Ascend mask/梯度/关键层对齐通过 |
| RDG | 具名Cells + concat + 1×1 Conv | 五层宽度/heads/ratio；最终不激活；残差×0.2 | 首组与末组关键值对齐通过 |
| Conv2d | 默认MatmulConv2d；保留native诊断 | 原OIHW权重、bias、零padding；slice/stack/GEMM等价实现 | 默认后端完整PYNATIVE对齐通过；GRAPH卷积输出/梯度通过 |
| LeakyReLU | nn.LeakyReLU | RDG slope=0.2，重建 slope=0.01 | 包含在完整网络对齐通过路径 |
| PixelShuffle | reshape + transpose 的 nn.Cell | PyTorch 通道交错顺序；避免 DCR/CRD 歧义 | Ascend两模式逐元素对照通过 |
| DRCT-L | 12 RDG + 全局残差 + ×4 重建 | RGB mean、img_range、patch_norm、确切参数量 | Ascend两输入PYNATIVE/GRAPH O0完整输出对齐通过 |
| eval 包装器 | reflect pad / unpad / uint8 metrics | 与standalone padding区分；Y4/RGB8分开 | BasicSR参考核验及Set14全14图对照通过 |

预计算 NumPy 只用于常量几何，不作为模型计算后端或 MindSpore 的替代。
PixelShuffle 用 reshape/transpose 的等价实现必须与 PyTorch 排序精确对照。
GELU 使用 `0.5*x*(1+erf(x/sqrt(2)))`，原因是 MindSpore 默认近似 GELU，
且官方 GELU 文档提示 CPU 对 approximate=False 有限制；目标 Ascend 上仍需实测。

MindSpore 官方依据：
[LayerNorm](https://www.mindspore.cn/docs/en/r2.7.2/api_python/nn/mindspore.nn.LayerNorm.html)、
[Dense](https://www.mindspore.cn/docs/en/r2.7.2/api_python/nn/mindspore.nn.Dense.html)、
[GELU](https://www.mindspore.cn/docs/en/r2.7.2/api_python/nn/mindspore.nn.GELU.html)、
[transpose](https://www.mindspore.cn/docs/en/r2.7.2/api_python/ops/mindspore.ops.transpose.html)。
transpose 在 Ascend 不用负轴索引。文档支持某平台不代表项目已在该平台验证。

## 2. 权重映射设计

- Linear / Dense：二者权重均为 `[out, in]`，先对形状，不能默认转置。
- Conv2d：预期 `[out, in, kh, kw]`，逐项验证，尤其 bias 默认设置。
- LayerNorm：仅对应 norm 模块的 `.weight→.gamma`、`.bias→.beta`，不能全局替换 weight/bias。
- 相对位置偏置表：保留 `[961, heads]`；训练参数必须 100% 覆盖。
- `relative_position_index`、`attn_mask`：按配置重建，先逐项对照 checkpoint 的 buffer。
  不可把所有未知项当作 buffer 丢弃。mean 在源码中不进入 state_dict，需显式固定。
- 具名结构尽量保留 `layers.0.swin1.attn.qkv` 等路径，转换脚本生成逐项映射报告。
- `params` / `params_ema` 选择必须显式写入报告；不存在时直接报错或由用户明确选择，
  不能静默替代。用 `torch.load(..., weights_only=True, map_location='cpu')`。
- 输出报告：来源 URL/哈希、源/目标参数数量和元素数、缺失/多余项、形状不匹配、
  buffer 白名单及校验结果、目标 ckpt 哈希和 strict load 结果。

## 3. 验证顺序与门槛

1. 索引、mask、窗口分组、PixelShuffle：与官方实现逐元素相等；先验证坐标语义。
2. LN、MLP、attention：固定输入和共享权重，eval/dropout=0，记录 max_abs、mean_abs、
   RMSE、相对 L2；测试包含近常量 LN 输入、负值、非方形、多 batch 和 shifted mask。
3. Swin / RDG：逐层 hook 对照，避免最终输出掩盖中间排列错误；核验 0.2 与激活位置。
4. 完整 L：参数名称/形状/数量对齐，随机输入、小幅值和真实 LR 图分别对照。
   PyTorch参考只在云端隔离环境，MindSpore目标在Ascend；不执行任何本地模型计算。
5. 目标设备PYNATIVE与GRAPH O0的两个合成尺寸已通过；Stage3完整Set14使用PYNATIVE。
   不能将两尺寸GRAPH验证扩展为任意动态shape或GRAPH Set14全量已通过。
6. Set14全图同权重、同LR/HR、同量化/协议的14图对照已通过，已保存逐图PSNR/SSIM。
   在确认偏差来源前，不为“达标”改变模型、数据或协议。
7. 推理对齐后才做小数据训练 smoke；核验 finite gradients、loss、参数更新、
   checkpoint 恢复，再实测 step 时间、HBM，给出预算估算后申请长训练。

现有 MLP 小模块候选容差为输出 atol=2e-6/rtol=2e-5，输入梯度 atol=3e-6/rtol=3e-5；
这是待测门槛，不是声称已达到的实测结果。完整网络误差门槛应依据 CPU FP32 对照结果
制定，并与图像量化及 PSNR 差异一起报告。

## 4. 本地与云端边界

用户已更新约束：本地只编辑/静态检查/整理传输文件；不安装或运行推理框架，
不使用本地CPU/GPU/NPU做数值实验。第一阶段本地环境和结果只作为历史记录，不再调用。
模型、转换、参考输出和测试均在ModelArts完成，执行入口有云端目录与平台保护。
云端仅在算子、推理或训练需要时使用，先核对 Conda/Python/MS/CANN/NPU；
保留预置环境，将项目和产物放入 `/home/ma-user/work`，不创建新付费实例。
预算约 600 元，对应标称约 28.8 小时实例费用，实际还需扣除已有运行、数据盘及其他费用。
这是预算算术，不是可承诺的训练时长。

昂贵训练必须先实测，给出总耗时和费用，等待批准。当前阶段没有运行云端任务。

## 5. 当前可运行入口（仅云端，尚待实际执行）

`tools.export_reference_bundle`在云端独立PyTorch环境导出官方state和中间值；
`tools.convert_checkpoint`在预置MindSpore环境完成100%参数/确定性buffer核查与保存回读；
`tools.verify_alignment`比较完整输出，PYNATIVE可额外追踪首组五层Swin与主干节点。
检查点分支、源哈希、中间包哈希、输出ckpt哈希和误差均进入报告。
模型适配命令见[云端操作单](cloud_validation.md)，Stage3命令见[评测操作单](stage3_evaluation.md)。

当前权重初始化以预训练加载为前提；尚未复现官方从零训练的初始化分布和随机数流。
完整模型位于`src/drct_ms/models/drct.py`；已有Ascend执行证据，以进度和阶段记录中的
具体模式、后端、输入与误差为准，不能推广为Set14指标或任意动态shape已通过。
