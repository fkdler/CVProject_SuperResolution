# Stage4：完整 DRCT-L 训练能力云端验收

日期：2026-11-16。结论：**预训练继续训练的2+1步smoke通过，完整网络可反向传播、更新、验证、保存并跨进程恢复。**
本轮未启动长训练，不宣称复现论文的完整训练收敛、从零初始化或训练指标。

## 实现与来源

- 完整12 RDG / 60 Swin、916参数张量、27,580,719元素，FP32、matmul等价卷积，PYNATIVE。
- 保留发布微调YAML的mean L1、Adam LR1e-5/beta=(0.9,0.99)/eps1e-8、weight_decay0、
  EMA0.999及原始milestones；batch明确由4降至1，LR64/HR256不变。
- 使用官方发布的转换EMA权重继续训练，权重SHA256
  `146cfca01fd10398fc0ce1e52534b64f76af16dcebcfd9ae03235bcebbbb716b`。
  该发布权重并非微调YAML引用的ImageNet中间快照，不能声称训练起点与论文相同。
- DIV2K官方训练0001–0004、官方valid0801–0802，源ZIP抽取原始PNG，不使用Set14。
  六对图尺寸均精确4倍，本次HR modcrop余数全0。哈希见[数据清单](stage4/dataset_manifest.json)。
- 每步记录样本ID、裁剪、增强、loss、LR、finite检查、参数变化、耗时与内存。
  见[操作单](../stage4_training.md)及`configs/train_smoke.json`。

## 实测结果

### Adam语义修正

原生`nn.Adam`的普通量级梯度测试曾通过，但补充微小梯度测试发现epsilon位置不同。
对于首步梯度`[1e-9,1e-7]`，PyTorch公式更新为`[-9.090909e-7,-9.090909e-6]`，
原生MS实际为`[-9.900984e-8,-4.999999e-6]`，不能视为相同优化器。
失败日志见[native_adam_epsilon_failure](stage4/native_adam_epsilon_failure.log)。
两种公式分别见[MS2.7.2融合算子](https://www.mindspore.cn/docs/en/r2.7.2/api_python/ops/mindspore.ops.Adam.html)
与[PyTorch2.7 Adam](https://docs.pytorch.org/docs/2.7/generated/torch.optim.Adam.html)。

已加入`ReferenceAdam`：复用2.7.2融合算子、原状态与调度，每步将原生算子epsilon输入
乘以`sqrt(1-beta2^t)`，使其等价于在偏差校正后的分母加epsilon。
新增三步微小/变号/零梯度独立解析解检查，以及学习率里程碑与实际更新联动测试。
原生版本的2+1步记录作为诊断单列于`native_adam_*_report.json`，不作为修正版本训练证据。
下面的最终报告必须来自修正后重新执行的完整2+1步验证。

| 验证 | 实际结果 |
|---|---|
| 训练组件测试 | 6 passed，37.57秒；LR边界/实际调度、配对增强、数据恢复、Adam解析更新/EMA/完整状态恢复 |
| 第1步 L1 / 更新张量数 | 0.02911445 / 864，全部916个梯度finite |
| 第2步 L1 / 更新张量数 | 0.02106535 / 916，全部916个梯度finite |
| 独立进程第3步 L1 / 更新张量数 | 0.01593148 / 916，全部916个梯度finite |
| 模型+Adam+EMA恢复 | 3667状态张量逐项精确回读，形状/dtype/键集合严格匹配 |
| 跨进程恢复 | 同一step2的验证输出哈希、逐图指标与下一batch哈希完全一致；global step从2继续到3 |
| 检查点再回读 | 两个进程各自保存后回读，live验证输出/指标精确不变 |
| 验收脚本 | 11项检查全部通过，见[acceptance](stage4/acceptance.json) |

不同step使用不同随机训练crop，且包含官方DropPath；不能把这三个loss数值解释成收敛曲线。
验收要求每步梯度finite且参数实际更新，不要求所有张量每步均发生可见变化；保留原始DropPath规则。
小网络无随机层的连续/恢复训练状态逐项一致；完整网络只保证状态和数据位置恢复，
**尚不保证DropPath跨进程随机流逐位续接**。

验证采用两个固定中心LR64块、Y/裁4、uint8量化：

| 时点 | live PSNR / SSIM | EMA PSNR / SSIM |
|---|---|---|
| 训练前 | 35.36312909 / 0.8990480792 | 初始化为同一权重 |
| step2 | 35.36211897 / 0.8990501882 | 35.36314923 / 0.8990476221 |
| step3 | 35.36015756 / 0.8990364500 | 35.36314451 / 0.8990476300 |

这些指标用于验证数据/评测/恢复通路，不是完整DIV2K或Set14指标；没有以它们选参或宣称效果提升。
逐图记录见[first_report](stage4/first_report.json)、[resumed_report](stage4/resumed_report.json)。

## 资源与费用估算

- 修正后第1步复用了部分先前算子缓存，仍含编译：36.06秒；第2步：1.53446秒。
  恢复进程首步仍需部分编译：38.76秒。两步训练循环总39.08秒，恢复一步39.73秒。
  最初原生Adam诊断的冷编译首步为147.77秒，单列保留，不能把缓存后的36秒当作冷启动时间。
- 计时包含梯度finite检查、Adam、EMA；每参数变更审计、验证、checkpoint和初始化另有开销。
  只有**一次**热身后观测，不能视为稳定吞吐基准。
- MindSpore分配器峰值6,279,979,520字节（约5.85GiB），池峰值6,727,848,960字节（约6.27GiB）。
  恢复运行`npu-smi`取样HBM为30,939/32,768MB，进程显示28,121MB；这与MS分配器统计口径不同。
  **未连续采集物理HBM峰值，不能据MS分配器数字断言batch4能放入显存。**
  原始观察摘要见[resource_observation](stage4/resource_observation.json)。
- 按本次batch1、每步1.53446秒和标称20.828元/小时粗略外推：

| 步数 | 仅更新循环时间 | 仅实例标称费用 |
|---|---|---|
| 1000 | 约25.6分钟 | 约8.88元 |
| 5000 | 约2.13小时 | 约44.39元 |
| 250000 | 约106.56小时 | 约2219.42元 |

估算不含数据、初始化/验证/保存和空闲/存储等费用；不是账单，不代表剩余预算，也不等同于原batch4配方的费用。
原250k完整配方不应直接启动。下一次有意义的微调应先扩大DIV2K训练/验证覆盖，
再批准步数和费用，并保留相同数据/预算用于后续比较；本轮不会在四张训练图上追加长训练。

## 环境、证据与备份

- 预置Python3.11.10、MindSpore2.7.2、CANN8.5.2、Ascend910B4；官方run_check通过。
  预置TE7.8兼容告警和弃用警告保留，未安装/升级任何框架。
  首次测试的学习率基类导入错误和原生Adam epsilon差异已修正；最终6项测试通过。
- 本地只运行标准库文件下载/哈希/打包、源码编辑和AST/配置静态检查；所有图片像素操作、
  框架导入、数值测试及训练都在Notebook完成。
- SSH按用户最新要求跳过人工指纹核验，使用任务专用known_hosts+accept-new，不改全局设置。
- 最终训练检查点云端：`experiments/stage4/reference_adam_resumed/step_000003.ckpt`，441,530,270字节；
  SHA256 `da9b99ee993722fb8d6dfafc8c9bb5aeaea4f170b06c07f919fbdde1e6e84faf`。
  本地备份`experiments/stage4_reference_adam_checkpoint/`，不进入Git，不在本地加载。
  原生Adam的先前检查点只作诊断，未覆盖或用作最终恢复起点。
- 报告/元数据/日志归档`experiments/stage4_final_records.tar`，SHA256
  `2765f2815f868a7ef0475199d7886665832cd7398f9e1b21cc01f34c617f40f0`。
  小报告、数据清单、测试日志与[环境](stage4/environment.json)跟踪于本目录。
- 03:59:51UTC检查：NPU无运行进程，HBM恢复约2874MB。未替用户停止Notebook。

## 尚未覆盖

完整数据集/长时间微调与收敛、GRAPH训练、原始batch4显存、准确持续吞吐、
论文原始ImageNet/DF2K完整训练和L2收尾、随机深度跨进程随机流一致性均未验证。
这些不能由本次smoke推断为完成。
