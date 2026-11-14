# DRCT-L MindSpore适配与Ascend验证

日期：2026-11-14。完整网络与EMA转换已完成，合成输入数值对齐通过；Set14指标尚未测量。

## 1. 执行范围

本地只编辑源码、AST/JSON/TOML静态解析、打包、文件传输和哈希核验。
未安装、导入或运行任何本地推理框架，未运行本地pytest；第一阶段已有环境不再使用。
所有张量计算、框架导入、参考生成、转换、测试与推理均在用户已有ModelArts实例中完成。
没有安装或升级云端框架；镜像已有MindSpore2.7.2、PyTorch2.7.1+cpu、NumPy1.26.4和pytest8.3.5。
PyTorch参考与MindSpore在不同进程执行，未同时导入两个框架。

用户确认Notebook就绪后才连接。首次发现主机指纹变化，用户通过控制台核验后才接受新公钥，
使用项目忽略目录内专用known_hosts，全程保留严格校验，未改全局known_hosts。
非交互会话缺少CANN库路径，加载已有`/usr/local/Ascend/cann-8.5.2/set_env.sh`并保留
原PYTHONPATH后修复；没有改动预置安装。

环境：Python3.11.10，Linux aarch64，MindSpore2.7.2，CANN8.5.2，Ascend910B4/32GiB。
官方`ms.run_check()`成功。预置环境仍输出te7.8版本提示和第三方弃用告警，保留日志，未擅自升级。

## 2. 模型、权重与覆盖

官方commit：`c0a8b51e69241cf7f8ae598053235ff498517935`。
完整12 RDG，每组5个Swin，总计60个Swin、27,580,719参数，未缩小网络。
真实MS参数名称/形状与第一阶段原始源码审计一致。

源官方权重SHA-256：`a99044c0275699d1a296ae21b8f322fa8c65d7b9be2213ee2a3dcc280ab8d64b`。
明确选择`params_ema`。云端参考导出全部学习参数和buffer为NumPy文件，再由MS环境转换。

| 检查 | 结果 |
|---|---|
| 学习参数张量 | 916/916，100%映射 |
| 学习参数元素 | 27,580,719 |
| 位置索引/名义mask | 84项，逐项一致 |
| 缺失/额外/形状/非finite项 | 均无 |
| LayerNorm映射 | 仅weight→gamma、bias→beta；其他权重不转置 |
| 保存后严格载入、逐参数回读 | 通过 |

MindSpore ckpt SHA-256：`146cfca01fd10398fc0ce1e52534b64f76af16dcebcfd9ae03235bcebbbb716b`。
云路径：`/home/ma-user/work/drct_ms/weights/drct_l_x4_params_ema.ckpt`。
本地已备份到`weights/`并核对同一哈希，仅存储，不加载运行。
完整转换报告：[conversion](ascend_stage2/drct_l_x4_params_ema.json)。
其中`numeric_forward_validation=pending`是转换创建时的状态；独立的后续对齐报告才是数值验证证据。

## 3. 原生卷积偏差与等价实现

第一轮原生`nn.Conv2d`首层最大误差约3.23e-4，最终32×48/64×64输出最大误差
约9.26e-4/1.20e-3，未通过预先声明的atol=rtol=1e-4门槛。
采用旧`ascend_config`与推荐`op_precision` API设置相同保留FP32/禁用HF32策略后，
原生结果仍相同。本轮不能据此断言具体的底层舍入或HF32机制。
原失败报告均保留，未通过放宽阈值“修复”。

默认改为`MatmulConv2d`：对同一个零padding输入取空间窗口，以`[C,Kh,Kw]`顺序展开，
与原`[O,C,Kh,Kw]`权重做FP32 GEMM，再加同一bias。
计算的交叉相关公式、感受野、学习参数数量和名字均不变，不增加/删减网络层。
1×1卷积直接转换布局后GEMM，3×3卷积使用slice/stack；所有张量操作属于MindSpore。
`conv_backend='native'`保留用于诊断，不能作为当前严格FP32默认路径。

独立小型卷积测试用直接空间求和检查输出，并用解析输入梯度检查反传，1×1/3×3均通过。
此实现会增加展开窗口临时内存；尚未测Set14全图峰值和速度，不声称性能更优。

## 4. 实测结果

使用完整L、官方EMA、eval、RGB NCHW FP32输入，无AMP、无图像量化或裁边。
输入由云端NumPy seed=20261114生成，是合成LR，不是Set14。
所有参考数组、源分支、来源/产物哈希在
[reference manifest](ascend_stage2/manifest.json)记录。

| 模式/后端 | 输入 | 输出max_abs | 输出mean_abs | RMSE | 结果 |
|---|---|---:|---:|---:|---|
| PYNATIVE/matmul | 1×3×32×48 | 3.15905e-6 | 4.85400e-7 | 6.15582e-7 | 通过 |
| PYNATIVE/matmul | 1×3×64×64 | 2.92063e-6 | 4.84099e-7 | 6.12896e-7 | 通过 |
| GRAPH O0/matmul | 1×3×32×48 | 3.42727e-6 | 4.88388e-7 | 6.18767e-7 | 通过 |
| GRAPH O0/matmul | 1×3×64×64 | 3.42727e-6 | 4.87958e-7 | 6.17353e-7 | 通过 |

完整输出allclose门槛始终为`abs(error) <= 1e-4 + 1e-4*abs(reference)`。
PYNATIVE额外核对首个RDG的五个Swin及入口/末端关键值，均通过同一门槛。
中间特征值域较大，不能将最终输出3e-6的绝对误差概括为所有层都达到该绝对误差。
GRAPH检查实际完整`construct`，未用eager追踪代码替代图执行。

模块测试：PYNATIVE首版14项通过，最终GRAPH O0共16项通过，包括参数形状、几何、
LN/GELU/MLP、Attention、DropPath、Swin输入梯度及等价卷积输出/解析梯度。
初次GRAPH的两项失败是测试代码内联lambda的解析限制，改为具名loss后通过。
初次GE默认图编译与最终O0配置分开记录，不将最终O0通过扩展为O2通过。

证据：[PYNATIVE](ascend_stage2/alignment_pynative_matmul.json)、
[GRAPH O0](ascend_stage2/alignment_graph_matmul.json)、
[native失败](ascend_stage2/alignment_pynative.json)、
[模块测试](ascend_stage2/pytest_graph.log)。
GRAPH两个新形状首次编译及运行分别约210.6秒、165.2秒；这些不是稳态推理延迟。

## 5. 预算、收尾与边界

本轮窗口约18:02—18:32（北京时间），含指纹核验、传输、环境处理和编译，约30分钟。
按用户提供的20.828元/小时标称费率，约10.4元，非实际账单；未核查此前空闲计费或总余额。
没有新建付费实例、修改框架依赖、启动训练或追加Set14全量评测。
结束时本项目验证进程已退出，npu-smi无运行进程；已通知用户可以停止Notebook节费，未自行关机。

当前完成的是完整模型适配、转换与两个固定空间尺寸的数值验证，不能代替：
Set14数据/指标复现、任意动态shape、GRAPH O2、稳态速度/HBM测试、训练与改进消融。
初始化仍以完整预训练加载为前提，未复现官方从零训练随机初始化和随机数流。
下一步先固定Set14来源、配对清单和哈希，核对reflect padding、uint8量化及Y4/RGB8两种协议，
在同一数据/权重下比较原始PyTorch与默认MS等价卷积结果。
