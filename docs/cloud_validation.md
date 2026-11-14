# 云端验证操作单

状态：用户已启动既有Notebook并授权SSH；环境、转换、PYNATIVE对齐及GRAPH模块测试已执行。
完整GRAPH O0对齐已通过，两种输入最终输出max_abs均约3.43e-6。
本轮已结束，完整证据见 [阶段记录](verification/stage2.md)。下列命令用于云端重现，
不应在已经完成的验证后无理由重复运行。
本地只编辑源码、做纯文本静态检查及整理传输文件。已有本地PyTorch环境不再使用，
不在本地安装或运行任何推理框架。第一阶段历史CPU结果只作为来源记录保留。

## 1. 就绪与成本边界

请用户启动已有ModelArts Notebook，再使用之前配置的SSH连接；不创建新实例。
先核对远程Conda环境、Python、MindSpore2.7.2、CANN8.5.2与910B4。
使用系统已有`ssh`/`scp`，不读取私钥内容、不关闭主机验证；指纹变更须重新核验。
云项目根目录约定为`/home/ma-user/work/drct_ms`。

首次窗口计划约30分钟，标称实例费用约10.41元，不含可能额外的存储/流量费。
这只是预算上限目标：下载依赖、编译和问题定位可能改变耗时；接近窗口时汇报剩余事项。
仅环境检查、参数审计、小输入FP32输出/梯度验证；不启动训练或Set14全量评测。
用户已有实例此前计费时长未知，不将600元视为全部仍可用。

## 2. 传输内容

本地可以用标准库打包源码（无框架导入/数值计算）：

```powershell
python -S tools/static_check.py
python -S tools/build_cloud_bundle.py
```

源码包`experiments/cloud_source.tar`只包含白名单源码、规格、测试、约50KB参考数据
以及固定的官方网络/辅助函数。没有环境、凭据、Git历史和大权重。
另行传输现有`weights/DRCT-L_X4.pth`，约463MiB。
目标目录不存在才解包，避免覆盖云端已有实验。先检查`df -h /home/ma-user/work`。
源码包包含精简参考源码，不含完整上游Git仓库，云端无需运行`tools.fetch_reference`。
`reference_loader`以固定文件哈希验证所需原始定义。

## 3. MindSpore环境：只读核对后做最小验证

以下均在远程项目根目录、已经确认的预置MindSpore Conda环境执行：

```bash
. /usr/local/Ascend/cann-8.5.2/set_env.sh
export PYTHONPATH="$PWD/src:$PWD:$PYTHONPATH"
python -m tools.probe_cloud --run-check
python -m pytest tests/test_mindspore_primitives.py tests/test_mindspore_network.py -q -ra --ms-mode pynative
```

不直接`pip install -r requirements/ascend.txt`覆盖镜像。若缺少pytest，先检查能否在
work目录建`--system-site-packages`测试venv继承预置MindSpore，再只安装缺失的测试依赖。
若预置版本与目标不同，停止依赖变更并报告，不能自动升级MindSpore/CANN。

基础测试先确认几何、LN、精确GELU、PixelShuffle和输入梯度；网络测试核对全部参数
形状与27,580,719参数总数、实际RDG配置、动态mask、slice/concat循环移位、
人工可解析Attention和DropPath语义。测试不会启动优化器或训练循环。

## 4. 云端参考环境：与预置MindSpore分开

本轮预置镜像已包含PyTorch2.7.1+cpu与pytest8.3.5，没有安装新框架或修改依赖。
参考与MindSpore计算在独立进程中执行；无需为“隔离”重复安装已有框架。
对于其他镜像，若已有隔离的云端PyTorch环境，先检查版本；否则在`/home/ma-user/work`内建立独立
venv，只在那里安装CPU PyTorch（`weights_only=True`需较新版本）和NumPy。
所有参考计算也都在远程主机，绝不调用本地框架。
不能向预置MindSpore环境安装PyTorch。安装前记录当前Python路径和磁盘余量。

在远程参考环境执行：

```bash
export PYTHONPATH="$PWD/src:$PWD:$PYTHONPATH"
python -m tools.export_reference_bundle --branch params_ema --output experiments/reference_bundle
```

该工具核对官方checkpoint哈希，严格载入完整L模型，导出全部参数/buffer与两个输入：
32×48（动态空间尺寸）和64×64（构造时名义尺寸）。保存最终输出及首个RDG五层Swin、
主干入口/末端等中间值。输入小但网络仍是完整12组，未缩小模型。
`manifest.json`记录分支、来源、配置、软件版本、时间与产物哈希。
这是合成输入数值对照，不是Set14指标。

## 5. 回到MindSpore环境：转换和数值比较

```bash
python -m tools.convert_checkpoint --bundle experiments/reference_bundle --output weights/drct_l_x4_params_ema.ckpt
python -m tools.verify_alignment --checkpoint weights/drct_l_x4_params_ema.ckpt --bundle experiments/reference_bundle --mode pynative --trace --report experiments/alignment_pynative.json
```

转换默认只做LayerNorm gamma/beta名称对应，其他权重不转置；100%覆盖学习参数，
逐项比较所有位置索引和名义mask，拒绝未知项、非FP32或非finite参数。
保存ckpt后重新严格载入并逐参数检查序列化一致性。转换报告不等同于数值对齐通过。

运行FP32时设置`must_keep_origin_dtype`，禁用Conv/Matmul HF32；不启用AMP。
默认卷积后端为`matmul`，即slice/stack构造输入窗口并以FP32 GEMM计算同一卷积，
OIHW权重、bias、padding与梯度语义不变，参数和层数不变。
原生Ascend卷积在本环境未过门槛，切换新旧精度配置API均未消除偏差；
仅凭本轮证据不能断言是内部HF32还是其他原生kernel精度行为。
通过`--conv-backend native`可重现诊断，但不要据此输出“严格FP32已对齐”。
等价GEMM实现会增加展开窗口的临时内存，正式Set14评测前需实测峰值与速度。

输出max/mean absolute error、RMSE、relative L2与预先声明的候选allclose容差1e-4。
若失败先定位首个偏差层，不放宽阈值掩盖错误。完整网络容差仍需依据实测合理性确认。

PYNATIVE通过后，再在剩余时间允许时单独运行GRAPH：

```bash
python -m pytest tests/test_mindspore_primitives.py tests/test_mindspore_network.py -q -ra --ms-mode graph
python -m tools.verify_alignment --checkpoint weights/drct_l_x4_params_ema.ckpt --bundle experiments/reference_bundle --mode graph --report experiments/alignment_graph.json
```

GRAPH显式设置`jit_level='O0'`，只检查实际完整construct；逐层诊断目前是独立eager路径。两个固定空间尺寸的
运行不等于证明任意动态shape或所有Ascend算子已兼容。编译/算子失败需保留错误与环境。

## 6. 本轮明确未做的工作

未实现从零初始化与官方随机分布/随机数流完全一致。网络暂以完整预训练权重加载为前提，
转换及后续推理入口都拒绝不完整权重。Attention位置偏置的临时零初始化会被官方权重
覆盖，不用于训练或报告模型效果。DropPath保留官方组内共享概率，但跨框架不共享随机mask。

完成目标设备数值验证后，再固定Set14数据来源和哈希、实现与核验正式评测入口。
之后才考虑训练、预算估算和改进实验。
