# Stage3：Set14 ×4 推理复现

更新：2026-11-16。状态：**已执行完成**。主基线29.57982569dB / 0.8008819660，
14图MS/PT对照通过，完整数据、协议、资源与残余论文差异见[结果记录](verification/stage3.md)。
以下是本次固定方案与复现命令，不需要在已通过后无理由重跑。

## 固定方案

- 数据来自SelfExSR作者仓库，固定commit为`8f6dd8c1d20cb7e8792a7177b4f6fd677633f598`，
  目录`data/Set14/image_SRF_4`，使用全部14对现成HR/LR PNG，不重采样。
- 原始28个文件约5.1MiB，下载只处理文件和PNG头信息；云端再验证解码、精确4倍配对和哈希。
- `configs/set14_source.json`固定来源；`data/Set14/manifest.json`记录每个文件URL、哈希和头信息。
- 单张灰度PNG按OpenCV IMREAD_COLOR转换成三通道，与BasicSR读取路径一致。
- 公开来源的预生成LR并不构成独立证明其逐项等同于原作者当年测试数据；此项需结合官方模型结果判断。

## 协议与运行配置

主协议：发布YAML的BT.601 Y、每边裁4个HR像素；诊断协议：论文v5文本的RGB、每边裁8。
先clamp、乘255、round、转uint8，再计算指标；逐图PSNR/SSIM算术平均。
RGB输入NCHW FP32，右侧/底部reflect补齐16倍数，输出去除相应×4 padding。
不tile、不self-ensemble、不替换测试数据。

默认分支固定`params_ema`。PyTorch采用完整官方DRCT-L，运行在云端CPU；
MindSpore采用此前通过FP32对齐的完整L、默认`matmul`等价卷积，在Ascend PYNATIVE执行。
不为14种图像尺寸逐一启动GRAPH编译；上一阶段GRAPH两种合成输入通过不能当成本轮GRAPH Set14结果。

## 执行顺序

以下命令仅在`/home/ma-user/work/drct_ms`的云端执行，保留预置框架环境。

```bash
. /usr/local/Ascend/cann-8.5.2/set_env.sh
export PYTHONPATH="$PWD/src:$PWD:$PYTHONPATH"
python -m tools.validate_evaluation --dataset data/Set14
```

先将指标、量化、padding与固定版本BasicSR实际函数/Torch reflect逐项对照。
不通过则不开始正式评测。BasicSR仅在内存中移除注册装饰器，函数体与文件保持原样并验证哈希。

先用较大图做资源检查（明确不是完整benchmark），不以测试图指标选模型：

```bash
python -m tools.evaluate_set14 --framework mindspore --checkpoint weights/drct_l_x4_params_ema.ckpt --dataset data/Set14 --output experiments/stage3/ms_resource_smoke --ids img_002
```

确认资源后运行完整14图：

```bash
python -m tools.evaluate_set14 --framework torch --checkpoint /home/ma-user/work/DRCT-L_X4.pth --dataset data/Set14 --output experiments/stage3/torch_ema
python -m tools.evaluate_set14 --framework mindspore --checkpoint weights/drct_l_x4_params_ema.ckpt --dataset data/Set14 --output experiments/stage3/mindspore_ema
python -m tools.compare_set14 --torch-run experiments/stage3/torch_ema --ms-run experiments/stage3/mindspore_ema --output experiments/stage3/comparison.json
```

产物：逐图原始浮点输出、SR PNG、PSNR/SSIM CSV、完整JSON、时间与内存统计。
原始数据/大输出不进入Git，小型manifest和结果摘要回传到`docs/verification/`。
已有实验不覆盖；重跑需选择新目录，错误时保留已完成图像和报告。

## 预先声明的核验条件

1. 同一数据manifest/权重来源/分支、相同模型与前后处理、14图齐全。
2. 指标实现对BasicSR候选绝对误差≤1e-12；padding与量化逐元素相等。
3. 真实LR输入的raw输出沿用Stage2门槛：atol=rtol=1e-4。
4. 量化后逐图两框架PSNR差≤0.01dB、SSIM差≤1e-4；报告所有真实差异，不要求PNG完全一致。
5. 单独比较论文参考29.54dB。若未达到，检查数据、checkpoint、协议及数值因素，不改变上述门槛凑结果。

MindSpore内存统计使用2.7.2的runtime内存池接口，记录tensor实际分配峰值和pool保留峰值，
不将其冒充驱动总HBM。单次forward时间包含新尺寸算子编译和主机取回，不能称为稳态延迟，
也不将云端CPU参考与Ascend速度作同硬件性能比较。

本轮计划约30–45分钟，标称约10.4–15.6元；实际取决于传输/算子编译，不是云账单。
不创建新实例、不安装本地框架、不启动训练。结束前保存结果并检查验证进程退出。
