# Stage4：预训练继续训练能力

状态：完整网络PYNATIVE的2+1步云端smoke及跨进程恢复通过，见[验收记录](verification/stage4.md)。
长训练尚未启动；本操作单命令需选择新目录，已存在的实验不会被覆盖。

## 配置与数据

原始配方来自固定版本的
`reference/DRCT/options/train/train_DRCT-L_SRx4_finetune_from_ImageNet_pretrain.yml`：
DF2K、HR256、batch4、mean L1、Adam(1e-5, beta1=0.9, beta2=0.99, eps=1e-8)、
weight_decay=0、EMA0.999，milestones125000/200000/225000/240000、gamma0.5、总250000步。
本阶段是已发布EMA权重的继续训练能力验证，不是原始ImageNet预训练或DF2K完整收敛复现。

优化器使用`ReferenceAdam`适配，不能直接替回`nn.Adam`：Ascend上的原生融合Adam
将epsilon加到未作偏差校正的sqrt(v)，与官方PyTorch语义不同，微小梯度测试已证实。
适配仍使用MindSpore的融合算子与状态，在每步传入`eps * sqrt(1-beta2^t)`，
使等价分母为`sqrt(v/(1-beta2^t)) + eps`。该适配固定于2.7.2，仅覆盖本配方的dense FP32、
weight_decay0；不声称支持稀疏/分组/offload/AMSGrad。

`configs/train_smoke.json`保留模型规模、FP32和优化参数，明确将batch降至1；
训练LR64/HR256。训练只取官方DIV2K训练0001–0004，验证只取官方valid0801–0802。
固定中心LR64验证块用于验证流程；其指标不能作为完整DIV2K验证结果。
训练HR若有不足4像素的尺寸余数，显式从底/右modcrop并记录；不重采样。
Set14不参与训练、验证或选参。

来源：[DIV2K官方数据与学术用途说明](https://data.vision.ee.ethz.ch/cvl/DIV2K/)。
数据工具使用HTTP Range抽取原始ZIP成员，核对ZIP CRC、记录每个PNG的SHA256及原始成员名；
归档ETag/长度仅用于标识服务器快照，不冒充整个ZIP的SHA256。
本地允许执行下载/文件整理工具，不解码图像；所有像素与模型处理都有云端保护。

## 有限云端验收

用户确认Notebook就绪后，先核对预置环境和NPU，不安装/升级框架。
SSH使用本任务专用known_hosts及`StrictHostKeyChecking=accept-new`；按2026-11-16用户要求
跳过人工指纹核对，不改全局SSH配置。

每个计算会话先执行：

```bash
. /usr/local/Ascend/cann-8.5.2/set_env.sh
cd /home/ma-user/work/drct_ms
export PYTHONPATH=$PWD/src:$PWD:$PYTHONPATH
```

```bash
python -m pytest tests/test_training.py -q --basetemp=/home/ma-user/work/drct_ms/experiments/stage4/pytest_tmp
timeout 900 python -u -m tools.train --steps 2 --output experiments/stage4/reference_adam_first
timeout 900 python -u -m tools.train --steps 1 --resume experiments/stage4/reference_adam_first/step_000002.ckpt --output experiments/stage4/reference_adam_resumed
```

验收要求：全部916个FP32参数梯度finite、loss finite、参数实际更新、优化器计数正确、
EMA可验证、保存与回读覆盖全部模型/Adam状态/EMA、独立进程恢复后再更新一步。
恢复前后live验证输出哈希/指标一致，下一个训练batch哈希一致。
额外小网络测试核对Adam首步解析解、EMA公式、LR边界和连续/恢复轨迹一致。

检查点JSON绑定配置、数据、预训练权重、源码哈希和下一样本编号；不支持静默部分加载。
采样/裁剪/增强由全局样本编号确定，可恢复数据位置。
完整网络含DropPath，暂不承诺跨进程随机流逐位续接；小网络精确恢复测试无随机层。
训练逐步保存JSONL，最终保存配置、哈希、验证逐图指标、耗时与NPU内存高水位。
计时包含每个梯度finite检查、优化器和EMA；参数变更审计在计时区间外，整体耗时另记。

只进行2+1步smoke；入口总步数限制20，长训练需根据实测速度和可用预算重新批准。
单步或任务超时、OOM、非finite均保留失败日志，不能静默更换native卷积或缩小模型。

API依据：[MindSpore2.7.2 Adam](https://www.mindspore.cn/docs/en/r2.7.2/api_python/nn/mindspore.nn.Adam.html)、
[value_and_grad](https://www.mindspore.cn/docs/en/r2.7.2/api_python/mindspore/mindspore.value_and_grad.html)。
