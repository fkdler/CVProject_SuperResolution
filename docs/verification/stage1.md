# 第一阶段执行记录

日期：2026-11-14，Windows本地执行。

## 交付物

1. 官方DRCT与BasicSR的独立Git checkout、commit锁定及LF规范化源码SHA-256。
2. 源码分析、迁移清单、项目README、AGENTS和进度记录。
3. 本地独立CPU参考环境、实际网络参数与全部state_dict形状审计。
4. 官方DRCT-L ×4权重下载及安全加载审计。
5. 首批真实MindSpore Cells与测试；约50KB的官方PyTorch参考输出/梯度数据。

## 执行结果

```text
python -m tools.fetch_reference
  DRCT    c0a8b51e69241cf7f8ae598053235ff498517935
  BasicSR 5668ba75eb8a77e8d2dd46746a36fee0fbb0fdcd

python -m tools.inspect_reference
  Base: 14,139,579 parameters, 6 RDG, 30 Swin, 508 state_dict entries
  L:    27,580,719 parameters, 12 RDG, 60 Swin, 1000 state_dict entries

python -m tools.inspect_checkpoint weights/DRCT-L_X4.pth
  params:     strict_load passed; 1000 entries
  params_ema: strict_load passed; 1000 entries
  both: no missing/unexpected/shape/buffer mismatch; floating tensors finite

python -m pytest -q -ra
  14 passed, 1 skipped
  skipped module: MindSpore not installed locally

python -m compileall -q src tools tests
  passed
```

CPU环境与完整结果分别见 `reference_environment.txt`、`reference_audit.json`、
`checkpoint_audit.json`、`local_pytest.txt`。源码审计适配器仅在内存中加载官方定义及
辅助函数，不安装或声称运行完整BasicSR训练/评测栈。

权重文件485,570,697字节，SHA-256：
`a99044c0275699d1a296ae21b8f322fa8c65d7b9be2213ee2a3dcc280ab8d64b`。
文件源于官方README链接。首次下载超时，后续断点续传完成；最终安全加载与结构检查通过。

## 环境与限制

默认本地Python为3.13.14；新建项目`.venv-reference`安装CPU PyTorch 2.15.0+cpu。
仅只读探测了已有Conda环境，未在其他项目环境中安装依赖或执行实验。
网络获取通过提升权限运行；沙箱进程后续发生`setup refresh`初始化错误，
本地验证因此也通过提升权限执行，操作范围为本项目目录与项目专用环境。
没有更改全局Git信任目录，参考仓库只按命令指定safe.directory。

当前通过的是CPU参考结构和几何测试。MS Cells尚未运行，完整MS网络、转换和Set14
评测尚未完成；本阶段没有PSNR/SSIM实测结果，也没有云端连接、训练或新增付费实例。
用户原有Notebook的运行和计费状态未查询。

下一阶段首先运行已有MS基础模块测试，再实现并逐级对齐attention/Swin/RDG与完整网络。
