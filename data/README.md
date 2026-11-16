# 数据

此目录存放本地数据，内容不进入Git。Set14已下载，DIV2K/DF2K尚未准备。

Set14固定来源为SelfExSR作者仓库commit `8f6dd8c1d20cb7e8792a7177b4f6fd677633f598`，
保存原始`image_SRF_4`的14对PNG，未重新采样；`Set14/manifest.json`记录URL/哈希/头信息。
同一清单已跟踪至`docs/verification/stage3/dataset_manifest.json`，实测结果见Stage3记录。
本地只做文件准备/存储，图片解码和所有数值处理均在云端执行。

后续建议：`Set14/GTmod4`、`Set14/LRbicx4`、`DIV2K/train/HR`、`DIV2K/train/LRbicx4`。
Set14 仅用于最终测试，不用于训练或选择改进版本。数据准备需记录来源、文件清单、
SHA-256、配对规则、modcrop 和 bicubic 实现。不能用 PIL / OpenCV 的 bicubic
直接替代 MATLAB 兼容降采样并宣称同一数据。
