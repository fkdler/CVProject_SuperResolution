# 数据

此目录存放本地数据，内容不进入 Git。尚未下载 Set14 / DIV2K / DF2K。

后续建议：`Set14/GTmod4`、`Set14/LRbicx4`、`DIV2K/train/HR`、`DIV2K/train/LRbicx4`。
Set14 仅用于最终测试，不用于训练或选择改进版本。数据准备需记录来源、文件清单、
SHA-256、配对规则、modcrop 和 bicubic 实现。不能用 PIL / OpenCV 的 bicubic
直接替代 MATLAB 兼容降采样并宣称同一数据。
