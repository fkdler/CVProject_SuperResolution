# 权重

官方 DRCT-L ×4：[DRCT-L_X4.pth](https://drive.google.com/file/d/1bVxvA6QFbne2se0CQJ-jyHFy94UOi3h5/view)。
下载后保存到本目录，不进入 Git。实际 SHA-256、顶层键和严格加载结果由
`python -m tools.inspect_checkpoint weights/DRCT-L_X4.pth` 写入
`docs/verification/checkpoint_audit.json`。文件名不证明模型身份。

注意：官方测试 YAML 请求 `params_ema`，`inference.py` 则读取 `params`。
必须根据实际 checkpoint 选择并记录分支，不能静默回退。
