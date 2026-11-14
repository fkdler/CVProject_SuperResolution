# 权重

官方 DRCT-L ×4：[DRCT-L_X4.pth](https://drive.google.com/file/d/1bVxvA6QFbne2se0CQJ-jyHFy94UOi3h5/view)。
下载后保存到本目录，不进入 Git。实际 SHA-256、顶层键和严格加载结果由
`python -m tools.inspect_checkpoint weights/DRCT-L_X4.pth` 写入
`docs/verification/checkpoint_audit.json`。文件名不证明模型身份。

注意：官方测试 YAML 请求 `params_ema`，`inference.py` 则读取 `params`。
必须根据实际 checkpoint 选择并记录分支，不能静默回退。

已在云端转换并备份：`drct_l_x4_params_ema.ckpt`，对应元数据为同名`.json`。
SHA-256：`146cfca01fd10398fc0ce1e52534b64f76af16dcebcfd9ae03235bcebbbb716b`。
源分支为`params_ema`，全部916个学习参数张量/84个几何buffer核查通过。
PYNATIVE和GRAPH O0对齐结果见`docs/verification/stage2.md`；默认等价卷积后端为`matmul`。
本地仅存储备份，不能在本地导入或执行框架。
