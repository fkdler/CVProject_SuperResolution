# 项目与工作规范

- 本项目是课程大作业：用 MindSpore 2.7.2 在 Ascend 910B4 上复现 DRCT-L ×4。
- 优先级：结构正确 → 权重/数值对齐 → Set14 推理 → 训练 → 改进与消融。
- 开始工作时先读 `docs/progress.md`、`docs/source_analysis.md`、`docs/migration.md`。
- 官方参考源码独立存于 `reference/DRCT`、`reference/BasicSR`，版本由
  `reference/manifest.json` 固定。不要直接修改参考仓库；模型依据真实源码与 checkpoint。
- L 是 12 RDG、每组 5 Swin，主宽度 180，gc=32。不能按 depths 值误实现为 72 Swin。
- MindSpore 代码放 `src/drct_ms/`，不能用 PyTorch 或 NumPy 推理冒充 MindSpore。
- 先 FP32。API 名称相似不代表语义相同；先验证布局、LN epsilon、GELU、QKV、
  相对位置、shift/mask、残差缩放及 PixelShuffle，再优化。
- Ascend当前默认`conv_backend='matmul'`（slice/stack/GEMM等价卷积，OIHW参数不变）。
  native卷积即使禁用HF32仍未通过完整对齐，不能静默切回native。相关失败/通过证据见
  `docs/verification/ascend_stage2/`。GRAPH基线使用O0，精度设置用2.7.2的op_precision API。
- 评测主协议按发布 YAML：量化 uint8、BT.601 Y、每边裁 4 HR 像素。
  论文 v5 的 RGB/裁8 文本与代码冲突，单独记录诊断结果，不能混用或掩盖。
- Set14 是测试集，不作训练或改进方案的选参集。保存逐图指标、数据/权重哈希、环境与配置。
- Stage3已完成14图EMA基线：Y/裁4，29.57982569dB / 0.8008819660；与原始PT对齐通过。
  数据源与commit固定在`configs/set14_source.json`，证据见`docs/verification/stage3.md`。
  论文0.8025 SSIM未逐项完全对应，不替换数据/协议/分支凑数；`params`仅为独立诊断。
- 本地与 ModelArts 环境分开。云端先核对 Conda、Python、MindSpore、CANN 和设备，
  保留预置环境，项目和输出存 `/home/ma-user/work`。
- 2026-11-14用户更新：所有模型/数值计算只在云端执行；本地仅源码编辑、静态检查、
  文件整理和传输。不要在本地安装、导入或运行PyTorch、MindSpore或其他推理框架，
  不使用本地CPU/GPU/NPU做实验。既有本地参考环境不再使用。
- 需要启动既有Notebook和SSH连接时先通知用户，待其确认实例就绪；不要自行启动实例。
- 总预算约600元，不创建额外付费实例；长时间或大规模训练先实测并估算费用，等待用户批准。
- 不读取、输出或提交 SSH 私钥。保留 SSH 主机验证；不在代码和日志中写凭据。
- 数据集、大权重、缓存及大日志不进 Git；小型审计摘要和约50KB的基础模块参考数据可跟踪。
- 本地只允许 `python -S tools/static_check.py` 等纯文本静态检查，不运行pytest。
  数值测试、参考生成、转换和推理入口均应有ModelArts Linux ARM64/work目录保护。
  云端验证命令见 `docs/cloud_validation.md`。静态通过不代表目标运行成功。
- 每个关键阶段更新 `docs/progress.md`，报告实际完成、验证证据、待办与不确定项。
