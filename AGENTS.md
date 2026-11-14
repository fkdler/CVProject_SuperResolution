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
- 评测主协议按发布 YAML：量化 uint8、BT.601 Y、每边裁 4 HR 像素。
  论文 v5 的 RGB/裁8 文本与代码冲突，单独记录诊断结果，不能混用或掩盖。
- Set14 是测试集，不作训练或改进方案的选参集。保存逐图指标、数据/权重哈希、环境与配置。
- 本地与 ModelArts 环境分开。云端先核对 Conda、Python、MindSpore、CANN 和设备，
  保留预置环境，项目和输出存 `/home/ma-user/work`。
- 总预算约600元，不创建额外付费实例；长时间或大规模训练先实测并估算费用，等待用户批准。
- 不读取、输出或提交 SSH 私钥。保留 SSH 主机验证；不在代码和日志中写凭据。
- 数据集、大权重、缓存及大日志不进 Git；小型审计摘要和约50KB的基础模块参考数据可跟踪。
- 本地验证：项目 CPU 虚拟环境运行 `python -m pytest -q -ra`，
  `python -m tools.fetch_reference`，`python -m tools.inspect_reference`。
  MindSpore 缺失导致的 skip 必须在进度中说明；语法通过不代表目标运行成功。
- 每个关键阶段更新 `docs/progress.md`，报告实际完成、验证证据、待办与不确定项。
