# 运行脚本

第一阶段开发验证命令位于 `tools/`，通过 `python -m tools.<name>` 调用。
正式Stage3评测通过`python -m tools.evaluate_set14`调用，云端比较入口为
`python -m tools.compare_set14`；本目录保留给后续薄脚本和Stage4训练入口。
