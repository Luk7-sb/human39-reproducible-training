# 与实际运行快照的关系

`provenance/original_training/` 来自已结束的 `pd_large_v1_65536_12h` run 保存的源码。

- `training/train.py` 与原始快照字节一致。
- `training/human39_env.py` 与原始快照字节一致。
- `training/run.sh` 仅新增 `TRAIN_PYTHON` 环境变量、用它替换原服务器绝对解释器路径，并将无参数默认 run 名改为 `reproduction`。奖励、控制、物理和训练超参数未修改。
- 重定向脚本是包含浮点采样末帧修复的最终版本。最终原始训练数据有部分在修复前生成；发布验证从该脚本重新生成全部 257 个来源，逐数组核验结果见 VALIDATION.md。
- `configs/large_motion_train_v1.json` 与原训练冻结清单一致。`configs/regenerated_manifest.json` 是本机重建后写入新 NPZ 哈希的文件，不提交到 Git。
- 模型原件和参考文件保留；PhysX 适配副本由脚本再生成，不在版本控制中保存引擎 USD 缓存。

复现运行请使用未用过的 run 名。评估若重复使用相同目录，`train.py` 本身会覆盖评估输出；README 命令只用于新目录。启动器 `run.sh` 则拒绝已有 state.txt 的同名训练 run。
