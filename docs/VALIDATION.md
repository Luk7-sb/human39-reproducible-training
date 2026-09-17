# 发布前复现验证（2026-09-17）

## 实际执行并通过

1. 使用发布包内的最终重定向脚本，从本机合法持有的原始 KIT/SMPL-H 输入，在新的发布工作目录重新生成全部 **257 个来源、280 个片段**。qpos/qvel/xpos/xquat/frequency/joint_names/body_names 的形状、dtype 及完整数组 SHA256 均与原训练数据一致；没有启用 `--allow-numeric-drift`。
2. 使用发布包中的 `prepare_training_assets.py` 重新生成 PhysX URDF，与原训练模型 SHA256 一致：`a30f51bd667cab3b294c586c9fea8333cef4224d39081eaa61054fd30de37a04`。
3. 在原服务器的**独立工作目录**中运行发布包的训练代码，使用新生成的数据和 manifest、最终 `large_v1_model_4162.pt`，280 个环境、400 步上限进行确定性首次 episode 评估。训练模型的 USD 缓存由该目录重新生成。
4. 结果 **280/280 完成**，每片段时长与原评估一致，逐片段奖励最大绝对差异 **0.0**。原结果在 `reports/final/evaluation.json`，独立复跑结果在 `reports/reproduction/evaluation.json`。
5. Python 编译、shell 语法、训练代码与原始 run 快照一致性、训练/测试来源不相交检查通过；最终权重本地与远端 SHA256 一致。上传前检查 Git 跟踪列表，不含动作/身体参数 NPZ、私密路径配置、会话交接文件或凭据。

## 没有验证的范围

- 没有在空白操作系统上重新安装整个 NVIDIA 软件栈；本次复跑复用了已记录的相同 Isaac Lab/Python 环境。
- 没有再训练一次 12 小时并要求权重字节完全相同；GPU PhysX/PPO 不保证逐位确定性。
- 没有进行 108 条独立测试动作或外骨骼穿戴后的泛化评估。
- 280/280 是同训练集片段的确定性完成率，不等于鲁棒性、真实机器人安全性或完整自碰撞认证。
