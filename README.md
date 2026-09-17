# Human39：可复现的 PD 力矩人体动作模仿

FreeCAD 自建人体，39 个活动关节、70 kg、无肌肉。策略输出参考关节角修正，经显式 PD 生成限幅力矩；不施加骨盆扶持力。物理频率 500 Hz，控制频率 50 Hz，219 维观测、39 维动作。

本公开仓库保存实际训练源码、模型、冻结动作清单、环境记录和评估证据；关键权重位于 [v1.0.0 Release](https://github.com/Luk7-sb/human39-reproducible-training/releases/tag/v1.0.0)。KIT/AMASS、SMPL-H 参数及重定向轨迹不随仓库分发，需要使用者自行合法取得。

## 已记录的结果

| 检查点 | 作用 | 280 个训练片段完成数 |
|---|---|---:|
| v1_model_1098.pt | 最初三动作已验证参考 | 未做全库评估 |
| v2_model_1297.pt | 大库训练初始化，含优化器与归一化状态 | 33/280 |
| large_v1_model_1850.pt | 中期对照 | 261/280 |
| large_v1_model_4162.pt | 最终结果 | **280/280** |

最终逐片段奖励算术平均 **0.99126**。全部结果只统计第一次 episode，不拼接 reset 后的轨迹。257 条原始训练动作切为 280 段，每段不超过 6 秒；108 条原测试动作未用于训练。这是训练集拟合结果，**不是独立测试集泛化结果**。模型目前只有胸腔、左右分块脚掌碰撞体，不代表完成了全身自碰撞验证。

## 1. 获取代码和权重

```bash
gh repo clone Luk7-sb/human39-reproducible-training
cd human39-reproducible-training
python3 scripts/download_checkpoints.py
```

下载脚本使用本机 `gh auth`，不需要把 GitHub token 写入仓库。四个文件逐一校验 `checkpoints/manifest.json` 中的大小与 SHA256。加载 PyTorch 检查点前，应确认它来自可信仓库且通过校验。

## 2. 配置两个环境

重定向在 Python 3.12 / MuJoCo 3.3.3 上运行；训练在 Python 3.11 / Isaac Sim 5.1.0 / Isaac Lab 2.3.1 上运行。两者 NumPy/SciPy 版本不同，**不要共用一个虚拟环境**。

完整版本与安装步骤见 [environment/README.md](environment/README.md)。以下用 `RETARGET_PYTHON` 和 `TRAIN_PYTHON` 指定你自己的解释器，不依赖原服务器目录。

## 3. 从合法取得的源数据重建动作

准备原始 AMASS KIT NPZ（目录末级名称为 `KIT`），以及 SMPL-H 的 `smplh/{male,female,neutral}/model.npz`。具体所需文件、原始 SHA256 和训练/测试划分在 `configs/`。不需要旧 MyoFullBody 轨迹、CMU 或过渡动作。

```bash
export RETARGET_PYTHON=/your/retarget-env/bin/python
"$RETARGET_PYTHON" scripts/regenerate_dataset.py \
  --kit /your/AMASS/KIT \
  --body-models /your/body_models \
  --workers 4
"$RETARGET_PYTHON" scripts/prepare_training_assets.py
```

生成器只重建冻结清单中已准入的 257 个来源，保持原 280 个片段及顺序。它校验源文件、SMPL-H、模型 SHA256，以及 qpos/qvel/xpos/xquat 等数组的形状、dtype 与完整内容摘要。数值不一致时默认拒绝生成可用清单。`--allow-numeric-drift` 是显式的非精确复现选项，不能把这种输出称为数值完全相同。

NPZ 容器字节可能随打包实现变化，因此独立核验数值数组后，将新文件哈希写入本机 `configs/regenerated_manifest.json`；原始 `configs/large_motion_train_v1.json` 不改。该步骤不上传或下载任何动作数据。进度和核验结果写入本机 `reports/regeneration*`，不纳入版本控制。

## 4. 复跑最终权重评估

先确认已接受 Isaac Sim 的许可条款，再运行：

```bash
export TRAIN_PYTHON=/your/isaaclab-env/bin/python
export OMNI_KIT_ACCEPT_EULA=YES
"$TRAIN_PYTHON" training/train.py --headless --device cuda:0 \
  --num_envs 280 --run reproduce_final_eval \
  --checkpoint checkpoints/large_v1_model_4162.pt \
  --motion_manifest configs/regenerated_manifest.json --eval_steps 400
python3 scripts/verify_reproduction.py runs/reproduce_final_eval/evaluation.json
```

检查目标是 280/280 完成、相同存活时长和逐片段奖励差异不超过 0.005；阈值是跨硬件核查容差，不意味着保证跨硬件逐位一致。详细基准在 `reports/final/evaluation.json`。不同驱动、PhysX/GPU、库版本可能影响结果，需记录差异。

使用重定向环境渲染实际策略轨迹：

```bash
MUJOCO_GL=egl "$RETARGET_PYTHON" scripts/render_policy_rollout.py \
  runs/reproduce_final_eval/evaluation.npz --envs 0 1 2 197
```

没有可用 EGL 时可用已安装的 OSMesa：`MUJOCO_GL=osmesa`。原机器还需要为该命令单独设置 `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` 以避免 Anaconda 的旧 libstdc++ 冲突；不要无条件写入全局环境。

## 5. 复跑大库训练

实际运行从 v2 checkpoint 开始，65536 环境、32 rollout steps、64 minibatches、5 epochs、seed=42，每 50 轮保存。RTX PRO 6000 Blackwell 96 GiB、22 CPU 核预算、110 GiB 内存，训练稳定吞吐约 14 万环境步/s。

原实验按 12 小时预算停止，完成 **2866 个 PPO 迭代**（编号 1297–4162）。下面固定迭代数，消除不同机器运行速度对训练量的影响：

```bash
TRAIN_PYTHON="$TRAIN_PYTHON" bash training/run.sh reproduce_large 2866 65536 \
  checkpoints/v2_model_1297.pt configs/regenerated_manifest.json 0
```

或复用原时间预算：将迭代上限改为 `100000`，最后一项改为 `12`。同名已有 run 会被拒绝，避免覆盖。最后自动评估所有片段。没有大显存 GPU 时可降低环境数，但样本量、minibatch 与优化轨迹会变化，不属于同一训练配置。

PPO + GPU PhysX 训练并非严格确定性系统；权重、seed 和环境一致也不保证重新训练获得相同权重字节。这里提供的是**数据内容核验、最终权重评估复现和完整续训配方**，不宣称从随机初始化逐位重建最终权重。三动作前期训练权重作为可追溯初始化提供。

## 目录与变更来源

- `training/`：可移植运行代码；只将启动器的绝对 Python 路径替换为 `TRAIN_PYTHON`。
- `provenance/original_training/`：实际长期 run 启动时保存的三份源码，保持原样。
- `models/source/`：用户提供的人体模型及网格；`prepare_training_assets.py` 单独生成 PhysX 副本。
- `configs/large_motion_train_v1.json`：原冻结清单；`reproduction_assets.json`：逐数组核验信息；`source_split.json`：仅来源 ID 的分组。
- `environment/`：实测版本记录与安装说明。
- `reports/{baseline,midpoint,final}/`：可分享的评估 JSON、配置及最终训练指标，不含动作 NPZ。
- `docs/VALIDATION.md`：本发布包实际执行的复现检查及边界。

权利和数据获取说明见 [DATA_AND_MODEL_TERMS.md](DATA_AND_MODEL_TERMS.md)。仓库不包含 SSH 密码、会话日志、机器交接文件或 GitHub 凭据。
