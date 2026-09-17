# 固定手指的人体模型

这是独立的新模型，原 `human_robot_export` 和 FreeCAD 源模型保持不变。

- 双手的 8 个手指关节与 2 个拇指关节固定在零位。
- 每侧手腕保留屈伸、侧偏两个自由度，手指随手掌和手腕整体运动。
- 全身活动关节由 49 个降为 39 个；外形和质量保留，总质量 70 kg。
- 活动碰撞体保留胸腔、左右脚和左右前脚掌，并全部使用简单、稳定的长方体；其他部位只显示、不参与碰撞。
- 坐标仍为 X 向前、Y 向人体左侧、Z 向上。URDF 网格仍采用相对路径。

## 文件

- `human_description/urdf/human.urdf`：URDF，10 个手指关节设为 `fixed`。
- `mujoco/human.xml`：浮动基座版。
- `mujoco/human_fixed.xml`：固定骨盆版。
- `scripts/create_fixed_hands.py`：从原导出目录生成此版本的脚本。
- `scripts/validate_fixed_hands.py`：相对原模型检查网格、关节和姿势，并运行 MuJoCo 仿真。

MuJoCo 中固定连接使用不含 joint 的子 body 表达，因此移除了 10 个手指 hinge 及对应执行器；保留这些 body 的几何、质量和惯性。URDF 中保留对应 link 和固定关节。当前没有删除手部几何。

碰撞 STL 文件仍保留在资源目录中，但 URDF 和 MuJoCo 不再引用这些碰撞网格。
胸腔碰撞盒属于 `ChestLink`，尺寸为 `0.18 × 0.32 × 0.24 m`（前后 × 左右 × 高），
相对胸腔视觉网格略微内缩，减少肩部和腰部运动时的误碰撞。
每只脚由两块长方体组成：后脚板属于 `FootLink`，尺寸为
`0.165 × 0.09 × 0.02 m`；前脚板属于 `ForefootLink`，尺寸为
`0.065 × 0.09 × 0.02 m`。因此前脚板会随 `ForefootJoint` 一起旋转，
不会被错误地固定在后脚板上。零位时两块脚板之间沿 X 方向保留 `0.02 m`
的空间，脚底均位于 `z = 0`。MuJoCo 地面碰撞继续保留。

## ROS 2

ROS 包名称为 `human_fixed_hands_description`，可与原包一起安装。进入本目录执行：

```bash
source /opt/ros/humble/setup.bash
colcon build --base-paths human_description --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
ros2 launch human_fixed_hands_description display.launch.py
```

需要原模型说明中列出的 ROS 2 可视化依赖。关节滑块中只会出现 39 个活动关节。

## MuJoCo

```bash
python3 -m venv .venv
.venv/bin/pip install -r scripts/requirements.txt
.venv/bin/python scripts/simulate.py --fixed --hold --viewer --seconds 60
```

浮动版去掉 `--fixed`，自由基座另有 6 个自由度。执行器输入仍是力矩，参数为未标定的仿真初值；示例不含平衡或步态控制器。

## 重新生成

默认原模型目录为同级 `../human_robot_export`：

```bash
python3 scripts/create_fixed_hands.py
.venv/bin/python scripts/validate_fixed_hands.py
```

也可分别传入 `--source /path/to/human_robot_export`。重新生成会覆盖本变体的 URDF、XML、网格和执行器配置。

已验证：60 个网格与原版完全一致；两种 MuJoCo 模型各用六组姿势与原版（手指角度为零）及 URDF 对照；各仿真 2000 步无警告。ROS 2 Humble 实际发布 39 个关节状态、39 个动态 TF 和 11 个静态 TF。详情见两个 validation_report JSON 文件。
