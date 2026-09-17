# 环境记录

## 训练环境

实测 Python 3.11.15，Isaac Sim 5.1.0.0，Isaac Lab release 2.3.1，源码 commit `5c2ec81cb17532d32f7922dd7fcaae40d123b71a`（采集时工作树干净）。其 editable 包 metadata 显示 `isaaclab=0.48.0`、`isaaclab-rl=0.4.4`，不是另外一个 release；不要用这两个 metadata 数字去猜 PyPI 发行版本。

核心依赖：torch 2.7.0+cu128、numpy 1.26.4、scipy 1.15.3、rsl-rl-lib 3.0.1、gymnasium 1.2.0。完整已安装包版本见 `training-observed.json`，它是运行环境盘点，不是经求解的全量安装锁文件。GPU 驱动 580.142，RTX PRO 6000 Blackwell Server Edition 97887 MiB，容器 CPU 22 核、内存 110 GiB。

按 [Isaac Lab 2.3.1 官方 pip 安装说明](https://isaac-sim.github.io/IsaacLab/v2.3.1/source/setup/installation/pip_installation.html)建立 Python 3.11 环境并安装 Isaac Sim 5.1.0。该方式要求兼容的 NVIDIA 驱动和 GLIBC 2.35+。示意命令：

```bash
python3.11 -m venv /your/isaaclab-env
source /your/isaaclab-env/bin/activate
python -m pip install --upgrade pip
python -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/isaac-sim/IsaacLab.git /your/IsaacLab
cd /your/IsaacLab
git checkout 5c2ec81cb17532d32f7922dd7fcaae40d123b71a
./isaaclab.sh --install rsl_rl
python -m pip install rsl-rl-lib==3.0.1 numpy==1.26.4 scipy==1.15.3 gymnasium==1.2.0 tensorboard==2.21.0 gitpython==3.1.62
```

最后切回本仓库。原运行环境已恢复并实测通过；**本次发布没有在空机器上重新安装 NVIDIA 全部依赖**。官方扩展下载、驱动和系统图形库仍是外部条件。复现实测采用现有相同环境与独立工作目录，详情在 `docs/VALIDATION.md`。

## CPU 重定向/渲染环境

实测 Python 3.12.7，MuJoCo 3.3.3、NumPy 2.2.6、SciPy 1.18.0；记录见 `retarget-observed.json`。这些是实际包 metadata，优先匹配记录，勿自行使用训练环境的 NumPy/SciPy 替代。

```bash
python3.12 -m venv /your/retarget-env
/your/retarget-env/bin/python -m pip install -r environment/requirements-retarget.txt
```

源数据重建使用单线程 BLAS、4 个 CPU 工作进程，不调用训练 GPU。若某个记录版本不在使用者的包索引中可用，需要配置原版本的包源或明确记录替代版本，并以逐数组校验判断数据是否一致，不能默默宣称版本相同。
