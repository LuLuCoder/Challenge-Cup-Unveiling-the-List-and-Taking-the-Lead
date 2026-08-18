"""第五章配图公共工具：字体、保存、路径批量绘制等。"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面出图
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def output_dir():
    """输出目录：优先取环境变量 FIG_OUT，否则默认 figures/output。"""
    env = os.environ.get("FIG_OUT")
    if env:
        d = Path(env)
    else:
        d = Path(__file__).resolve().parent / "output"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(fig, name):
    """保存当前图为 PNG（300 dpi），并关闭释放内存。"""
    path = output_dir() / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("已生成:", path)
    return path


def set_axes_equal_3d(ax):
    """使 3D 坐标轴三个方向等比例。"""
    lims = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = lims[:, 1] - lims[:, 0]
    r = float(np.max(spans)) / 2.0
    centers = lims.mean(axis=1)
    ax.set_xlim3d(centers[0] - r, centers[0] + r)
    ax.set_ylim3d(centers[1] - r, centers[1] + r)
    ax.set_zlim3d(centers[2] - r, centers[2] + r)


def path_line_collection(xyz, colors, skip=None, lw=0.8, alpha=0.9):
    """把路径点序列拆成两段线，返回 Line3DCollection（可跳过空区段）。

    xyz: (N,3)；colors: (N,) 或标量；skip: (N-1,) 布尔，True 表示该段断开。
    """
    segs = []
    cols = []
    for i in range(len(xyz) - 1):
        if skip is not None and skip[i]:
            continue
        segs.append([xyz[i], xyz[i + 1]])
        if np.ndim(colors) == 0:
            cols.append(colors)
        else:
            cols.append(colors[i])
    lc = Line3DCollection(segs, colors=cols, linewidths=lw, alpha=alpha)
    return lc


def subsample_indices(n, limit, seed=0):
    """均匀随机抽样的下标。"""
    if n <= limit:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, limit, replace=False))
