"""绘图辅助：路径密度等级配色与 colorbar 管理。"""
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from path_planner import config


def density_levels(spacing_values):
    """把路径间距映射为 0~4 的密度等级（间距小 = 等级高）。"""
    spacing = np.asarray(spacing_values, dtype=float)
    s_min, s_max = float(np.nanmin(spacing)), float(np.nanmax(spacing))
    if abs(s_max - s_min) < 1e-12:
        density = np.full(len(spacing), 0.5)
    else:
        density = (s_max - spacing) / (s_max - s_min)
    return np.clip((density * 5).astype(int), 0, 4)


def plot_density_path(ax, xyz, spacing_values, linewidth=2.2, alpha=0.95,
                      attach_labels=True, with_legend=False,
                      legend_kwargs=None, skip_mask=None, max_segments=None):
    """
    把路径段按密度等级着色绘制到 3D 坐标轴上。

    skip_mask：长度 = len(xyz) - 1 的布尔数组，为 True 的段不绘制
    （用于跳过穿越空区的路径段）。
    max_segments：路径段过多时等间隔抽稀，只绘制约 max_segments 段，
    避免十几万段路径糊成一团。

    返回实际使用到的等级集合（可用于构造图例）。

    性能说明：按密度等级把路径段合并为 LineCollection 一次绘制
    （最多 5 个图元），避免逐段 ax.plot() 在大节点数时创建海量对象。
    """
    xyz = np.asarray(xyz, dtype=float)
    levels = density_levels(spacing_values)

    segments = [[] for _ in range(5)]
    n_segments = len(xyz) - 1
    step = 1
    if max_segments and n_segments > max_segments:
        step = max(1, int(np.ceil(n_segments / max_segments)))
    for i in range(0, n_segments, step):
        if skip_mask is not None and skip_mask[i]:
            continue
        level = int(max(levels[i], levels[i + 1]))
        segments[level].append(xyz[i:i + 2])

    used = set()
    for level in range(5):
        segs = segments[level]
        if not segs:
            continue
        collection = Line3DCollection(
            segs,
            linewidths=linewidth,
            colors=[config.DENSITY_COLORS[level]],
            alpha=alpha,
        )
        try:
            collection.set_capstyle("round")
        except Exception:
            pass
        if attach_labels:
            collection.set_label(config.DENSITY_LABELS[level])
        ax.add_collection3d(collection)
        used.add(level)

    if with_legend:
        kwargs = {"loc": "upper right", "fontsize": 9, "frameon": True}
        if legend_kwargs:
            kwargs.update(legend_kwargs)
        ax.legend(**kwargs)

    return used


def attach_colorbar(fig, ax, scatter, label=None, shrink=0.68, pad=0.08,
                    current_cbar=None):
    """
    添加或替换 colorbar，返回新 colorbar 对象。

    传入 current_cbar 会在添加新色条前移除旧色条，
    避免多次生成路径时色条在图中堆积。
    """
    if current_cbar is not None:
        try:
            current_cbar.remove()
        except Exception:
            pass

    kwargs = {"shrink": shrink, "pad": pad}
    if label:
        kwargs["label"] = label
    return fig.colorbar(scatter, ax=ax, **kwargs)
