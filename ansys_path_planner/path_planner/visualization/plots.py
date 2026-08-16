"""绘图辅助：路径密度等级配色与 colorbar 管理。"""
import numpy as np

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
                      legend_kwargs=None, skip_mask=None):
    """
    把路径段按密度等级着色绘制到 3D 坐标轴上。

    skip_mask：长度 = len(xyz) - 1 的布尔数组，为 True 的段不绘制
    （用于跳过穿越空区的路径段）。

    返回实际使用到的等级集合（可用于构造图例）。
    """
    xyz = np.asarray(xyz, dtype=float)
    levels = density_levels(spacing_values)
    used = set()

    for i in range(len(xyz) - 1):
        if skip_mask is not None and skip_mask[i]:
            continue
        level = int(max(levels[i], levels[i + 1]))
        label = (
            config.DENSITY_LABELS[level]
            if attach_labels and level not in used
            else None
        )
        ax.plot(
            xyz[i:i + 2, 0],
            xyz[i:i + 2, 1],
            xyz[i:i + 2, 2],
            linewidth=linewidth,
            color=config.DENSITY_COLORS[level],
            alpha=alpha,
            solid_capstyle="round",
            label=label,
        )
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
