"""5.2 应力场驱动的连续纤维路径规划算法配图。"""

import sys
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from common import (
    path_line_collection,
    save,
    set_axes_equal_3d,
    subsample_indices,
)

PROJECT = Path(__file__).resolve().parent.parent
APP = PROJECT / "ansys_path_planner"
DATA = PROJECT / "data"
sys.path.insert(0, str(APP))

from path_planner import config
from path_planner.analysis.path_planning import generate_layer_path
from path_planner.analysis.shape_signature import canonical_points, compute_signature
from path_planner.analysis.stress import merge_ansys_files_data
from path_planner.analysis.template_library import (
    find_best_template,
    map_from_template,
)
from path_planner.parsers.step_mesh import mesh_step_file

STEP_FILE = PROJECT / "ansys" / "零件1.STEP"


def _stress_files():
    return {
        comp: str(DATA / name)
        for name, comp in config.STRESS_FILE_MAP.items()
    }


@lru_cache(maxsize=1)
def load_merged():
    """加载 13.8 万节点真实数据并完成应力分析。"""
    return merge_ansys_files_data(
        str(DATA / "node_coordinates_s.csv"), _stress_files()
    )


@lru_cache(maxsize=1)
def load_part1_mesh():
    """零件1 STEP 网格化结果（节点点云）。"""
    mesh = mesh_step_file(str(STEP_FILE))
    return mesh


@lru_cache(maxsize=1)
def load_template_mapping():
    """零件1 网格点云 → 模板匹配 → 应力/路径映射 → 层式重规划。"""
    mesh = load_part1_mesh()
    entry, sim = find_best_template(
        compute_signature(mesh["points"]), threshold=0.0
    )
    mapped_geom, _mapped_path = map_from_template(entry["path"], mesh["node_df"])
    path_data, _ = generate_layer_path(
        mapped_geom,
        "Maximum_Principal",
        percentile=config.DEFAULT_PERCENTILE,
        n_layers=int(entry.get("n_layers") or config.DEFAULT_LAYERS),
    )
    return mesh, entry, sim, mapped_geom, path_data


# ---------------------------------------------------------------


def fig_5_1_flowchart():
    """图5-1 算法总体框架图（自绘流程）。"""
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.axis("off")
    stages = [
        "① 应力场分析\n主应力 / 主方向 / von Mises",
        "② 层式路径生成\n自适应间距 / zigzag / 空区检测",
        "③ 模板化与映射复用\n形状签名 / 相似度 / 映射",
        "④ 输出组织\n标准化路径表",
    ]
    x0, y0, w, h, gap = 0.02, 0.28, 0.20, 0.44, 0.045
    for i, text in enumerate(stages):
        x = x0 + i * (w + gap)
        box = FancyBboxPatch(
            (x, y0), w, h,
            boxstyle="round,pad=0.012",
            fc="#EAF2FB", ec="#2C7FB8", lw=1.6,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y0 + h / 2, text, ha="center", va="center",
                fontsize=11)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch(
                (x + w + 0.004, y0 + h / 2), (x + w + gap - 0.004, y0 + h / 2),
                arrowstyle="-|>", mutation_scale=18, color="#2C7FB8", lw=1.8,
            ))
    ax.text(0.5, 0.92, "输入：节点云 + 应力场（或 STEP 几何）",
            ha="center", fontsize=11, color="#444444")
    ax.text(0.5, 0.08, "输出：携带层号 / 段类型 / 间距 / 优先级的路径表",
            ha="center", fontsize=11, color="#444444")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, "fig5_1_framework")


def fig_5_2_stress_field():
    """图5-2 构件应力场与主应力方向可视化。"""
    merged = load_merged()
    fig = plt.figure(figsize=(13, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    idx = subsample_indices(len(merged), 60000, seed=1)
    d = merged.iloc[idx]
    sc = ax1.scatter(
        d["X"], d["Y"], d["Z"],
        c=d["Von_Mises"], cmap="turbo", s=3, alpha=0.85,
    )
    ax1.set_title("(a) von Mises 应力分布", fontsize=12)
    fig.colorbar(sc, ax=ax1, shrink=0.6, pad=0.08, label="von Mises (Pa)")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    idx2 = subsample_indices(len(merged), 1500, seed=2)
    q = merged.iloc[idx2]
    ax2.scatter(q["X"], q["Y"], q["Z"], c="#9EA7B3", s=2, alpha=0.5)
    ax2.quiver(
        q["X"], q["Y"], q["Z"],
        q["Principal_VX"], q["Principal_VY"], q["Principal_VZ"],
        length=0.0008, normalize=True, color="#D62728", alpha=0.75, lw=0.6,
    )
    ax2.set_title("(b) 最大主应力方向场（局部采样）", fontsize=12)
    for ax in (ax1, ax2):
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.tick_params(labelsize=7)
        set_axes_equal_3d(ax)
    fig.tight_layout()
    save(fig, "fig5_2_stress_field")


def fig_5_3_direction_continuity():
    """图5-3 方向场连续性处理前后对比（真实 ANSYS 应力方向场）。"""
    from path_planner.analysis.stress import _continuize_directions

    merged = load_merged()
    tensors = np.zeros((len(merged), 3, 3), dtype=float)
    tensors[:, 0, 0] = merged["SX"].to_numpy(float)
    tensors[:, 1, 1] = merged["SY"].to_numpy(float)
    tensors[:, 2, 2] = merged["SZ"].to_numpy(float)
    tensors[:, 0, 1] = tensors[:, 1, 0] = merged["SXY"].to_numpy(float)
    tensors[:, 1, 2] = tensors[:, 2, 1] = merged["SYZ"].to_numpy(float)
    tensors[:, 0, 2] = tensors[:, 2, 0] = merged["SXZ"].to_numpy(float)
    _vals, vecs = np.linalg.eigh(tensors)
    raw = vecs[:, :, 2].copy()          # 未做连续性化的原始方向场
    smooth = _continuize_directions(raw)  # 连续性化后的方向场

    # 标记"方向跳变"：原始方向与连续性化方向反向（点积 < 0）
    flip = np.einsum("ij,ij->i", raw, smooth) < 0

    # 平面投影（XY），逐箭头归一化
    xy = merged[["X", "Y"]].to_numpy(float)
    stress = merged["Von_Mises"].to_numpy(float)

    idx_bg = subsample_indices(len(merged), 50000, seed=11)
    idx_ar = subsample_indices(len(merged), 2600, seed=12)

    def norm_xy(v):
        v = v[:, :2]
        n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        return v / n

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))

    # ---- 渲染：坐标统一为 mm，图例代替散乱文字标注 ----
    for ax, title, arrows, flipped in (
        (axes[0], "(a) 处理前：存在方向跳变", raw, flip),
        (axes[1], "(b) 处理后：方向场平滑连续", smooth, np.zeros_like(flip)),
    ):
        b = merged.iloc[idx_bg]
        ax.scatter(b["X"] * 1000, b["Y"] * 1000, c="#E3E7EC", s=1.5,
                   alpha=0.4, rasterized=True)
        a = merged.iloc[idx_ar]
        v = norm_xy(arrows[idx_ar])
        is_bad = flipped[idx_ar]
        if np.all(~flipped):
            # (b) 处理后：全部蓝色平滑方向箭头
            ax.quiver(
                a["X"] * 1000, a["Y"] * 1000, v[:, 0], v[:, 1],
                color="#2C7FB8", scale=60, width=0.0020, alpha=0.9,
                zorder=4,
            )
            ax.plot([], [], color="#2C7FB8", lw=2, label="连续方向")
            ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        else:
            # (a) 蓝色连续箭头 + 红色跳变箭头，图例说明
            ax.quiver(
                a["X"] * 1000, a["Y"] * 1000, v[:, 0], v[:, 1],
                color=np.where(is_bad, "#D62728", "#2C7FB8"),
                scale=60, width=0.0020, alpha=0.9, zorder=4,
            )
            ax.plot([], [], color="#D62728", lw=2,
                    label="方向跳变（与邻域反向）")
            ax.plot([], [], color="#2C7FB8", lw=2, label="连续方向")
            ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("X (mm)", fontsize=11)
        ax.set_ylabel("Y (mm)", fontsize=11)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax.set_aspect("equal")
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.15)

    fig.suptitle("最大主应力方向场：2-RoSy 符号连续性化前后对比", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig5_3_direction_continuity")


def fig_5_4_spacing_curve():
    """图5-4 应力自适应间距映射：(a) 映射曲线 (b) 扫描线间距示意。"""
    from matplotlib.collections import LineCollection

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4),
                             gridspec_kw={"width_ratios": [1.0, 1.1]})
    ax, ax2 = axes
    s = np.linspace(0, 1, 400)
    dmax, dmin, gamma = 0.045, 0.018, 0.75
    d = dmax - (dmax - dmin) * s ** gamma

    # (a) 渐变映射曲线
    pts = np.array([s, d]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="coolwarm", norm=plt.Normalize(0, 1),
                        linewidths=3.2)
    lc.set_array(s[:-1])
    ax.add_collection(lc)
    ax.autoscale_view()

    # 高低应力区背景与对比曲线
    ax.axvspan(0, 0.3, color="#2C7FB8", alpha=0.06)
    ax.axvspan(0.7, 1.0, color="#D62728", alpha=0.06)
    for g, col, ls in ((0.5, "#BBBBBB", "--"), (1.0, "#BBBBBB", ":")):
        ax.plot(s, dmax - (dmax - dmin) * s ** g, color=col, ls=ls, lw=1.2)
    ax.axhline(dmax, color="#777777", ls="--", lw=1.1)
    ax.axhline(dmin, color="#777777", ls="--", lw=1.1)

    ax.annotate("高应力 → 小间距（路径加密）", xy=(0.98, dmin),
                xytext=(0.58, 0.040),
                arrowprops=dict(arrowstyle="->", color="#D62728", lw=1.4),
                fontsize=10.5, color="#D62728",
                bbox=dict(fc="white", ec="none", alpha=0.9, pad=1))
    ax.annotate("低应力 → 大间距（路径稀疏）", xy=(0.025, dmax - 0.0008),
                xytext=(0.02, 0.030), ha="left",
                arrowprops=dict(arrowstyle="->", color="#2C7FB8", lw=1.4),
                fontsize=10.5, color="#2C7FB8",
                bbox=dict(fc="white", ec="none", alpha=0.9, pad=1))

    ax.set_xlabel(r"归一化应力 $\bar{\sigma}$", fontsize=13)
    ax.set_ylabel(r"扫描间距 $d(\bar{\sigma})$ / $L_c$", fontsize=13)
    ax.set_title(r"(a) 应力自适应间距映射（$\gamma=0.75$）", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)

    # (b) 不同应力水平下的扫描线间距示意
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(0, 1)
    rows = [("低应力 $\\bar\\sigma=0.1$", 0.1),
            ("中应力 $\\bar\\sigma=0.5$", 0.5),
            ("高应力 $\\bar\\sigma=0.9$", 0.9)]
    d_ref = dmax - (dmax - dmin) * 0.1 ** gamma
    for k, (lbl, s0) in enumerate(rows):
        dd = dmax - (dmax - dmin) * s0 ** gamma
        step = 0.32 * dd / d_ref
        y = 0.78 - k * 0.3
        x, cnt = 0.0, 0
        while x <= 1.0:
            ax2.plot([x, x], [y - 0.09, y + 0.09], color=cmap(norm(s0)),
                     lw=2.4, solid_capstyle="round")
            x += step
            cnt += 1
        ax2.text(1.04, y, f"{cnt} 条扫描线", va="center", fontsize=10)
        ax2.text(-0.03, y, lbl, ha="right", va="center", fontsize=10.5,
                 color=cmap(norm(s0)))
    ax2.axvspan(0.0, 0.34, color="#2C7FB8", alpha=0.05)
    ax2.axvspan(0.66, 1.0, color="#D62728", alpha=0.05)
    ax2.set_xlim(-0.42, 1.35)
    ax2.set_ylim(0.05, 1.05)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("(b) 不同应力水平的扫描线间距示意", fontsize=13)
    ax2.spines[["top", "right", "bottom", "left"]].set_visible(False)

    fig.suptitle("应力自适应扫描间距映射", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig5_4_spacing_curve")


def fig_5_5_zigzag():
    """图5-5 分层与层内锯齿扫描（基于模板映射的真实路径数据）。"""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    mesh, _entry, sim, _mapped, path_data = load_template_mapping()
    pts = mesh["points"]  # (N,3) mm
    n_layers = int(path_data["Layer"].max())
    k_show = 6  # 显示用分层数

    # ---- (a) 沿切片轴分层（按 Z 分位分层着色 + 切片平面） ----
    fig = plt.figure(figsize=(13.5, 6.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    z = pts[:, 2]
    edges = np.quantile(z, np.linspace(0, 1, k_show + 1))
    layer_id = np.clip(np.searchsorted(edges, z, side="right") - 1,
                       0, k_show - 1)
    cmap_l = plt.cm.viridis
    norm_l = plt.Normalize(1, k_show)
    idx = subsample_indices(len(pts), 30000, seed=5)
    sc = ax.scatter(pts[idx, 0], pts[idx, 1], pts[idx, 2],
                    c=layer_id[idx] + 1, cmap=cmap_l, norm=norm_l,
                    s=4, alpha=0.85, depthshade=True)
    # 切片平面
    x0, x1 = pts[:, 0].min(), pts[:, 0].max()
    y0, y1 = pts[:, 1].min(), pts[:, 1].max()
    for zz in edges[1:-1]:
        corners = np.array(
            [[x0, y0, zz], [x1, y0, zz], [x1, y1, zz], [x0, y1, zz]]
        )
        ax.add_collection3d(Poly3DCollection(
            [corners], facecolors="#B0BEC5", alpha=0.22, edgecolors="none"
        ))
    fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.08, label="层号")
    ax.set_title(f"(a) 沿切片轴分层（Z，共 {n_layers} 层）", fontsize=13)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    set_axes_equal_3d(ax)

    # ---- (b) 中间一层内的锯齿扫描路径 ----
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    # 自动选择密度等级颜色种类最多（变化最丰富）的层，并列时取路径点多的
    def _layer_score(l):
        s = path_data[path_data["Layer"] == l]["Density_Level"]
        return (int(s.nunique()), int(len(s)))

    L = int(max(path_data["Layer"].unique(), key=_layer_score))
    sub = path_data[path_data["Layer"] == L]
    zmin, zmax = float(sub["Z"].min()), float(sub["Z"].max())
    m = (pts[:, 2] >= zmin - 1e-9) & (pts[:, 2] <= zmax + 1e-9)
    lay_pts = pts[m]
    if len(lay_pts) > 20000:
        lay_pts = lay_pts[subsample_indices(len(lay_pts), 20000, seed=6)]
    ax2.scatter(lay_pts[:, 0], lay_pts[:, 1], lay_pts[:, 2],
                c="#C6CDD6", s=3, alpha=0.55)
    xyz = sub[["X", "Y", "Z"]].to_numpy(float)
    segt = sub["Segment_Type"].to_numpy()
    levels = sub["Density_Level"].to_numpy(int)
    colors = np.array(config.DENSITY_COLORS)[levels]
    skip = segt[1:] == "空区断开"
    ax2.add_collection3d(path_line_collection(
        xyz, colors, skip=skip, lw=1.4, alpha=0.95,
    ))
    handles = [
        plt.Line2D([0], [0], color=c, lw=2, label=lbl)
        for c, lbl in zip(config.DENSITY_COLORS, config.DENSITY_LABELS)
    ]
    ax2.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, -0.08), ncol=5,
               fontsize=8, framealpha=0.85, columnspacing=1.2)
    ax2.set_title(f"(b) 第 {L} 层内锯齿扫描（模板映射路径，S={sim:.2f}）",
                  fontsize=13)
    ax2.set_xlabel("X (mm)"); ax2.set_ylabel("Y (mm)"); ax2.set_zlabel("Z (mm)")
    set_axes_equal_3d(ax2)

    fig.suptitle("应力自适应层式路径：分层与层内锯齿扫描", fontsize=15)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    save(fig, "fig5_5_zigzag")


def fig_5_6_void_detection():
    """图5-6 空区检测与子路径切分：(a) 3D 整体 (b) 单层 2D 俯视。"""
    mesh, _entry, sim, _mapped, path_data = load_template_mapping()
    xyz = path_data[["X", "Y", "Z"]].to_numpy(float)
    seg = path_data["Segment_Type"].to_numpy()
    skip = seg[1:] == "空区断开"
    n_void = int(skip.sum())
    void_idx = np.where(skip)[0]

    pts = mesh["points"]
    x0, x1 = pts[:, 0].min(), pts[:, 0].max()
    y0, y1 = pts[:, 1].min(), pts[:, 1].max()
    idx = subsample_indices(len(pts), 1500, seed=7)
    void_segs = [np.vstack([xyz[i], xyz[i + 1]]) for i in void_idx]

    fig = plt.figure(figsize=(13.5, 6.2))

    # (a) 3D 整体视图
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    p = pts[idx]
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c="#C6CDD6", s=2, alpha=0.16)
    # 空区断开段：淡红色底纹
    ax.add_collection3d(Line3DCollection(
        void_segs, colors="#F6B7B0", linewidths=2.4, alpha=0.8,
    ))
    # 连续路径：按子路径画连续细折线，避免逐段绘制造成糊在一起
    ids_all = np.arange(len(xyz))
    runs, cur = [], [0]
    for a, b in zip(ids_all[:-1], ids_all[1:]):
        if skip[a]:
            runs.append(cur)
            cur = [b]
        else:
            cur.append(b)
    runs.append(cur)
    for run in runs:
        poly = xyz[run]
        if len(poly) > 1:
            ax.plot(poly[:, 0], poly[:, 1], poly[:, 2], color="#2C7FB8",
                    lw=0.7, alpha=0.9, solid_capstyle="round")
    ax.view_init(elev=25, azim=-60)
    ax.set_title("(a) 整体路径（蓝=连续路径，淡红=空区断开段）", fontsize=13)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.tick_params(labelsize=8)
    set_axes_equal_3d(ax)

    # (b) 单层 2D 俯视：空区断开 → 子路径切分
    ax2 = fig.add_subplot(1, 2, 2)
    layers = path_data["Layer"].to_numpy(int)
    vl = layers[1:][skip]
    L = (int(pd.Series(vl).value_counts().idxmax())
         if len(vl) else int(layers.max()))
    lay_mask = layers == L
    zmin = float(xyz[lay_mask, 2].min())
    zmax = float(xyz[lay_mask, 2].max())
    pm = (pts[:, 2] >= zmin - 1e-9) & (pts[:, 2] <= zmax + 1e-9)
    ax2.scatter(pts[pm, 0], pts[pm, 1], c="#C6CDD6", s=3, alpha=0.35,
                zorder=1)

    # 该层路径画成一整条连续蓝线，保持纤维连续性；
    # 空区断开段先在蓝线下垫一条淡红色粗底纹，表示该段被删除
    ids = np.where(lay_mask)[0]
    poly = xyz[ids][:, :2]
    if len(poly) > 1:
        ax2.plot(poly[:, 0], poly[:, 1], color="#2C7FB8", lw=0.9,
                 alpha=0.9, zorder=3, solid_capstyle="round")
    for i in np.where(skip)[0]:
        if layers[i] == L and layers[i + 1] == L:
            ax2.plot([xyz[i, 0], xyz[i + 1, 0]],
                     [xyz[i, 1], xyz[i + 1, 1]],
                     color="#F6B7B0", lw=3.2, alpha=0.85, zorder=2,
                     solid_capstyle="round")

    handles = [
        plt.Line2D([0], [0], color="#2C7FB8", lw=1.5,
                   label="连续纤维路径"),
        plt.Line2D([0], [0], color="#F6B7B0", lw=3,
                   label="空区断开（删除段）"),
    ]
    ax2.legend(handles=handles, loc="upper left",
               bbox_to_anchor=(1.02, 1.0), fontsize=9, framealpha=0.9)
    ax2.set_xlim(x0 - 1, x1 + 1)
    ax2.set_ylim(y0 - 1, y1 + 1)
    ax2.set_aspect("equal")
    ax2.set_title(f"(b) 第 {L} 层俯视：空区断开 → 子路径切分", fontsize=13)
    ax2.set_xlabel("X (mm)"); ax2.set_ylabel("Y (mm)")
    ax2.tick_params(labelsize=9)
    ax2.grid(alpha=0.12)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"空区检测与子路径切分（相似度 S={sim:.2f}）", fontsize=15)
    fig.tight_layout(rect=(0, 0, 0.86, 0.94))
    save(fig, "fig5_6_void_detection")


def fig_5_7_path_density():
    """图5-7 规划路径密度：(a) 3D 整体 (b) 单层俯视，叠加应力热图。"""
    merged = load_merged()
    path_data, _ = generate_layer_path(
        merged,
        "Maximum_Principal",
        percentile=config.DEFAULT_PERCENTILE,
        n_layers=int(config.DEFAULT_LAYERS),
    )
    xyz = path_data[["X", "Y", "Z"]].to_numpy(float)
    seg = path_data["Segment_Type"].to_numpy()
    levels = path_data["Density_Level"].to_numpy(int)
    colors = np.array(config.DENSITY_COLORS)[levels]
    stress = merged["Von_Mises"].to_numpy(float)
    pts = merged[["X", "Y", "Z"]].to_numpy(float)
    norm_stress = plt.Normalize(float(stress.min()), float(stress.max()))

    n = len(xyz)
    step = max(1, n // 60000)
    sel = np.arange(0, n, step)
    xyz_r = xyz[sel]
    colors_r = colors[sel]
    skip_r = np.zeros(len(sel) - 1, dtype=bool)
    for i in range(len(sel) - 1):
        skip_r[i] = (seg[sel[i] + 1: sel[i + 1] + 1] == "空区断开").any()

    fig = plt.figure(figsize=(13.5, 6.2))
    # (a) 3D 整体：应力热力背景 + 密度着色路径
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    idx = subsample_indices(len(pts), 50000, seed=13)
    ax.scatter(pts[idx, 0], pts[idx, 1], pts[idx, 2], c=stress[idx],
               cmap="turbo", norm=norm_stress, s=2, alpha=0.22)
    ax.add_collection3d(path_line_collection(
        xyz_r, colors_r, skip=skip_r, lw=1.1, alpha=0.95,
    ))
    handles = [
        plt.Line2D([0], [0], color=c, lw=2, label=lbl)
        for c, lbl in zip(config.DENSITY_COLORS, config.DENSITY_LABELS)
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=9, framealpha=0.8)
    ax.set_title("(a) 整体路径：应力背景 + 密度着色", fontsize=13)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.view_init(elev=25, azim=-60)
    set_axes_equal_3d(ax)

    # (b) 单层俯视：节点应力热图 + 密度路径
    ax2 = fig.add_subplot(1, 2, 2)
    layers = path_data["Layer"].to_numpy(int)

    def _score(l):
        s = levels[layers == l]
        return (int(np.unique(s).size), int(len(s)))

    L = int(max(np.unique(layers), key=_score))
    lay = layers == L
    zmin = float(xyz[lay, 2].min())
    zmax = float(xyz[lay, 2].max())
    pm = (pts[:, 2] >= zmin - 1e-9) & (pts[:, 2] <= zmax + 1e-9)
    ax2.scatter(pts[pm, 0], pts[pm, 1], c=stress[pm], cmap="turbo",
                norm=norm_stress, s=7, alpha=0.8, zorder=1)
    ids = np.where(lay)[0]
    for a, b in zip(ids[:-1], ids[1:]):
        ax2.plot([xyz[a, 0], xyz[b, 0]], [xyz[a, 1], xyz[b, 1]],
                 color=colors[a], lw=1.4, alpha=0.95, zorder=3,
                 solid_capstyle="round")
    fig.colorbar(plt.cm.ScalarMappable(cmap="turbo", norm=norm_stress),
                 ax=ax2, shrink=0.78, pad=0.03, label="von Mises (Pa)")
    ax2.set_xlim(pts[:, 0].min() - 0.002, pts[:, 0].max() + 0.002)
    ax2.set_ylim(pts[:, 1].min() - 0.002, pts[:, 1].max() + 0.002)
    ax2.set_aspect("equal")
    ax2.set_title(f"(b) 第 {L} 层俯视：应力热图 + 密度路径", fontsize=13)
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)")
    ax2.tick_params(labelsize=9)
    ax2.grid(alpha=0.15)

    fig.suptitle("应力自适应层式路径（密度等级着色）", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig5_7_path_density")


def fig_5_8_signature():
    """图5-8 点云规范化与形状签名：主图 + 周边直方图（渐变配色）。"""
    mesh = load_part1_mesh()
    pts = mesh["points"]
    normed, (_c, axes, _s) = canonical_points(pts)
    sig = compute_signature(pts)
    blocks = sig["blocks"]  # 3×16 箱主轴块 + 1×24 箱径向块
    labels = ["主轴 X", "主轴 Y", "主轴 Z", "径向距离"]
    from matplotlib.colors import LinearSegmentedColormap

    def _grad(name):
        base = plt.get_cmap(name)
        return LinearSegmentedColormap.from_list(
            f"{name}Mid", base(np.linspace(0.35, 1.0, 256))
        )

    cmap_a = _grad("Purples")    # (a) 规范化后点云（浅紫）
    cmap_b = _grad("PuBu")       # (b) 原始点云（蓝紫）
    # 右侧直方图统一用 Blues 色带，但每张取不同深浅段（浅→深递进）
    _bl = plt.get_cmap("Blues")
    cmap_h = [
        LinearSegmentedColormap.from_list(
            f"BluesBand{i}",
            _bl(np.linspace(0.25 + 0.15 * i, 0.55 + 0.15 * i, 256)),
        )
        for i in range(4)
    ]

    fig = plt.figure(figsize=(12.5, 8.8))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.35, 1.0],
                          hspace=0.58, wspace=0.24)

    # 左列上：规范化后点云 + 主轴
    ax_top = fig.add_subplot(gs[0:2, 0], projection="3d")
    idx = subsample_indices(len(normed), 30000, seed=3)
    r = np.linalg.norm(normed[idx], axis=1)
    sc = ax_top.scatter(normed[idx, 0], normed[idx, 1], normed[idx, 2],
                        c=r, cmap=cmap_a, s=4, alpha=0.9,
                        vmin=float(np.quantile(r, 0.10)),
                        vmax=float(r.max()))
    o = np.zeros(3)
    for k in range(3):
        ax_top.quiver(*o, *axes[:, k], color="#D62728", lw=2.4,
                      arrow_length_ratio=0.08)
    ax_top.set_title("(a) 规范化后点云与主轴", fontsize=12)
    ax_top.set_xlabel("Xc"); ax_top.set_ylabel("Yc")
    ax_top.set_zlabel("Zc")
    fig.colorbar(sc, ax=ax_top, shrink=0.55, pad=0.06,
                 label="归一化径向距离")
    set_axes_equal_3d(ax_top)

    # 左列下：原始点云（质心对齐）+ 主轴
    ax_bot = fig.add_subplot(gs[2:4, 0], projection="3d")
    cent = pts.mean(axis=0)
    p0 = pts - cent
    idx2 = subsample_indices(len(p0), 30000, seed=14)
    r2 = np.linalg.norm(p0[idx2], axis=1)
    sc2 = ax_bot.scatter(p0[idx2, 0], p0[idx2, 1], p0[idx2, 2],
                         c=r2, cmap=cmap_b, s=2.5, alpha=0.9,
                         vmin=float(np.quantile(r2, 0.05)),
                         vmax=float(r2.max()))
    for k in range(3):
        ax_bot.quiver(*o, *axes[:, k], color="#D62728", lw=2.4,
                      arrow_length_ratio=0.08)
    ax_bot.set_title("(b) 原始点云（质心对齐）与主轴", fontsize=12)
    ax_bot.set_xlabel("X (mm)"); ax_bot.set_ylabel("Y (mm)")
    ax_bot.set_zlabel("Z (mm)")
    fig.colorbar(sc2, ax=ax_bot, shrink=0.55, pad=0.06,
                 label="距质心距离 (mm)")
    set_axes_equal_3d(ax_bot)

    specs = [
        (gs[0, 1], "主轴 X"),
        (gs[1, 1], "主轴 Y"),
        (gs[2, 1], "主轴 Z"),
        (gs[3, 1], "径向距离"),
    ]
    right_axes = []
    for i, (cell, lbl) in enumerate(specs):
        axs = fig.add_subplot(cell)
        right_axes.append(axs)
        block = np.asarray(blocks[i], dtype=float)
        colors = cmap_h[i](np.linspace(0, 1, len(block)))
        axs.bar(np.arange(len(block)), block, color=colors, width=0.9)
        axs.set_title(f"({chr(99 + i)}) {lbl}直方图", fontsize=11)
        axs.set_xlabel("箱"); axs.set_ylabel("概率密度")
        axs.grid(alpha=0.2)
        axs.spines[["top", "right"]].set_visible(False)
    # 左右图之间的淡连接箭头：上左→前两个直方图，下左→后两个
    ann = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
    ann.set_xlim(0, 1)
    ann.set_ylim(0, 1)
    for axr, src in zip(right_axes[:2], (ax_top, ax_top)):
        ann.annotate(
            "", xy=(0.0, 0.5), xytext=(1.0, 0.5),
            xycoords=axr.transAxes, textcoords=src.transAxes,
            arrowprops=dict(arrowstyle="->", color="#B9C2CC", lw=0.9,
                            alpha=0.5, shrinkA=2, shrinkB=2),
        )
    for axr in right_axes[2:]:
        ann.annotate(
            "", xy=(0.0, 0.5), xytext=(1.0, 0.5),
            xycoords=axr.transAxes, textcoords=ax_bot.transAxes,
            arrowprops=dict(arrowstyle="->", color="#B9C2CC", lw=0.9,
                            alpha=0.5, shrinkA=2, shrinkB=2),
        )
    save(fig, "fig5_8_signature")


def fig_5_9_template_mapping():
    """图5-9 模板映射复用结果（模板 vs 目标零件）。"""
    mesh, entry, sim, mapped_geom, path_data = load_template_mapping()
    geom_t = pd.read_csv(entry["path"] / "geometry.csv", encoding="utf-8-sig")
    path_t = pd.read_csv(entry["path"] / "path.csv", encoding="utf-8-sig")

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    g = geom_t.iloc[subsample_indices(len(geom_t), 30000, seed=4)]
    ax1.scatter(g["X"], g["Y"], g["Z"], c="#B0BEC5", s=3, alpha=0.6)
    p = path_t[["X", "Y", "Z"]].to_numpy(float)
    ax1.add_collection3d(path_line_collection(
        p, "#E45756", lw=1.1, alpha=0.95,
    ))
    ax1.set_title(f"(a) 模板「{entry['name']}」路径", fontsize=12)
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    m = mesh["points"]
    idx = subsample_indices(len(m), 30000, seed=5)
    ax2.scatter(m[idx, 0], m[idx, 1], m[idx, 2], c="#B0BEC5", s=3, alpha=0.6)
    xyz = path_data[["X", "Y", "Z"]].to_numpy(float)
    seg = path_data["Segment_Type"].to_numpy()
    skip = seg[1:] == "空区断开"
    ax2.add_collection3d(path_line_collection(
        xyz, "#4C78A8", skip=skip, lw=1.1, alpha=0.95,
    ))
    ax2.set_title(f"(b) 映射至目标零件（相似度 S={sim:.2f}）", fontsize=12)
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    for ax in (ax1, ax2):
        set_axes_equal_3d(ax)
    fig.tight_layout()
    save(fig, "fig5_9_template_mapping")


def run_all():
    """生成 5.2 节全部可自动绘制的图（界面截图除外）。"""
    fig_5_1_flowchart()
    fig_5_2_stress_field()
    fig_5_3_direction_continuity()
    fig_5_4_spacing_curve()
    fig_5_5_zigzag()
    fig_5_6_void_detection()
    fig_5_7_path_density()
    fig_5_8_signature()
    fig_5_9_template_mapping()
    print("5.2 配图完成（图5-10 为软件界面截图，需人工截取）")


if __name__ == "__main__":
    run_all()
