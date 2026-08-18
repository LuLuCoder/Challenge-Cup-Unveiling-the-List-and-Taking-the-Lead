"""STEP 导入检查窗口：模板 / 线框 / 四面体网格 / 点云 四视图 + 网格质量报告。"""

import tkinter as tk
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Line3DCollection

MAX_TET_EDGES = 120000   # 四面体网格边绘制上限
MAX_CLOUD_POINTS = 30000


def _load_template_points(entry):
    """读取模板 geometry.csv 的原始坐标点（未归一化）。"""
    if not entry:
        return None
    try:
        csv = Path(entry["path"]) / "geometry.csv"
        if not csv.is_file():
            return None
        df = pd.read_csv(csv)
        cols = [c for c in ("X", "Y", "Z") if c in df.columns]
        if len(cols) != 3:
            return None
        return df[cols].to_numpy(float)
    except Exception:
        return None


def _tet_edges(tets, points, max_edges=MAX_TET_EDGES):
    """四面体单元去重边，供网格视图绘制。"""
    if tets is None or len(tets) == 0:
        return np.zeros((0, 2, 3))
    pairs = np.vstack(
        [
            tets[:, [0, 1]], tets[:, [0, 2]], tets[:, [0, 3]],
            tets[:, [1, 2]], tets[:, [1, 3]], tets[:, [2, 3]],
        ]
    )
    pairs = np.sort(pairs, axis=1)
    pairs = np.unique(pairs, axis=0)
    if len(pairs) > max_edges:
        rng = np.random.default_rng(0)
        pairs = pairs[rng.choice(len(pairs), max_edges, replace=False)]
    return points[pairs.astype(np.int64)]


def _equal_aspect(ax, pts_list):
    """根据多个点集合统一 3D 坐标轴比例。"""
    lo = np.array([np.inf] * 3)
    hi = np.array([-np.inf] * 3)
    for pts in pts_list:
        if pts is None or len(pts) == 0:
            continue
        lo = np.minimum(lo, np.asarray(pts).reshape(-1, 3).min(0))
        hi = np.maximum(hi, np.asarray(pts).reshape(-1, 3).max(0))
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():
        lo, hi = np.zeros(3), np.ones(3)
    span = hi - lo
    span[span < 1e-9] = 1e-9
    ax.set_box_aspect(tuple(span))
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])


def _style_ax(ax, title):
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("X", fontsize=9)
    ax.set_ylabel("Y", fontsize=9)
    ax.set_zlabel("Z", fontsize=9)
    ax.tick_params(labelsize=7)


def show_step_inspection(parent, mesh_result, template_entry=None):
    """弹出 STEP 检查窗口（非阻塞）。

    mesh_result 来自 step_mesh.mesh_step_file；
    template_entry 为模板库条目（用于显示"模板样子"），可为 None。
    """
    win = tk.Toplevel(parent)
    win.title("STEP 导入检查：网格与点云")
    win.geometry("1280x900")

    points = mesh_result["points"]
    tets = mesh_result["tets"]
    wire = mesh_result["wireframe"]
    stats = mesh_result.get("quality_stats")
    tmpl_pts = _load_template_points(template_entry)

    fig = plt.Figure(figsize=(12.5, 8.2), dpi=100)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.97, bottom=0.03,
                        wspace=0.02, hspace=0.18)
    axs = [fig.add_subplot(2, 2, i + 1, projection="3d") for i in range(4)]

    # 1) 模板形状
    ax = axs[0]
    if tmpl_pts is not None and len(tmpl_pts):
        tmpl = tmpl_pts[
            np.random.default_rng(0).choice(
                len(tmpl_pts), min(len(tmpl_pts), MAX_CLOUD_POINTS),
                replace=False,
            )
        ]
        ax.scatter(tmpl[:, 0], tmpl[:, 1], tmpl[:, 2],
                   c="#4C78A8", s=2, alpha=0.7, depthshade=True)
        _style_ax(ax, "模板形状（模板库）")
    else:
        ax.text(0.5, 0.5, 0.5, "无匹配模板",
                ha="center", transform=ax.transAxes, fontsize=12)
        _style_ax(ax, "模板形状（模板库）")

    # 2) STEP 线框
    ax = axs[1]
    if wire is not None and len(wire):
        ax.add_collection3d(
            Line3DCollection(wire, colors="#555555", linewidths=0.5, alpha=0.8)
        )
        _equal_aspect(ax, [wire])
        _style_ax(ax, "STEP 几何线框")
    else:
        ax.text(0.5, 0.5, 0.5, "无线框数据", ha="center",
                transform=ax.transAxes, fontsize=12)
        _style_ax(ax, "STEP 几何线框")

    # 3) 四面体网格
    ax = axs[2]
    edges = np.zeros((0, 2, 3))
    if tets is not None and len(tets):
        edges = _tet_edges(tets, points)
        ax.add_collection3d(
            Line3DCollection(edges, colors="#B07AA1", linewidths=0.35,
                             alpha=0.55)
        )
        _equal_aspect(ax, [edges])
        _style_ax(ax, f"四面体网格（{len(tets)} 单元）")
    else:
        ax.text(0.5, 0.5, 0.5, "无网格数据", ha="center",
                transform=ax.transAxes, fontsize=12)
        _style_ax(ax, "四面体网格")

    # 4) 网格点云
    ax = axs[3]
    if len(points):
        n = min(len(points), MAX_CLOUD_POINTS)
        idx = np.random.default_rng(0).choice(len(points), n, replace=False)
        cloud = points[idx]
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2],
                   c="#E45756", s=2, alpha=0.65, depthshade=True)
        _equal_aspect(ax, [points])
        _style_ax(ax, f"网格节点点云（{len(points)} 点）")
    else:
        ax.text(0.5, 0.5, 0.5, "无点云", ha="center",
                transform=ax.transAxes, fontsize=12)
        _style_ax(ax, "网格节点点云")

    # 统一各子图比例（以点云为基准）
    _equal_aspect(axs[0], [points, tmpl_pts])
    _equal_aspect(axs[1], [points, wire])
    _equal_aspect(axs[2], [points, edges])
    _equal_aspect(axs[3], [points])

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
    canvas.draw()

    # 质量报告区
    panel = tk.Frame(win)
    panel.pack(side="bottom", fill="x", padx=10, pady=(0, 8))
    txt = tk.Text(panel, height=7, font=("Microsoft YaHei", 9),
                  relief="groove", bd=1)
    txt.pack(side="left", fill="both", expand=True, padx=(0, 8))

    lines = [mesh_result.get("message", "")]
    if stats:
        q_arr = mesh_result.get("quality")
        n_quality = int(len(q_arr)) if q_arr is not None else 0
        lines += [
            "",
            f"平均质量（minSICN）：{stats['mean']:.4f}　"
            f"最差：{stats['min']:.4f}　标准差：{stats['std']:.4f}",
            f"差单元（质量<{0.30:g}）：{stats['n_bad']} / {n_quality}"
            f"（{stats['bad_frac'] * 100:.3f}%）",
            f"实体体积：{stats['volume_mm3']:.1f} mm³　"
            f"网格判定：{stats['verdict']}",
            stats["note"],
        ]
    else:
        lines.append("未生成网格（gmsh 不可用或读取失败），仅显示几何点云。")
    if mesh_result.get("error"):
        lines.append(f"网格错误信息：{mesh_result['error']}")
    txt.insert("1.0", "\n".join(lines))
    txt.configure(state="disabled")

    btns = tk.Frame(panel)
    btns.pack(side="right", fill="y")
    tk.Button(
        btns, text="导出点云 CSV", width=14,
        command=lambda: _export_cloud(win, mesh_result),
    ).pack(pady=2)
    tk.Button(btns, text="关闭", width=14,
              command=lambda: (_close_fig(fig), win.destroy()),
              ).pack(pady=2)


def _close_fig(fig):
    try:
        plt.close(fig)
    except Exception:
        pass


def _export_cloud(win, mesh_result):
    from tkinter import filedialog, messagebox

    path = filedialog.asksaveasfilename(
        parent=win, title="导出网格节点点云",
        defaultextension=".csv",
        filetypes=[("CSV", "*.csv")],
    )
    if not path:
        return
    try:
        mesh_result["node_df"].to_csv(path, index=False, encoding="utf-8-sig")
        messagebox.showinfo("导出成功", f"已保存 {len(mesh_result['node_df'])} 个节点到：\n{path}")
    except Exception as exc:
        messagebox.showerror("导出失败", str(exc))
