"""图5-2 应力场可视化多版本候选，供挑选。

输出到 output/fig5_2_variants/：
    v1 当前版          —— 真实 ANSYS 数据，(a) von Mises 云图 (b) 主方向场
    v2 深色大点版      —— 单面板，plasma 配色，深色背景
    v3 云图+方向融合   —— 单面板，应力着色 + 主方向箭头
    v4 俯视切片版      —— 2D 俯视，应力着色 + 方向箭头（更适合论文）
    v5 模板库版        —— 用模板库 test1 的 geometry.csv 应力场
    v6 网格表面版      —— 零件1 四面体网格表面按映射应力着色（FEM 风格）
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from common import output_dir, save, set_axes_equal_3d, subsample_indices

PROJECT = Path(__file__).resolve().parent.parent
APP = PROJECT / "ansys_path_planner"
DATA = PROJECT / "data"
sys.path.insert(0, str(APP))

from path_planner import config
from path_planner.analysis.shape_signature import compute_signature
from path_planner.analysis.stress import merge_ansys_files_data
from path_planner.analysis.template_library import find_best_template
from path_planner.parsers.step_mesh import mesh_step_file

OUT = output_dir() / "fig5_2_variants"
OUT.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("已生成:", path)


def _load_merged():
    stress = {
        comp: str(DATA / name)
        for name, comp in config.STRESS_FILE_MAP.items()
    }
    return merge_ansys_files_data(str(DATA / "node_coordinates_s.csv"), stress)


def _load_template():
    """模板库最相似模板的 geometry.csv（含应力场）。"""
    mesh = mesh_step_file(str(PROJECT / "ansys" / "零件1.STEP"))
    entry, sim = find_best_template(compute_signature(mesh["points"]), 0.0)
    geom = pd.read_csv(entry["path"] / "geometry.csv", encoding="utf-8-sig")
    return entry, sim, geom


def _surface_faces(tets):
    """由四面体提取边界三角形面（出现一次的三角面）。"""
    f = np.concatenate(
        [
            tets[:, [0, 1, 2]], tets[:, [0, 1, 3]],
            tets[:, [0, 2, 3]], tets[:, [1, 2, 3]],
        ]
    )
    fs = np.sort(f, axis=1)
    _u, inv, cnt = np.unique(fs, axis=0, return_inverse=True,
                             return_counts=True)
    return f[cnt[inv] == 1]


def v1_current():
    """(a) von Mises 云图 + (b) 主方向场。"""
    merged = _load_merged()
    fig = plt.figure(figsize=(13, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    idx = subsample_indices(len(merged), 60000, seed=1)
    d = merged.iloc[idx]
    sc = ax1.scatter(d["X"], d["Y"], d["Z"], c=d["Von_Mises"],
                     cmap="turbo", s=3, alpha=0.85)
    ax1.set_title("(a) von Mises 应力分布", fontsize=12)
    fig.colorbar(sc, ax=ax1, shrink=0.6, pad=0.08, label="von Mises (Pa)")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    idx2 = subsample_indices(len(merged), 1500, seed=2)
    q = merged.iloc[idx2]
    ax2.scatter(q["X"], q["Y"], q["Z"], c="#9EA7B3", s=2, alpha=0.5)
    ax2.quiver(q["X"], q["Y"], q["Z"], q["Principal_VX"], q["Principal_VY"],
               q["Principal_VZ"], length=0.0008, normalize=True,
               color="#D62728", alpha=0.75, lw=0.6)
    ax2.set_title("(b) 最大主应力方向场（局部采样）", fontsize=12)
    for ax in (ax1, ax2):
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.tick_params(labelsize=7)
        set_axes_equal_3d(ax)
    fig.tight_layout()
    _save(fig, "fig5_2_v1_current")


def v2_dark_plasma():
    """深色背景 + plasma，单面板大点。"""
    merged = _load_merged()
    plt.style.use("dark_background")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    idx = subsample_indices(len(merged), 80000, seed=6)
    d = merged.iloc[idx]
    sc = ax.scatter(d["X"], d["Y"], d["Z"], c=d["Von_Mises"], cmap="plasma",
                    s=4, alpha=0.95)
    ax.set_title("von Mises 应力分布", fontsize=14, color="white")
    fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.08, label="von Mises (Pa)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    set_axes_equal_3d(ax)
    fig.tight_layout()
    _save(fig, "fig5_2_v2_dark_plasma")
    plt.style.use("default")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False


def v3_combined():
    """单面板：应力着色 + 主方向箭头。"""
    merged = _load_merged()
    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    idx = subsample_indices(len(merged), 60000, seed=1)
    d = merged.iloc[idx]
    sc = ax.scatter(d["X"], d["Y"], d["Z"], c=d["Von_Mises"], cmap="turbo",
                    s=3, alpha=0.85)
    idx2 = subsample_indices(len(merged), 1200, seed=2)
    q = merged.iloc[idx2]
    ax.quiver(q["X"], q["Y"], q["Z"], q["Principal_VX"], q["Principal_VY"],
              q["Principal_VZ"], length=0.0007, normalize=True,
              color="white", alpha=0.9, lw=0.7)
    ax.set_title("应力分布与最大主应力方向", fontsize=13)
    fig.colorbar(sc, ax=ax, shrink=0.62, pad=0.08, label="von Mises (Pa)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    set_axes_equal_3d(ax)
    fig.tight_layout()
    _save(fig, "fig5_2_v3_combined")


def v4_topview():
    """俯视（XY 平面）应力 + 方向箭头，2D 版本。"""
    merged = _load_merged()
    fig, ax = plt.subplots(figsize=(9, 8))
    idx = subsample_indices(len(merged), 60000, seed=3)
    d = merged.iloc[idx]
    sc = ax.scatter(d["X"], d["Y"], c=d["Von_Mises"], cmap="turbo", s=4,
                    alpha=0.85)
    idx2 = subsample_indices(len(merged), 2500, seed=4)
    q = merged.iloc[idx2]
    ax.quiver(q["X"], q["Y"], q["Principal_VX"], q["Principal_VY"],
              color="#444444", alpha=0.85, width=0.0018, scale=60)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title("俯视：应力分布与主应力方向（XY 平面）", fontsize=13)
    fig.colorbar(sc, ax=ax, shrink=0.8, label="von Mises (Pa)")
    ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, "fig5_2_v4_topview")


def v5_template():
    """模板库 test1 的应力场。"""
    entry, sim, geom = _load_template()
    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    g = geom.iloc[subsample_indices(len(geom), 30000, seed=5)]
    sc = ax.scatter(g["X"], g["Y"], g["Z"], c=g["Maximum_Principal"],
                    cmap="turbo", s=10, alpha=0.9)
    ax.set_title(f"模板「{entry['name']}」应力场（相似度 S={sim:.2f}）",
                 fontsize=13)
    fig.colorbar(sc, ax=ax, shrink=0.62, pad=0.08,
                 label="Maximum Principal (Pa)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    set_axes_equal_3d(ax)
    fig.tight_layout()
    _save(fig, "fig5_2_v5_template")


def v6_mesh_surface():
    """零件1 四面体网格表面按映射应力着色（FEM 风格）。"""
    mesh = mesh_step_file(str(PROJECT / "ansys" / "零件1.STEP"))
    entry, sim = find_best_template(compute_signature(mesh["points"]), 0.0)
    from path_planner.analysis.template_library import map_from_template

    mapped_geom, _ = map_from_template(entry["path"], mesh["node_df"])
    stress = mapped_geom["Maximum_Principal"].to_numpy(float)
    faces = _surface_faces(mesh["tets"])
    face_stress = stress[faces].mean(axis=1)

    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    pc = Poly3DCollection(
        mesh["points"][faces], facecolors=plt.cm.turbo(
            (face_stress - face_stress.min())
            / max(face_stress.max() - face_stress.min(), 1e-12)
        ),
        edgecolors="k", linewidths=0.08, alpha=0.98,
    )
    ax.add_collection3d(pc)
    sc = ax.scatter([], [], [])
    fig.colorbar(
        plt.cm.ScalarMappable(
            cmap="turbo",
            norm=plt.Normalize(face_stress.min(), face_stress.max()),
        ),
        ax=ax, shrink=0.62, pad=0.08, label="Mapped Maximum Principal (Pa)",
    )
    ax.set_title(f"零件1 网格表面映射应力场（模板 S={sim:.2f}）", fontsize=13)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    lim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    ax.set_xlim3d(*lim[0]); ax.set_ylim3d(*lim[1]); ax.set_zlim3d(*lim[2])
    fig.tight_layout()
    _save(fig, "fig5_2_v6_mesh_surface")


def run_all():
    v1_current()
    v2_dark_plasma()
    v3_combined()
    v4_topview()
    v5_template()
    v6_mesh_surface()
    print("图5-2 六个候选版本已生成：", OUT)


if __name__ == "__main__":
    run_all()
