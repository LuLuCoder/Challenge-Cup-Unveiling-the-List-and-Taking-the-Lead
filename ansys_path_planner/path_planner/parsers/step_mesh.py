"""STEP 文件 -> 四面体网格 -> 节点点云（基于 gmsh + OpenCASCADE）。

背景：手写 STEP 解析器只能提取几何顶点/粗略面采样，无法给出真正可用于
路径规划与网格质量评判的四面体网格。本模块优先调用 gmsh（自带 OpenCASCADE
内核，可直接读 STEP/STP），生成四面体网格后：
    - 输出全部网格节点（即"网格提取的点云"）；
    - 输出四面体单元、表面线框、网格质量指标（minSICN 等）；
    - 自动评判网格质量，并给出"优秀/良好/需细化"结论。

若 gmsh 未安装或读取失败，自动回退到 step_cloud 的轻量解析（仅点云，
无网格与质量指标），保证导入流程不中断。
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from path_planner.parsers.step_cloud import parse_step_file

# 四面体质量（minSICN，0~1，1 为理想正四面体）的评判阈值
QUALITY_BAD = 0.30          # 低于该值的单元视为"差单元"
QUALITY_GOOD_MEAN = 0.60    # 平均质量 >= 该值且差单元极少 -> 优秀
QUALITY_OK_MEAN = 0.45      # 平均质量 >= 该值 -> 良好
MAX_TETS = 300000           # 四面体数量上限（超过则自动放大网格尺寸重试）
MAX_WIRE_SEGMENTS = 200000  # 线框/网格边绘制上限


def _tet_scaled_jacobian(tets, points):
    """向量化计算四面体单元质量（归一化缩放雅可比，0~1，正四面体=1）。

    对每个四面体，取 4 个角点中的最小 scaled Jacobian：
        q_i = |det(a, b, c)| / (||a|| ||b|| ||c||)
    其中 a,b,c 为从该角点出发的三条棱向量；再乘以 sqrt(2) 归一化，
    使正四面体 q=1，与 gmsh 的 minSICN 量纲一致。
    """
    t = tets.astype(np.int64)
    p = points.astype(np.float64)
    v = p[t]  # (M,4,3)
    corners = []
    # 每个角点 i：其他三个顶点作为棱向量
    for i in range(4):
        others = [j for j in range(4) if j != i]
        a = v[:, others[0]] - v[:, i]
        b = v[:, others[1]] - v[:, i]
        c = v[:, others[2]] - v[:, i]
        det = (
            a[:, 0] * (b[:, 1] * c[:, 2] - b[:, 2] * c[:, 1])
            - a[:, 1] * (b[:, 0] * c[:, 2] - b[:, 2] * c[:, 0])
            + a[:, 2] * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])
        )
        denom = (
            np.linalg.norm(a, axis=1)
            * np.linalg.norm(b, axis=1)
            * np.linalg.norm(c, axis=1)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.abs(det) / denom * np.sqrt(2.0)
        corners.append(q)
    return np.minimum.reduce(corners)


def _tet_volumes(tets, points):
    """向量化计算四面体体积（mm^3）。"""
    t = tets.astype(np.int64)
    v = points.astype(np.float64)[t]
    a = v[:, 1] - v[:, 0]
    b = v[:, 2] - v[:, 0]
    c = v[:, 3] - v[:, 0]
    det = (
        a[:, 0] * (b[:, 1] * c[:, 2] - b[:, 2] * c[:, 1])
        - a[:, 1] * (b[:, 0] * c[:, 2] - b[:, 2] * c[:, 0])
        + a[:, 2] * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])
    )
    return np.abs(det) / 6.0


def _mesh_quality_report(tets, points, quality=None):
    """汇总网格质量指标并给出自动评判。"""
    if tets is None or len(tets) == 0:
        return None
    if quality is None:
        quality = _tet_scaled_jacobian(tets, points)
    quality = np.asarray(quality, dtype=float)
    bad_frac = float((quality < QUALITY_BAD).mean())
    total_vol = float(_tet_volumes(tets, points).sum())
    stats = {
        "mean": float(quality.mean()),
        "min": float(quality.min()),
        "std": float(quality.std()),
        "bad_frac": bad_frac,
        "n_bad": int((quality < QUALITY_BAD).sum()),
        "volume_mm3": total_vol,
    }
    if stats["mean"] >= QUALITY_GOOD_MEAN and bad_frac <= 0.02:
        verdict = "优秀"
        note = "网格质量优秀，可直接用于路径规划与有限元分析。"
    elif stats["mean"] >= QUALITY_OK_MEAN and bad_frac <= 0.05:
        verdict = "良好"
        note = "网格质量良好，个别差单元不影响整体使用。"
    else:
        verdict = "需细化"
        note = "网格质量一般，建议减小网格尺寸重新划分或检查几何。"
    stats["verdict"] = verdict
    stats["note"] = note
    return stats


def _tri_edges_to_segments(tri_nodes, node_pos, max_segments=MAX_WIRE_SEGMENTS):
    """把表面三角形单元去重后转成线段 (K,2,3)，用于线框显示。"""
    if len(tri_nodes) == 0:
        return np.zeros((0, 2, 3))
    pairs = np.vstack(
        [tri_nodes[:, [0, 1]], tri_nodes[:, [1, 2]], tri_nodes[:, [2, 0]]]
    )
    pairs = np.sort(pairs, axis=1)
    pairs = np.unique(pairs, axis=0)
    if len(pairs) > max_segments:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pairs), max_segments, replace=False)
        pairs = pairs[idx]
    pos = node_pos.astype(np.float64)
    return pos[pairs.astype(np.int64)]


def _gmsh_mesh(step_path, max_size_mm, progress_cb=None):
    """调用 gmsh 生成四面体网格，返回原始数据字典（不 finalize 前可用）。"""
    import gmsh

    def progress(msg):
        if progress_cb:
            progress_cb(msg)

    progress("正在初始化 gmsh…")
    # 后台线程内调用 gmsh 时，默认的 interruptible=True 会执行
    # signal.signal(...)，而 Python 只允许主线程注册信号处理器，
    # 会抛 "signal only works in main thread"。关闭该选项即可在线程中网格化。
    gmsh.initialize(
        readConfigFiles=False, run=False, interruptible=False
    )
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)      # 正面 Delaunay 3D
    gmsh.option.setNumber("Mesh.ElementOrder", 1)   # 一阶四面体
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 18)

    progress("正在读取 STEP 几何…")
    gmsh.open(str(step_path))
    bb = gmsh.model.getBoundingBox(-1, -1)
    extent = np.array([bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]])
    extent = np.maximum(extent, 1e-6)

    if max_size_mm is None:
        max_size_mm = float(extent.max() / 25.0)
    max_size_mm = float(np.clip(max_size_mm, 0.15, 8.0))
    gmsh.option.setNumber("Mesh.MeshSizeMax", max_size_mm)
    gmsh.option.setNumber("Mesh.MeshSizeMin", max(0.05, max_size_mm / 5.0))

    # 自适应：若单元数过多则放大网格尺寸重试
    attempt = 0
    while True:
        progress(f"正在生成四面体网格（尺寸上限 {max_size_mm:.2f} mm）…")
        gmsh.model.mesh.generate(3)
        types, tags, conn_lists = gmsh.model.mesh.getElements(3, -1)
        n_tets = sum(len(t) for t in tags)
        if n_tets <= MAX_TETS or attempt >= 3:
            break
        progress(f"单元数 {n_tets} 过多，自动放大网格尺寸后重试…")
        gmsh.model.mesh.clear()
        max_size_mm = min(8.0, max_size_mm * 2.0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_size_mm)
        gmsh.option.setNumber("Mesh.MeshSizeMin", max(0.05, max_size_mm / 5.0))
        attempt += 1

    progress("正在提取节点与单元…")
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    points = np.asarray(coords, dtype=float).reshape(-1, 3)
    # node_tags 可能不是从 0 开始的连续编号：建立 tag -> 行号 的映射
    tag_to_row = {int(t): i for i, t in enumerate(node_tags)}
    tet_list = []
    for ttype, ttags, connectivity in zip(types, tags, conn_lists):
        _ = ttype
        npe = 4  # 一阶四面体
        conn = np.asarray(connectivity).reshape(-1, npe)
        for c in conn:
            tet_list.append([tag_to_row[int(x)] for x in c])
    tets = (
        np.asarray(tet_list, dtype=np.int64).reshape(-1, 4)
        if tet_list else np.zeros((0, 4), dtype=np.int64)
    )

    progress("正在计算网格质量…")
    all_tags = [t for lst in tags for t in lst]
    quality = None
    try:
        quality = np.asarray(
            gmsh.model.mesh.getElementQualities(all_tags, "minSICN"), dtype=float
        )
    except Exception:
        quality = None

    progress("正在提取表面线框…")
    wire = np.zeros((0, 2, 3))
    try:
        tri_types, tri_tags, _ = gmsh.model.mesh.getElements(2, -1)
        tri_node_tags = []
        for ttype in tri_types:
            ntags, _ncoords, _ = gmsh.model.mesh.getNodesByElementType(ttype, -1)
            tri_node_tags.append(np.asarray(ntags).reshape(-1, 3))
        if tri_node_tags:
            tri_nodes = np.vstack(tri_node_tags)
            wire = _tri_edges_to_segments(tri_nodes, points, MAX_WIRE_SEGMENTS)
    except Exception:
        wire = np.zeros((0, 2, 3))

    mesh_info = {
        "points": points,
        "tets": tets,
        "quality": quality,
        "wireframe": wire,
        "max_size_mm": max_size_mm,
        "bbox": np.array([bb[:3], bb[3:]]),
    }
    gmsh.finalize()
    return mesh_info


def mesh_step_file(step_path, max_size_mm=None, progress_cb=None):
    """STEP -> 四面体网格 -> 点云，返回结果字典。

    返回字段：
        ok            bool   是否成功生成网格（gmsh 可用）
        mesher        "gmsh" / "fallback"
        points        (N,3) 网格节点坐标（mm），即网格提取的点云
        node_df       DataFrame(Node, X, Y, Z)
        tets          (M,4) 四面体（0 基节点索引）或 None
        wireframe     (K,2,3) 表面线框线段或 None
        quality       (M,) minSICN 或 None
        quality_stats dict 质量统计 + verdict，或 None
        bbox          (2,3) min/max
        elapsed       秒
        message       摘要文本
    """
    step_path = Path(step_path)
    t0 = time.time()
    try:
        mesh = _gmsh_mesh(step_path, max_size_mm, progress_cb)
    except Exception as exc:  # noqa: BLE001 - 任何失败都回退
        try:
            df = parse_step_file(str(step_path))
            pts = df[["X", "Y", "Z"]].to_numpy(float)
            return {
                "ok": False,
                "mesher": "fallback",
                "error": str(exc),
                "points": pts,
                "node_df": df,
                "tets": None,
                "wireframe": None,
                "quality": None,
                "quality_stats": None,
                "bbox": np.vstack([pts.min(0), pts.max(0)]),
                "elapsed": time.time() - t0,
                "message": f"gmsh 网格失败，已退回几何点云（{len(pts)} 点）。"
                f"原因：{exc}",
            }
        except Exception as exc2:  # noqa: BLE001
            return {
                "ok": False,
                "mesher": "fallback",
                "error": f"{exc}; 回退解析也失败: {exc2}",
                "points": np.zeros((0, 3)),
                "node_df": pd.DataFrame(columns=["Node", "X", "Y", "Z"]),
                "tets": None,
                "wireframe": None,
                "quality": None,
                "quality_stats": None,
                "bbox": np.zeros((2, 3)),
                "elapsed": time.time() - t0,
                "message": "STEP 解析失败。",
            }

    points = mesh["points"]
    tets = mesh["tets"]
    quality = mesh["quality"]
    stats = _mesh_quality_report(tets, points, quality)
    df = pd.DataFrame(
        {"Node": np.arange(1, len(points) + 1), "X": points[:, 0],
         "Y": points[:, 1], "Z": points[:, 2]}
    )
    if stats:
        msg = (
            f"网格化完成：{len(points)} 节点，{len(tets)} 四面体，"
            f"平均质量 {stats['mean']:.3f}，判定：{stats['verdict']}。"
        )
    else:
        msg = f"几何读取完成：{len(points)} 节点，但未生成四面体单元。"
    return {
        "ok": True,
        "mesher": "gmsh",
        "error": None,
        "points": points,
        "node_df": df,
        "tets": tets,
        "wireframe": mesh["wireframe"],
        "quality": quality,
        "quality_stats": stats,
        "bbox": mesh["bbox"],
        "max_size_mm": mesh["max_size_mm"],
        "elapsed": time.time() - t0,
        "message": msg,
    }
