"""模板库：相似零件直接映射模板路径（无需重新 ANSYS 仿真）。

模板 = 一次完整的“真实仿真 + 路径规划”结果。
新零件导入后先做形状相似度评判：
    - 相似度 >= 阈值 -> 直接把模板路径与应力场映射到新零件；
    - 低于阈值      -> 提示需要真实仿真数据。
"""

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from path_planner.analysis.shape_signature import (
    canonical_points,
    compute_signature,
    signature_similarity,
)


# 常量：优先读 config（若 config 已扩展），否则使用内置默认值
try:
    from path_planner import config as _cfg

    TEMPLATE_DIR = Path(getattr(
        _cfg, "TEMPLATE_DIR",
        Path(__file__).resolve().parent.parent.parent / "template_library",
    ))
    DEFAULT_THRESHOLD = getattr(_cfg, "DEFAULT_SIMILARITY_THRESHOLD", 0.80)
    AXIS_BINS = getattr(_cfg, "SIGNATURE_AXIS_BINS", 16)
    RADIAL_BINS = getattr(_cfg, "SIGNATURE_RADIAL_BINS", 24)
    NEIGHBORS = getattr(_cfg, "MAPPING_NEIGHBORS", 3)
    PATH_SCALE = getattr(_cfg, "MAPPING_PATH_SCALE", 1.0)
    STRESS_SCALE = getattr(_cfg, "MAPPING_STRESS_SCALE", 1.0)
except Exception:  # pragma: no cover - 兜底，独立运行时也可用
    TEMPLATE_DIR = Path(__file__).resolve().parent.parent.parent / "template_library"
    DEFAULT_THRESHOLD = 0.80
    AXIS_BINS = 16
    RADIAL_BINS = 24
    NEIGHBORS = 3
    PATH_SCALE = 1.0
    STRESS_SCALE = 1.0


def ensure_library_dir():
    """确保模板库目录存在。"""
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    return TEMPLATE_DIR


def _safe_name(name):
    """文件夹名安全化（兼容中文，去除非法字符）。"""
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", str(name).strip())
    return cleaned or "template"


def save_template(geometry_df, path_df, name, description=""):
    """把一次真实仿真 + 规划路径保存为模板。

    geometry_df：含 Node,X,Y,Z 与应力字段（如 Maximum_Principal）的融合数据；
    path_df    ：generate_layer_path 输出的路径数据。
    返回模板文件夹 Path。
    """
    ensure_library_dir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = TEMPLATE_DIR / f"{_safe_name(name)}__{stamp}"
    folder.mkdir(parents=True)

    points = geometry_df[["X", "Y", "Z"]].to_numpy(float)
    normalized, (centroid, axes, scale) = canonical_points(points)
    signature = compute_signature(
        points, axis_bins=AXIS_BINS, radial_bins=RADIAL_BINS
    )

    geom = geometry_df.copy()
    geom["Xc"] = normalized[:, 0]
    geom["Yc"] = normalized[:, 1]
    geom["Zc"] = normalized[:, 2]
    geom.to_csv(folder / "geometry.csv", index=False, encoding="utf-8-sig")

    path_df.to_csv(folder / "path.csv", index=False, encoding="utf-8-sig")

    size_mm = np.ptp(geometry_df[["X", "Y", "Z"]].to_numpy(float), axis=0)
    n_layers = (
        int(path_df["Layer"].max())
        if "Layer" in path_df.columns and len(path_df)
        else 0
    )
    meta = {
        "name": str(name),
        "description": str(description),
        "created_at": stamp,
        "node_count": int(len(geometry_df)),
        "path_points": int(len(path_df)),
        "n_layers": n_layers,
        "size_mm": [round(float(v), 3) for v in size_mm],
        "signature": {
            "node_count": signature["node_count"],
            "blocks": signature["blocks"],
            "axis_bins": signature["axis_bins"],
            "radial_bins": signature["radial_bins"],
        },
        "transform": {
            "centroid": [float(v) for v in centroid],
            "axes": np.asarray(axes, dtype=float).tolist(),
            "scale": float(scale),
        },
    }
    (folder / "template.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return folder


def list_templates():
    """列出模板库中的所有模板。"""
    ensure_library_dir()
    entries = []
    for folder in sorted(TEMPLATE_DIR.iterdir()):
        meta_path = folder / "template.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        entries.append({
            "path": folder,
            "name": meta.get("name", folder.name),
            "created_at": meta.get("created_at", ""),
            "node_count": meta.get("node_count", 0),
            "path_points": meta.get("path_points", 0),
            "n_layers": meta.get("n_layers", 0),
            "size_mm": meta.get("size_mm", [0.0, 0.0, 0.0]),
        })
    return entries


def _load_meta(folder):
    with open(Path(folder) / "template.json", "r", encoding="utf-8") as f:
        return json.load(f)


def find_best_template(signature, threshold=None):
    """在模板库中检索最相似模板。

    返回 (entry, similarity)；全部低于阈值时返回 (None, 最高相似度)。
    """
    if threshold is None:
        threshold = DEFAULT_THRESHOLD
    best = None
    best_sim = 0.0
    for entry in list_templates():
        try:
            meta = _load_meta(entry["path"])
        except Exception:
            continue
        signature_meta = meta.get("signature") or {}
        if not signature_meta.get("invariant_blocks"):
            # 旧格式模板：用 geometry.csv 重建新签名（自动升级）
            try:
                geom = pd.read_csv(
                    entry["path"] / "geometry.csv", encoding="utf-8-sig"
                )
                if np.ptp(geom[["X", "Y", "Z"]].to_numpy(float),
                          axis=0).max() < 1e-9:
                    continue  # 退化模板（点云几乎为一点）
                signature_meta = compute_signature(
                    geom[["X", "Y", "Z"]].to_numpy(float),
                    axis_bins=AXIS_BINS,
                    radial_bins=RADIAL_BINS,
                )
                meta["signature"] = signature_meta
            except Exception:
                continue
            try:
                # 写回（文件只读等失败时忽略：本次会话仍使用内存中的新签名）
                (entry["path"] / "template.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
        try:
            sim = signature_similarity(signature, signature_meta)
        except Exception:
            continue  # 跳过签名格式不兼容（如旧版本模板）
        if sim > best_sim:
            best_sim = sim
            best = entry
    if best is not None and best_sim >= threshold:
        return best, best_sim
    return None, best_sim


def delete_template(folder):
    """删除模板（校验路径必须在模板库内，防止误删）。"""
    folder = Path(folder).resolve()
    root = TEMPLATE_DIR.resolve()
    if root not in folder.parents:
        raise ValueError("只能删除模板库内的模板。")
    if not folder.exists():
        raise FileNotFoundError(f"模板不存在：{folder}")
    shutil.rmtree(folder)


def map_from_template(folder, target_df, stress_scale=None, path_scale=None):
    """把模板的应力场与路径映射到新零件。

    返回：
        mapped_geometry  新零件节点 + 映射后的应力字段（Maximum_Principal 等）
        mapped_path      新零件坐标下的模板路径（吸附到最近节点）
    """
    folder = Path(folder)
    meta = _load_meta(folder)
    transform = meta["transform"]
    c_t = np.asarray(transform["centroid"], dtype=float)
    a_t = np.asarray(transform["axes"], dtype=float)
    s_t = float(transform["scale"])

    if stress_scale is None:
        stress_scale = STRESS_SCALE
    if path_scale is None:
        path_scale = PATH_SCALE

    target_pts = target_df[["X", "Y", "Z"]].to_numpy(float)
    t_norm, (c_q, a_q, s_q) = canonical_points(target_pts)

    # ---- 应力场映射：模板归一化坐标 -> 新零件节点 ----
    geom_t = pd.read_csv(folder / "geometry.csv", encoding="utf-8-sig")
    src_norm = geom_t[["Xc", "Yc", "Zc"]].to_numpy(float)
    src_vals = {
        col: geom_t[col].to_numpy(float)
        for col in ("Maximum_Principal", "Von_Mises")
        if col in geom_t.columns
    }
    if not src_vals:
        raise ValueError("模板 geometry.csv 中缺少应力字段。")

    tree = cKDTree(src_norm)
    dist, idx = tree.query(
        t_norm, k=min(NEIGHBORS, len(src_norm))
    )

    mapped_geom = target_df.copy()
    for col, values in src_vals.items():
        if idx.ndim == 1:
            mapped = values[idx]
        else:
            w = 1.0 / (np.maximum(dist, 1e-12) ** 2 + 1e-12)
            mapped = np.sum(values[idx] * w, axis=1) / np.sum(w, axis=1)
        mapped_geom[col] = mapped * stress_scale

    # ---- 路径映射：模板原坐标 -> 归一化 -> 新零件坐标 ----
    path_t = pd.read_csv(folder / "path.csv", encoding="utf-8-sig")
    path_pts = path_t[["X", "Y", "Z"]].to_numpy(float)
    path_canonical = (path_pts - c_t) @ a_t / s_t
    path_mapped = (path_canonical * s_q * path_scale) @ a_q.T + c_q

    mapped_path = path_t.copy()
    mapped_path["X"] = path_mapped[:, 0]
    mapped_path["Y"] = path_mapped[:, 1]
    mapped_path["Z"] = path_mapped[:, 2]

    # 吸附到新零件最近节点，并同步节点号与应力值
    node_tree = cKDTree(target_pts)
    _d2, idx2 = node_tree.query(path_mapped)
    nodes = target_df["Node"].to_numpy(int)
    mapped_path["Node"] = nodes[idx2]
    mapped_path["X"] = target_pts[idx2, 0]
    mapped_path["Y"] = target_pts[idx2, 1]
    mapped_path["Z"] = target_pts[idx2, 2]

    val_by_node = {
        int(n): float(v)
        for n, v in zip(mapped_geom["Node"].to_numpy(int),
                        mapped_geom["Maximum_Principal"].to_numpy(float))
    }
    mapped_path["Maximum_Principal"] = [
        val_by_node.get(int(n), 0.0)
        for n in mapped_path["Node"].to_numpy(int)
    ]

    return mapped_geom, mapped_path
