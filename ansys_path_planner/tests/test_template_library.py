"""模板库：保存、检索与路径/应力映射测试。"""

import numpy as np
import pandas as pd

from path_planner.analysis import template_library as lib
from path_planner.analysis.shape_signature import compute_signature


def _cube(n, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, 3)) * scale


def _geom_df(points):
    n = len(points)
    return pd.DataFrame({
        "Node": np.arange(1, n + 1),
        "X": points[:, 0],
        "Y": points[:, 1],
        "Z": points[:, 2],
        "Maximum_Principal": np.linspace(10.0, 100.0, n),
        "Von_Mises": np.linspace(20.0, 200.0, n),
    })


def _path_df(points):
    rows = []
    step = 1
    sample = points[:: max(1, len(points) // 30)][:30]
    for layer in range(1, 4):
        for i, p in enumerate(sample):
            rows.append({
                "Step": step,
                "Layer": layer,
                "Node": i + 1,
                "X": p[0], "Y": p[1], "Z": p[2],
                "Path_Spacing": 0.1,
                "Segment_Type": "层内扫描",
                "Maximum_Principal": 50.0,
            })
            step += 1
    return pd.DataFrame(rows)


def test_save_find_map(tmp_path, monkeypatch):
    monkeypatch.setattr(lib, "TEMPLATE_DIR", tmp_path)

    src_pts = _cube(2000, scale=2.0, seed=0)
    folder = lib.save_template(
        _geom_df(src_pts), _path_df(src_pts), "模板A"
    )
    assert (folder / "template.json").exists()
    assert (folder / "geometry.csv").exists()
    assert (folder / "path.csv").exists()

    # 相似但尺寸/密度不同的新零件
    tgt_pts = _cube(1200, scale=1.5, seed=2)
    signature = compute_signature(tgt_pts)
    entry, sim = lib.find_best_template(signature, threshold=0.7)
    assert entry is not None
    assert sim > 0.9

    tgt = _geom_df(tgt_pts).drop(
        columns=["Maximum_Principal", "Von_Mises"]
    )
    mapped_geom, mapped_path = lib.map_from_template(entry["path"], tgt)

    assert "Maximum_Principal" in mapped_geom.columns
    assert len(mapped_path) == len(_path_df(src_pts))
    # 映射路径应吸附在新零件包围盒内
    lo = tgt_pts.min(axis=0) - 1e-9
    hi = tgt_pts.max(axis=0) + 1e-9
    mapped_xyz = mapped_path[["X", "Y", "Z"]].to_numpy(float)
    assert np.all(mapped_xyz >= lo)
    assert np.all(mapped_xyz <= hi)


def test_no_match_below_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(lib, "TEMPLATE_DIR", tmp_path)

    src_pts = _cube(2000, seed=0)
    lib.save_template(_geom_df(src_pts), _path_df(src_pts), "模板B")

    # 球体与立方体相似度低，应返回 None
    rng = np.random.default_rng(7)
    sphere = rng.normal(size=(2000, 3))
    sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)
    sphere *= (rng.random(2000) ** (1.0 / 3.0))[:, None]
    signature = compute_signature(sphere)
    entry, sim = lib.find_best_template(signature, threshold=0.8)
    assert entry is None
    assert sim < 0.8
