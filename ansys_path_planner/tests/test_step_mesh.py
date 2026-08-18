"""STEP 网格化（gmsh）与网格质量评判测试。"""

import numpy as np
import pytest

from path_planner.parsers.step_mesh import (
    _mesh_quality_report,
    _tet_scaled_jacobian,
    mesh_step_file,
)


def _regular_tet_mesh(n_per_side=4):
    """构造 n_per_side^3 个小正方体，每个剖成 5 个正四面体。"""
    pts = []
    tets = []
    h = 1.0 / n_per_side
    for ix in range(n_per_side):
        for iy in range(n_per_side):
            for iz in range(n_per_side):
                base = len(pts)
                x0, y0, z0 = ix * h, iy * h, iz * h
                corners = [
                    [x0, y0, z0],
                    [x0 + h, y0, z0],
                    [x0, y0 + h, z0],
                    [x0, y0, z0 + h],
                    [x0 + h, y0 + h, z0 + h],
                ]
                pts.extend(corners)
                # 标准立方体 5 四面体剖分
                local = [[0, 1, 2, 3], [0, 1, 4, 2], [0, 1, 3, 4],
                         [0, 4, 3, 2], [1, 4, 2, 3]]
                tets.extend([[base + i for i in t] for t in local])
    return np.asarray(pts, dtype=float), np.asarray(tets, dtype=np.int64)


def _perfect_tet():
    """标准正四面体（边长 2√2），minSICN 应为 1.0。"""
    pts = np.array(
        [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
        dtype=float,
    )
    tets = np.array([[0, 1, 2, 3]])
    return pts, tets


def test_scaled_jacobian_regular_tets():
    pts, tets = _perfect_tet()
    q = _tet_scaled_jacobian(tets, pts)
    assert q[0] == pytest.approx(1.0, abs=1e-9)


def test_quality_report_verdict():
    pts, tets = _perfect_tet()
    stats = _mesh_quality_report(tets, pts)
    assert stats["verdict"] == "优秀"
    assert stats["mean"] == pytest.approx(1.0, abs=1e-9)

    # 立方体 5 四面体剖分：总体积应恰为 1
    cube_pts, cube_tets = _regular_tet_mesh()
    cube_stats = _mesh_quality_report(cube_tets, cube_pts)
    assert cube_stats["volume_mm3"] == pytest.approx(1.0, rel=1e-6)
    assert 0.0 < cube_stats["min"] < 1.0

    # 构造退化四面体（四点共面 -> 体积为 0，质量差）
    bad_pts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0.5, 0]], dtype=float
    )
    bad_tets = np.array([[0, 1, 2, 3]])
    stats2 = _mesh_quality_report(bad_tets, bad_pts)
    assert stats2["min"] < 1e-9
    assert stats2["verdict"] == "需细化"


def test_mesh_step_file_fallback_when_gmsh_missing(monkeypatch, tmp_path):
    """gmsh 缺失时应优雅回退（不抛异常）。"""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "gmsh":
            raise ImportError("no gmsh")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    step = tmp_path / "tiny.step"
    step.write_text(
        """ISO-10303-21;
HEADER;
FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('p0',(0.,0.,0.));
#2=CARTESIAN_POINT('p1',(10.,0.,0.));
ENDSEC;
END-ISO-10303-21;
""",
        encoding="utf-8",
    )
    result = mesh_step_file(step)
    assert result["ok"] is False
    assert result["mesher"] == "fallback"
    assert len(result["points"]) >= 2


def test_mesh_step_file_in_background_thread(tmp_path):
    """后台线程内网格化不因 signal 注册失败而回退（回归测试）。"""
    gmsh = pytest.importorskip("gmsh")
    import queue
    import threading

    step = tmp_path / "box.step"
    gmsh.initialize(readConfigFiles=False, run=False, interruptible=False)
    gmsh.model.occ.addBox(0, 0, 0, 10, 10, 10)
    gmsh.model.occ.synchronize()
    gmsh.write(str(step))
    gmsh.finalize()

    q = queue.Queue()

    def worker():
        try:
            q.put(mesh_step_file(step))
        except Exception as exc:  # noqa: BLE001
            q.put(exc)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(120)
    result = q.get(timeout=5)
    assert isinstance(result, dict)
    assert result["ok"] is True
    assert result["mesher"] == "gmsh"
