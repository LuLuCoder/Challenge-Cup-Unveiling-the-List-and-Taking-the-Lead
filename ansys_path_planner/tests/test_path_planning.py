"""路径规划测试。"""
import numpy as np
import pandas as pd
import pytest

from path_planner.analysis.path_planning import (
    LayerPathPlanner,
    PathPlanner,
    generate_layer_path,
    generate_surface_path,
    normalize,
)


def _make_mesh(n=16):
    """构造带 +X 方向场的规则网格，用于验证追踪逻辑。"""
    xs = np.linspace(0, 1, n)
    ys = np.linspace(0, 1, n)
    zs = np.linspace(0, 0.5, 3)

    rows = []
    node = 1
    for z in zs:
        for y in ys:
            for x in xs:
                rows.append({
                    "Node": node,
                    "X": x,
                    "Y": y,
                    "Z": z,
                    "Maximum_Principal": 1.0 + x + y,
                    "Principal_VX": 1.0,
                    "Principal_VY": 0.0,
                    "Principal_VZ": 0.0,
                })
                node += 1
    return pd.DataFrame(rows)


def test_normalize():
    assert np.allclose(normalize(np.array([1.0, 3.0, 2.0])), [0.0, 1.0, 0.5])
    assert np.allclose(normalize(np.array([2.0, 2.0, 2.0])), 0.0)


def test_path_covers_all_nodes():
    df = _make_mesh()
    path_df, threshold = generate_surface_path(df, percentile=75.0)

    assert set(path_df["Node"]) == set(df["Node"])
    assert list(path_df["Step"]) == list(range(1, len(df) + 1))

    for col in (
        "Path_Spacing", "Path_Weight", "Density",
        "Segment_Length", "Value_Normalized",
        "Priority", "Path_Priority", "Density_Level",
    ):
        assert col in path_df.columns

    assert threshold == pytest.approx(np.percentile(df["Maximum_Principal"], 75.0))


def test_high_stress_region_gets_denser_path():
    df = _make_mesh()
    path_df, _ = generate_surface_path(df, percentile=75.0)

    high = path_df.loc[path_df["Priority"] == 1, "Path_Spacing"]
    low = path_df.loc[path_df["Priority"] == 0, "Path_Spacing"]
    assert high.mean() < low.mean()


def test_missing_columns_raise():
    df = pd.DataFrame({"X": [0.0], "Y": [0.0], "Z": [0.0]})
    with pytest.raises(ValueError):
        PathPlanner(df)


def test_layer_path_is_grouped_by_layer():
    df = _make_mesh()
    path_df, threshold = generate_layer_path(df, percentile=75.0, n_layers=3)

    assert set(path_df["Node"]) == set(df["Node"])
    assert list(path_df["Step"]) == list(range(1, len(df) + 1))

    layers = path_df["Layer"].to_numpy()
    assert set(layers) == {1, 2, 3}
    # 层式路径必须按层连续：Layer 列不递减
    assert (np.diff(layers) >= 0).all()

    for col in (
        "Layer", "Segment_Type", "SubPath", "Path_Spacing", "Path_Weight",
        "Density", "Segment_Length", "Value_Normalized",
        "Priority", "Path_Priority",
    ):
        assert col in path_df.columns

    assert (path_df["Segment_Type"] == "层内路径").sum() > 0
    # 3 层 -> 恰好 2 处层边界；层间段可能为层间过渡或空区断开
    boundary_rows = path_df["Segment_Type"].isin(["层间过渡", "空区断开"])
    assert boundary_rows.sum() == 2
    # 空区断开后子路径编号递增且不回流
    assert (np.diff(path_df["SubPath"].to_numpy()) >= 0).all()
    assert threshold > 0


def test_layer_path_respects_layer_count():
    df = _make_mesh()
    path_df, _ = generate_layer_path(df, n_layers=6)
    assert path_df["Layer"].min() == 1
    assert path_df["Layer"].max() <= 6
    assert (np.diff(path_df["Layer"].to_numpy()) >= 0).all()


def test_void_segment_detection():
    """同一层内相邻节点应连通；跨越无节点区域（不同 z 平面）应断开。"""
    df = _make_mesh()
    planner = LayerPathPlanner(df, n_layers=3)

    # 同层相邻节点：连通
    assert planner._segment_supported(0, 1)
    # 层内跨整行的锯齿连接（实体内部）：连通
    assert planner._segment_supported(0, 15)
    # z=0 平面 (x=0,y=0) 与 z=0.25 平面 (x=0,y=0) 之间无节点：断开
    assert not planner._segment_supported(0, 256)
