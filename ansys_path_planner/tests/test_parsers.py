"""解析器测试：节点坐标、单文件结果、六应力分量。"""
import pytest

from path_planner.parsers.coordinates import parse_coordinate_file
from path_planner.parsers.results import parse_result_file
from path_planner.parsers.stress_components import load_six_stress_components


@pytest.fixture
def coord_file(tmp_path):
    path = tmp_path / "coords.csv"
    path.write_text(
        "Node,X,Y,Z\n"
        "3,3.0,3.0,3.0\n"
        "1,1.0,1.0,1.0\n"
        "2,2.0,2.0,2.0\n",
        encoding="utf-8",
    )
    return str(path)


def test_parse_coordinate_file(coord_file):
    df = parse_coordinate_file(coord_file)
    assert list(df.columns) == ["Node", "X", "Y", "Z"]
    assert list(df["Node"]) == [1, 2, 3]


def test_parse_coordinate_glued_yz(tmp_path):
    """表头少 Z 但数据行实际有 4 个数的 ANSYS 导出异常。"""
    path = tmp_path / "coords.txt"
    path.write_text(
        "Node,X,Y\n"
        "1,-0.002500,0.00000000E+00,0.50000000E-02\n",
        encoding="utf-8",
    )
    df = parse_coordinate_file(str(path))
    assert len(df) == 1
    assert df.iloc[0]["Y"] == pytest.approx(0.0)
    assert df.iloc[0]["Z"] == pytest.approx(0.005)


def test_parse_result_file(tmp_path):
    path = tmp_path / "result.txt"
    path.write_text(
        "Node Number    Equivalent (von-Mises) Stress (Pa)\n"
        "1              3.246e+006\n"
        "2              3.2855e+006\n",
        encoding="utf-8",
    )
    df, name = parse_result_file(str(path))
    assert name == "Equivalent_Stress"
    assert list(df["Node"]) == [1, 2]
    assert df.iloc[0]["Equivalent_Stress"] == pytest.approx(3.246e6)


def test_load_six_stress_components(tmp_path):
    for short in ("X", "Y", "Z", "XY", "YZ", "XZ"):
        (tmp_path / f"{short}.txt").write_text(
            "Node Number    Stress (Pa)\n"
            "1              10.0\n"
            "2              20.0\n",
            encoding="utf-8",
        )

    df = load_six_stress_components(str(tmp_path))
    assert list(df.columns) == [
        "Node", "SX", "SY", "SZ", "SXY", "SYZ", "SXZ",
    ]
    assert df.iloc[1]["SX"] == pytest.approx(20.0)


def test_load_six_stress_components_missing_file(tmp_path):
    (tmp_path / "X.txt").write_text("Node Value\n1 10.0\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_six_stress_components(str(tmp_path))
