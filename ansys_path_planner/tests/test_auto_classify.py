"""自动分类测试：多选 ANSYS 文件导入（与 SATC 优化器一致）。"""

import pytest

from path_planner import config
from path_planner.parsers.auto_classify import classify_ansys_files


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return str(path)


def _stress_file(tmp_path, short):
    return _write(
        tmp_path / f"{short}.txt",
        "Node Number    Stress (Pa)\n"
        "1              10.0\n"
        "2              20.0\n",
    )


def _coord_file(tmp_path):
    return _write(
        tmp_path / "coords.csv",
        "Node,X,Y,Z\n"
        "1,1.0,1.0,1.0\n"
        "2,2.0,2.0,2.0\n",
    )


def test_classify_all_files(tmp_path):
    paths = [_coord_file(tmp_path)] + [
        _stress_file(tmp_path, short)
        for short in ("X", "Y", "Z", "XY", "YZ", "XZ")
    ]
    node_path, stress_files, skipped = classify_ansys_files(paths)
    assert node_path.endswith("coords.csv")
    assert sorted(stress_files) == sorted(config.STRESS_COMPONENTS)
    assert skipped == []


def test_classify_case_insensitive(tmp_path):
    paths = [_coord_file(tmp_path)] + [
        _write(
            tmp_path / f"{short.upper()}.TXT",
            "Node Value\n1 10.0\n",
        )
        for short in ("x", "y", "z", "xy", "yz", "xz")
    ]
    node_path, stress_files, skipped = classify_ansys_files(paths)
    assert set(stress_files) == set(config.STRESS_COMPONENTS)
    assert skipped == []


def test_classify_missing_component(tmp_path):
    paths = [_coord_file(tmp_path), _stress_file(tmp_path, "X")]
    with pytest.raises(ValueError, match="缺少应力分量"):
        classify_ansys_files(paths)


def test_classify_multiple_node_files(tmp_path):
    paths = [_coord_file(tmp_path), _coord_file(tmp_path)]
    with pytest.raises(ValueError, match="多个节点坐标"):
        classify_ansys_files(paths)


def test_classify_skips_auxiliary_file(tmp_path):
    deform = _write(
        tmp_path / "deform.txt",
        "Node Deformation\n1 0.5\n2 0.6\n",
    )
    paths = [_coord_file(tmp_path), deform] + [
        _stress_file(tmp_path, short)
        for short in ("X", "Y", "Z", "XY", "YZ", "XZ")
    ]
    node_path, stress_files, skipped = classify_ansys_files(paths)
    assert skipped == [deform]
