"""自动权重（ANSYS 力学信息）测试。"""

from pathlib import Path

import numpy as np
import pytest

from satc.mechanics import (
    _component_from_filename,
    classify_ansys_files,
    suggest_weights,
)


def _write_node_file(tmp_path, nodes):
    path = tmp_path / "nodes.csv"
    lines = ["Node,X,Y,Z"]
    for node, x, y, z in nodes:
        lines.append(f"{node},{x},{y},{z}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def _write_stress_folder(tmp_path, values_by_node):
    folder = tmp_path / "stress"
    folder.mkdir(exist_ok=True)
    components = ["SX", "SY", "SZ", "SXY", "SYZ", "SXZ"]
    files = ["X.txt", "Y.txt", "Z.txt", "XY.txt", "YZ.txt", "XZ.txt"]
    for file, comp in zip(files, components):
        idx = components.index(comp)
        lines = ["Node Number    Stress (Pa)"]
        for node, vals in values_by_node.items():
            lines.append(f"{node}    {vals[idx]}")
        (folder / file).write_text("\n".join(lines), encoding="utf-8")
    return str(folder)


def _write_deform_file(tmp_path, values_by_node):
    path = tmp_path / "deform.txt"
    lines = ["Node Number    Total Deformation (mm)"]
    for node, d in values_by_node.items():
        lines.append(f"{node}    {d}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def test_suggest_weights_equal_when_no_mechanics(tmp_path):
    nodes = [(1, 0, 0, 0), (2, 1, 0, 0), (3, 0, 1, 0)]
    info = suggest_weights(_write_node_file(tmp_path, nodes))
    assert np.allclose(info["weights"], [1 / 3] * 3)
    assert info["n_nodes"] == 3


def test_suggest_weights_tension_dominant(tmp_path):
    nodes = [(1, 0, 0, 0), (2, 1, 0, 0), (3, 0, 1, 0), (4, 1, 1, 0.5)]
    stress = {n: [100.0, 0.0, 0.0, 0.0, 0.0, 0.0] for n in range(1, 5)}
    deform = {1: 0.001, 2: 0.01, 3: 0.1, 4: 1.0}

    info = suggest_weights(
        _write_node_file(tmp_path, nodes),
        stress_folder=_write_stress_folder(tmp_path, stress),
        deformation_path=_write_deform_file(tmp_path, deform),
    )
    w = info["weights"]
    assert np.isclose(w.sum(), 1.0)
    assert w[0] > w[1] > w[2]
    assert info["exposures"]["tensile"] == 1.0
    assert info["exposures"]["shear"] == 0.0


def test_suggest_weights_bending_proxy(tmp_path):
    """无变形文件时，用分层 σ1 变化近似弯曲暴露度。"""
    nodes = [
        (1, 0, 0, 0.0),
        (2, 0, 0, 0.1),
        (3, 0, 0, 0.2),
        (4, 0, 0, 0.3),
    ]
    # 上层受拉、下层受压：典型弯曲应力分布
    stress = {
        1: [-100.0, 0, 0, 0, 0, 0],
        2: [-100.0, 0, 0, 0, 0, 0],
        3: [100.0, 0, 0, 0, 0, 0],
        4: [100.0, 0, 0, 0, 0, 0],
    }
    info = suggest_weights(
        _write_node_file(tmp_path, nodes),
        stress_folder=_write_stress_folder(tmp_path, stress),
    )
    w = info["weights"]
    assert np.isclose(w.sum(), 1.0)
    assert w[1] > w[0]


def test_classify_ansys_files(tmp_path):
    nodes = _write_node_file(tmp_path, [(1, 0, 0, 0), (2, 1, 0, 0)])
    deform = _write_deform_file(tmp_path, {1: 0.1, 2: 0.2})
    stress_folder = _write_stress_folder(
        tmp_path, {1: [10.0] * 6, 2: [20.0] * 6}
    )
    stress_files = [
        str(Path(stress_folder) / name)
        for name in ("X.txt", "Y.txt", "Z.txt", "XY.txt", "YZ.txt", "XZ.txt")
    ]

    node_path, sf, df, notes = classify_ansys_files(
        [nodes, deform, *stress_files]
    )
    assert node_path == nodes
    assert set(sf) == {"SX", "SY", "SZ", "SXY", "SYZ", "SXZ"}
    assert sf["SX"] == str(Path(stress_folder) / "X.txt")
    assert df == deform
    assert notes == []


def test_classify_ansys_files_node_only(tmp_path):
    nodes = _write_node_file(tmp_path, [(1, 0, 0, 0)])
    node_path, sf, df, notes = classify_ansys_files([nodes])
    assert node_path == nodes
    assert sf is None
    assert df is None
    assert notes == []


def test_classify_ansys_files_missing_stress(tmp_path):
    nodes = _write_node_file(tmp_path, [(1, 0, 0, 0)])
    folder = tmp_path / "stress_partial"
    folder.mkdir()
    (folder / "X.txt").write_text("Node Value\n1 10\n", encoding="utf-8")

    _, sf, _, notes = classify_ansys_files([nodes, str(folder / "X.txt")])
    assert sf is None
    assert any("缺少应力分量" in note for note in notes)


def test_classify_ansys_files_ambiguous(tmp_path):
    nodes = _write_node_file(tmp_path, [(1, 0, 0, 0)])
    other = tmp_path / "other.csv"
    other.write_text("Node,X,Y,Z\n1,0,0,0\n", encoding="utf-8")

    with pytest.raises(ValueError):
        classify_ansys_files([nodes, str(other)])


def test_component_from_filename_variants():
    assert _component_from_filename("X.txt") == "SX"
    assert _component_from_filename("x.txt") == "SX"
    assert _component_from_filename("X.TXT") == "SX"
    assert _component_from_filename("X.txt ") == "SX"
    assert _component_from_filename("Y") == "SY"
    assert _component_from_filename("xy .txt") == "SXY"
    assert _component_from_filename("data.csv") is None
