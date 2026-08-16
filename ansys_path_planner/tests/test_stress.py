"""应力分析测试。"""
import numpy as np
import pandas as pd
import pytest

from path_planner import config
from path_planner.analysis.stress import (
    compute_principal_stress,
    merge_ansys_folder_data,
)


def _stress_df(rows):
    return pd.DataFrame(
        rows, columns=["Node"] + config.STRESS_COMPONENTS
    )


def test_pure_shear_principal_stress():
    """纯剪切：主应力应为 ±SXY，最大主应力方向为 [1,1,0]/√2。"""
    df = _stress_df([(1, 0, 0, 0, 10.0, 0, 0)])
    out = compute_principal_stress(df)

    assert out.iloc[0]["Maximum_Principal"] == pytest.approx(10.0)
    assert out.iloc[0]["Minimum_Principal"] == pytest.approx(-10.0)
    assert out.iloc[0]["Von_Mises"] == pytest.approx(np.sqrt(3) * 10.0)

    vx = out.iloc[0]["Principal_VX"]
    vy = out.iloc[0]["Principal_VY"]
    assert abs(vx - vy) < 1e-9
    assert abs(vx) > 0.7


def test_triaxial_von_mises():
    """三向正应力无剪应力：主应力等于正应力，von-Mises 按公式校验。"""
    df = _stress_df([(1, 100.0, 50.0, 0.0, 0.0, 0.0, 0.0)])
    out = compute_principal_stress(df)

    assert out.iloc[0]["Maximum_Principal"] == pytest.approx(100.0)
    assert out.iloc[0]["Middle_Principal"] == pytest.approx(50.0)
    assert out.iloc[0]["Minimum_Principal"] == pytest.approx(0.0)
    assert out.iloc[0]["Von_Mises"] == pytest.approx(np.sqrt(7500.0))


def test_merge_ansys_folder_data(tmp_path):
    coord = tmp_path / "coords.csv"
    coord.write_text(
        "Node,X,Y,Z\n1,0.0,0.0,0.0\n2,1.0,0.0,0.0\n",
        encoding="utf-8",
    )

    stress_dir = tmp_path / "stress"
    stress_dir.mkdir()
    for short in ("X", "Y", "Z", "XY", "YZ", "XZ"):
        (stress_dir / f"{short}.txt").write_text(
            "Node Number    Stress (Pa)\n"
            "1              10.0\n"
            "2              20.0\n",
            encoding="utf-8",
        )

    merged = merge_ansys_folder_data(str(coord), str(stress_dir))
    assert list(merged.columns)[:5] == ["Node", "X", "Y", "Z", "SX"]
    assert "Maximum_Principal" in merged.columns
    assert "Von_Mises" in merged.columns
    assert len(merged) == 2
