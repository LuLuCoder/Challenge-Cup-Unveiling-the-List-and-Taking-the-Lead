"""应力张量分析：主应力、主方向、von-Mises，以及数据融合入口。"""
import numpy as np
import pandas as pd

from path_planner import config
from path_planner.parsers.coordinates import parse_coordinate_file
from path_planner.parsers.results import parse_result_file
from path_planner.parsers.stress_components import (
    load_six_stress_components,
    load_stress_components,
)


def _continuize_directions(directions):
    """v 与 -v 表示同一主方向，按节点顺序翻转符号避免方向场跳变。"""
    result = directions.copy()
    for i in range(1, len(result)):
        if np.dot(result[i], result[i - 1]) < 0:
            result[i] *= -1
    return result


def compute_principal_stress(df):
    """
    根据六个应力分量计算最大/中间/最小主应力、最大主应力方向、von-Mises 应力。

    应力张量：
        [ SX   SXY  SXZ ]
        [ SXY  SY   SYZ ]
        [ SXZ  SYZ  SZ  ]
    """
    missing = [col for col in config.STRESS_COMPONENTS if col not in df.columns]
    if missing:
        raise ValueError(f"缺少应力分量：{missing}")

    tensors = np.zeros((len(df), 3, 3), dtype=float)
    tensors[:, 0, 0] = df["SX"].to_numpy(float)
    tensors[:, 1, 1] = df["SY"].to_numpy(float)
    tensors[:, 2, 2] = df["SZ"].to_numpy(float)
    tensors[:, 0, 1] = tensors[:, 1, 0] = df["SXY"].to_numpy(float)
    tensors[:, 1, 2] = tensors[:, 2, 1] = df["SYZ"].to_numpy(float)
    tensors[:, 0, 2] = tensors[:, 2, 0] = df["SXZ"].to_numpy(float)

    # 批量特征分解：eigh 返回升序特征值及其特征向量
    # （等价于旧的逐节点循环，但一次 BLAS 调用完成，大数据快很多）
    principal_values, principal_vectors = np.linalg.eigh(tensors)

    result = df.copy()
    result["Minimum_Principal"] = principal_values[:, 0]
    result["Middle_Principal"] = principal_values[:, 1]
    result["Maximum_Principal"] = principal_values[:, 2]

    directions = _continuize_directions(principal_vectors[:, :, 2])
    result["Principal_VX"] = directions[:, 0]
    result["Principal_VY"] = directions[:, 1]
    result["Principal_VZ"] = directions[:, 2]

    sx, sy, sz = (result[col].to_numpy(float) for col in ["SX", "SY", "SZ"])
    sxy, syz, sxz = (
        result[col].to_numpy(float) for col in ["SXY", "SYZ", "SXZ"]
    )
    result["Von_Mises"] = np.sqrt(
        0.5 * (
            (sx - sy) ** 2
            + (sy - sz) ** 2
            + (sz - sx) ** 2
            + 6 * (sxy ** 2 + syz ** 2 + sxz ** 2)
        )
    )

    return result


def merge_ansys_folder_data(coord_path, stress_folder):
    """
    一次性完成：坐标 + 六个应力分量 -> Node 匹配 -> 应力张量分析。
    """
    coord_df = parse_coordinate_file(coord_path)
    stress_df = load_six_stress_components(stress_folder)

    merged = pd.merge(coord_df, stress_df, on="Node", how="inner")
    if merged.empty:
        raise ValueError("节点坐标与六个应力文件没有匹配到共同 Node。")

    merged = compute_principal_stress(merged)
    return merged.sort_values("Node").reset_index(drop=True)


def merge_ansys_files_data(coord_path, stress_files):
    """
    多选文件方式：坐标 + {分量名: 路径} 六个应力分量
    -> Node 匹配 -> 应力张量分析。
    """
    coord_df = parse_coordinate_file(coord_path)
    stress_df = load_stress_components(stress_files)

    merged = pd.merge(coord_df, stress_df, on="Node", how="inner")
    if merged.empty:
        raise ValueError("节点坐标与六个应力文件没有匹配到共同 Node。")

    merged = compute_principal_stress(merged)
    return merged.sort_values("Node").reset_index(drop=True)


def merge_ansys_data(coord_path, result_path):
    """
    兼容旧版：坐标文件 + 单文件仿真结果 -> Node,X,Y,Z,SimulationValue。

    新版界面使用六应力分量文件夹方式，此函数保留供批处理/命令行使用。
    """
    coord_df = parse_coordinate_file(coord_path)
    result_df, result_name = parse_result_file(result_path)

    merged = pd.merge(coord_df, result_df, on="Node", how="inner")
    if merged.empty:
        raise ValueError(
            "节点号无法匹配。\n"
            "请检查坐标文件和仿真结果文件是否来自同一个 ANSYS 模型。"
        )

    merged = merged.sort_values("Node").reset_index(drop=True)
    return merged, result_name, coord_df, result_df
