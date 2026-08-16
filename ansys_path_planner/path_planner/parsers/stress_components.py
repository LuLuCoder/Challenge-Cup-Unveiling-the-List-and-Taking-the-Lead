"""ANSYS 六个应力分量文件夹导入。"""
from pathlib import Path

import pandas as pd

from path_planner import config
from path_planner.utils.numeric import numeric_tokens
from path_planner.utils.text import read_text_auto


def parse_ansys_stress_component(path):
    """
    解析 ANSYS 单个应力分量文件（Node Number + 数值）。

    实际分量由文件名决定（见 config.STRESS_FILE_MAP），而不是由表头决定。
    """
    text = read_text_auto(path)
    rows = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        low = line.lower()
        if "node number" in low:
            continue
        if set(line) <= set("-_= \t,"):
            continue

        nums = numeric_tokens(line)
        if len(nums) >= 2:
            try:
                rows.append([int(float(nums[0])), float(nums[1])])
            except (TypeError, ValueError):
                pass

    if not rows:
        raise ValueError(
            f"文件没有识别到有效数据：{path}\n"
            "应包含 Node Number + Stress 数值。"
        )

    df = pd.DataFrame(rows, columns=["Node", "Value"])
    df["Node"] = df["Node"].astype(int)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Node", "Value"])
    df = df.drop_duplicates(subset=["Node"], keep="last")
    return df


def load_six_stress_components(folder):
    """
    从同一个文件夹自动读取六个分量文件：
        X.txt -> SX, Y.txt -> SY, ..., XZ.txt -> SXZ

    返回列：Node, SX, SY, SZ, SXY, SYZ, SXZ
    """
    folder = Path(folder)
    missing = [
        name for name in config.STRESS_FILE_MAP
        if not (folder / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "缺少 ANSYS 应力分量文件：\n"
            + "\n".join(missing)
            + "\n\n需要：" + "、".join(config.STRESS_FILE_MAP)
        )

    merged = None
    for filename, component in config.STRESS_FILE_MAP.items():
        df = parse_ansys_stress_component(folder / filename).rename(
            columns={"Value": component}
        )
        merged = (
            df if merged is None
            else pd.merge(merged, df, on="Node", how="inner")
        )

    if merged is None or merged.empty:
        raise ValueError("六个应力文件没有找到共同 Node。")

    return merged.sort_values("Node").reset_index(drop=True)


def load_stress_components(component_files):
    """
    按 {分量名: 文件路径} 读取六个应力分量文件并按 Node 合并。

    返回列：Node, SX, SY, SZ, SXY, SYZ, SXZ
    """
    merged = None
    for component in config.STRESS_COMPONENTS:
        path = component_files.get(component)
        if path is None:
            raise ValueError(f"缺少应力分量：{component}")
        df = parse_ansys_stress_component(path).rename(
            columns={"Value": component}
        )
        merged = (
            df if merged is None
            else pd.merge(merged, df, on="Node", how="inner")
        )

    if merged is None or merged.empty:
        raise ValueError("六个应力文件没有找到共同 Node。")

    return merged.sort_values("Node").reset_index(drop=True)
