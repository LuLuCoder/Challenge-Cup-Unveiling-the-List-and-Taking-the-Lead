"""ANSYS 单文件仿真结果解析（如 von-Mises 应力文件）。"""
import re

import pandas as pd

from path_planner.utils.numeric import numeric_tokens
from path_planner.utils.text import read_text_auto


def detect_result_name(text, default="SimulationValue"):
    """从 ANSYS 文件头尝试识别结果名称。"""
    for line in text.splitlines()[:30]:
        low = line.lower()
        if "von-mises" in low or "von mises" in low:
            return "Equivalent_Stress"
        if "stress" in low:
            return "Stress"
        if "temperature" in low:
            return "Temperature"
        if "deformation" in low:
            return "Total_Deformation"
        if "displacement" in low:
            return "Displacement"
    return default


def parse_result_file(path):
    """
    解析 ANSYS 仿真结果（节点号 + 结果值两列）。

    典型格式：
        Node Number    Equivalent (von-Mises) Stress (Pa)
        1              3.246e+006

    也兼容：
        Node,Stress
        1,3.246e6
    """
    text = read_text_auto(path)
    result_name = detect_result_name(text)
    rows = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        low = line.lower()
        if "node number" in low:
            continue
        if low.startswith("node") and not re.match(r"node\s*\d", low):
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
            "没有识别到 ANSYS 仿真结果。\n"
            "请确认文件中包含：节点号 + 仿真结果值。"
        )

    df = pd.DataFrame(rows, columns=["Node", result_name])
    df["Node"] = df["Node"].astype(int)
    df[result_name] = pd.to_numeric(df[result_name], errors="coerce")

    df = df.dropna(subset=["Node", result_name])
    df = df.drop_duplicates(subset=["Node"], keep="last")
    return df.sort_values("Node").reset_index(drop=True), result_name
