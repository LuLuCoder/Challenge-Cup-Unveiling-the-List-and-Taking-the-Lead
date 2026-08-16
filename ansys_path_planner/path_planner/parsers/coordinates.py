"""ANSYS 节点坐标文件解析。"""
import pandas as pd

from path_planner.utils.numeric import numeric_tokens
from path_planner.utils.text import read_text_auto

_COLUMNS = ["Node", "X", "Y", "Z"]


def parse_coordinate_file(path):
    """
    解析 ANSYS 节点坐标文件。

    兼容：
        Node,X,Y,Z
        1,1.25,5.21,8.42

    也兼容 ANSYS 某些导出异常：
        Node,X,Y
        1,-0.002500,0.00000000E+00,0.50000000E-02
    即表头少 Z，但数据行实际有 4 个数（Y/Z 粘在一起的情况）。
    """
    text = read_text_auto(path)
    rows = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        low = line.lower()
        if "node" in low and ("x" in low or "coordinate" in low):
            continue

        nums = numeric_tokens(line)
        if len(nums) >= 4:
            try:
                node = int(float(nums[0]))
                x = float(nums[1])
                y = float(nums[2])
                z = float(nums[3])
                rows.append([node, x, y, z])
            except (TypeError, ValueError):
                pass

    if not rows:
        raise ValueError(
            "没有识别到节点坐标。\n"
            "请确认文件中包含 Node,X,Y,Z 数据。"
        )

    df = pd.DataFrame(rows, columns=_COLUMNS)
    df["Node"] = df["Node"].astype(int)
    for col in ["X", "Y", "Z"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=_COLUMNS)
    df = df.drop_duplicates(subset=["Node"], keep="first")
    return df.sort_values("Node").reset_index(drop=True)
