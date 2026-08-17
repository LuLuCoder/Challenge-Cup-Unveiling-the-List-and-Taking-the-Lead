"""ANSYS 节点坐标文件解析。"""
import warnings

import pandas as pd

from path_planner.utils.numeric import numeric_tokens
from path_planner.utils.text import read_text_auto

_COLUMNS = ["Node", "X", "Y", "Z"]


def _first_line(path):
    """读取文件第一行（按多种编码尝试解码）。"""
    with open(path, "rb") as f:
        head = f.readline(4096)
    for enc in ("utf-8-sig", "utf-8", "gb18030", "latin1"):
        try:
            return head.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return head.decode("latin1", errors="replace")


def _looks_like_header(line):
    return any(ch.isalpha() for ch in line)


def _try_fast_parse(path):
    """pandas C 引擎快速路径；失败（格式异常）返回 None 交给旧解析器。

    兼容 ANSYS 常见导出：
        Node,X,Y
            1., -0.25000000E+01,  0.00000000E+00,  0.50000000E+01
    即表头少列、数据行逗号+空格混合、数字带尾点与科学计数法。
    """
    first = _first_line(path)
    skiprows = 1 if _looks_like_header(first) else 0

    raw = None
    attempts = (
        {"sep": ",", "skipinitialspace": True, "engine": "c"},
        {"delim_whitespace": True, "engine": "c"},
    )
    for kwargs in attempts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                candidate = pd.read_csv(
                    path, header=None, skiprows=skiprows, **kwargs
                )
        except Exception:
            continue
        if candidate is not None and candidate.shape[1] >= 4:
            raw = candidate
            break
    if raw is None:
        return None

    nums = raw.iloc[:, :4].apply(pd.to_numeric, errors="coerce")
    nums = nums.dropna()
    if len(nums) == 0:
        return None

    df = nums.copy()
    df.columns = _COLUMNS
    df["Node"] = df["Node"].astype(int)
    df = df.drop_duplicates(subset=["Node"], keep="first")
    return df.sort_values("Node").reset_index(drop=True)


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
    fast = _try_fast_parse(path)
    if fast is not None:
        return fast

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
