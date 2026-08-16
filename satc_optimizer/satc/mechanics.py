"""基于 ANSYS 力学结果的自动权重推荐。

输入文件格式与 ANSYS 路径规划项目一致：
    1. 节点坐标文件：Node, X, Y, Z（CSV/TXT）
    2. 六应力分量文件夹：X.txt / Y.txt / Z.txt / XY.txt / YZ.txt / XZ.txt
    3. 变形结果文件：节点号 + 变形值（两列）

思路：
    从力学结果中提取三类失效风险暴露度，按占比给出
    ΔT（拉伸）/ ΔB（弯曲）/ ΔS（层间剪切）权重：
    - 拉伸暴露度：最大主应力 σ1 的正值部分（拉应力水平）；
    - 弯曲暴露度：总变形量（挠度）；若缺少变形文件，
      用沿打印方向（默认 Z）各层平均 σ1 的变化幅度近似；
    - 层间剪切暴露度：层间剪应力 sqrt(τ_xz² + τ_yz²)。
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd


STRESS_FILE_MAP = {
    "X.txt": "SX",
    "Y.txt": "SY",
    "Z.txt": "SZ",
    "XY.txt": "SXY",
    "YZ.txt": "SYZ",
    "XZ.txt": "SXZ",
}
SLICE_AXIS = "Z"
NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


# ---------- 文件解析（与路径规划项目相同格式） ----------

def _read_text_auto(path):
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]
    last_error = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except Exception as e:
            last_error = e
    raise RuntimeError(f"无法读取文件：{path}\n{last_error}")


def _numeric_tokens(line):
    return re.findall(NUMBER_PATTERN, line)


def _parse_node_value_file(path, value_name):
    """解析两列文件：节点号 + 数值（兼容表头/分隔线）。"""
    text = _read_text_auto(path)
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if "node" in low and not re.match(r"node\s*\d", low):
            continue
        if set(line) <= set("-_= \t,"):
            continue
        nums = _numeric_tokens(line)
        if len(nums) >= 2:
            try:
                rows.append([int(float(nums[0])), float(nums[1])])
            except (TypeError, ValueError):
                pass
    if not rows:
        raise ValueError(f"文件没有识别到有效数据：{path}")
    df = pd.DataFrame(rows, columns=["Node", value_name])
    df["Node"] = df["Node"].astype(int)
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    return df.dropna().drop_duplicates(subset=["Node"], keep="last")


def parse_node_file(path):
    """解析 ANSYS 节点坐标文件，返回 Node,X,Y,Z。"""
    text = _read_text_auto(path)
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if "node" in low and ("x" in low or "coordinate" in low):
            continue
        nums = _numeric_tokens(line)
        if len(nums) >= 4:
            try:
                rows.append([
                    int(float(nums[0])),
                    float(nums[1]),
                    float(nums[2]),
                    float(nums[3]),
                ])
            except (TypeError, ValueError):
                pass
    if not rows:
        raise ValueError(f"没有识别到节点坐标：{path}")
    df = pd.DataFrame(rows, columns=["Node", "X", "Y", "Z"])
    df["Node"] = df["Node"].astype(int)
    for col in ["X", "Y", "Z"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna().drop_duplicates(subset=["Node"], keep="first")


def parse_stress_folder(folder):
    """解析六个应力分量文件夹，返回 Node,SX,SY,SZ,SXY,SYZ,SXZ。"""
    folder = Path(folder)
    missing = [name for name in STRESS_FILE_MAP if not (folder / name).exists()]
    if missing:
        raise FileNotFoundError(
            "缺少 ANSYS 应力分量文件：\n" + "\n".join(missing)
        )

    merged = None
    for filename, component in STRESS_FILE_MAP.items():
        df = _parse_node_value_file(folder / filename, "Value").rename(
            columns={"Value": component}
        )
        merged = (
            df if merged is None
            else pd.merge(merged, df, on="Node", how="inner")
        )
    if merged is None or merged.empty:
        raise ValueError("六个应力文件没有找到共同 Node。")
    return merged


def load_stress_components(component_files):
    """按 {分量名: 文件路径} 解析六个应力分量并按 Node 合并。"""
    merged = None
    for component in STRESS_FILE_MAP.values():
        path = component_files.get(component)
        if path is None:
            raise ValueError(f"缺少应力分量：{component}")
        df = _parse_node_value_file(path, "Value").rename(
            columns={"Value": component}
        )
        merged = (
            df if merged is None
            else pd.merge(merged, df, on="Node", how="inner")
        )
    if merged is None or merged.empty:
        raise ValueError("六个应力文件没有找到共同 Node。")
    return merged


def parse_deformation_file(path):
    """解析变形结果文件，返回 Node,Deformation。"""
    return _parse_node_value_file(path, "Deformation")


def _first_numeric_line(path):
    """返回文件中第一条含至少两个数字的行（用于文件类型识别）。"""
    text = _read_text_auto(path)
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        nums = _numeric_tokens(line)
        if len(nums) >= 2:
            return nums
    return []


def _component_from_filename(path):
    """
    根据文件名识别应力分量。
    兼容大小写、扩展名大小写、文件名首尾空格：
    X.txt / x.txt / X.TXT / "X.txt " 等均识别为 SX。
    """
    name = Path(path).name.strip().lower()
    if name in STRESS_FILE_MAP:
        return STRESS_FILE_MAP[name]

    stem = name.split(".")[0].strip()
    letter_map = {
        "x": "SX",
        "y": "SY",
        "z": "SZ",
        "xy": "SXY",
        "yz": "SYZ",
        "xz": "SXZ",
    }
    return letter_map.get(stem)


def classify_ansys_files(paths):
    """
    把一次多选的文件自动分类。

    - 按文件名识别六个应力分量（大小写、首尾空格均兼容）；
    - 按内容识别节点坐标（一行 ≥4 个数字）与变形结果（一行 2 个数字）。

    返回：
        node_path        节点坐标文件
        stress_files     应力分量文件字典 {分量名: 路径}（不全时为 None）
        deformation_path 变形结果文件（未提供为 None）
        notes            提示信息列表
    """
    stress_files = {}
    node_path = None
    deformation_path = None

    for path in paths:
        component = _component_from_filename(path)
        if component is not None:
            stress_files[component] = str(path)
            continue

        tokens = _first_numeric_line(path)
        if len(tokens) >= 4:
            if node_path is not None:
                raise ValueError(f"识别到多个节点坐标文件：{node_path} 与 {path}")
            node_path = str(path)
        elif len(tokens) >= 2:
            if deformation_path is not None:
                raise ValueError(f"识别到多个变形结果文件：{deformation_path} 与 {path}")
            deformation_path = str(path)
        else:
            raise ValueError(f"无法识别文件：{path}")

    notes = []
    if stress_files:
        missing = [
            comp for comp in STRESS_FILE_MAP.values()
            if comp not in stress_files
        ]
        if missing:
            notes.append(
                "缺少应力分量：" + "、".join(missing)
                + "，本次将跳过应力数据"
            )
            stress_files = None

    if node_path is None:
        raise ValueError("未识别到节点坐标文件（需包含 Node,X,Y,Z 数据）。")
    return node_path, (stress_files or None), deformation_path, notes


# ---------- 力学指标 ----------

def _max_principal_stress(df):
    """由六个应力分量逐节点计算最大主应力 σ1。"""
    tensors = np.zeros((len(df), 3, 3), dtype=float)
    tensors[:, 0, 0] = df["SX"].to_numpy(float)
    tensors[:, 1, 1] = df["SY"].to_numpy(float)
    tensors[:, 2, 2] = df["SZ"].to_numpy(float)
    tensors[:, 0, 1] = tensors[:, 1, 0] = df["SXY"].to_numpy(float)
    tensors[:, 1, 2] = tensors[:, 2, 1] = df["SYZ"].to_numpy(float)
    tensors[:, 0, 2] = tensors[:, 2, 0] = df["SXZ"].to_numpy(float)

    sigma1 = np.zeros(len(df))
    for i, tensor in enumerate(tensors):
        values = np.linalg.eigvalsh(tensor)
        sigma1[i] = values[-1]
    return sigma1


def _interlaminar_shear(df):
    """层间剪应力：sqrt(τ_xz² + τ_yz²)（z 为打印方向）。"""
    return np.sqrt(df["SXZ"].to_numpy(float) ** 2
                   + df["SYZ"].to_numpy(float) ** 2)


def _exposure(values):
    """失效暴露度：高值区加权占比，mean((v / v_max)²)。"""
    values = np.asarray(values, dtype=float)
    vmax = float(np.max(np.abs(values))) if len(values) else 0.0
    if vmax < 1e-12:
        return 0.0
    normalized = values / vmax
    return float(np.mean(normalized ** 2))


def _bending_exposure_from_stress(z_values, sigma1, n_layers=10):
    """弯曲代理指标：沿打印方向各层平均 σ1 的变化幅度。"""
    z = np.asarray(z_values, dtype=float)
    s = np.asarray(sigma1, dtype=float)
    zmin, zmax = float(z.min()), float(z.max())
    if zmax - zmin < 1e-12:
        return None

    edges = np.linspace(zmin, zmax, n_layers + 1)
    layer_means = []
    for k in range(n_layers):
        if k == n_layers - 1:
            mask = (z >= edges[k]) & (z <= edges[k + 1])
        else:
            mask = (z >= edges[k]) & (z < edges[k + 1])
        if np.sum(mask) > 0:
            layer_means.append(float(np.mean(s[mask])))

    if not layer_means:
        return None
    span = max(layer_means) - min(layer_means)
    scale = max(float(np.max(np.abs(s))), 1e-12)
    return min(max(span / scale, 0.0), 1.0)


def _weights_from_exposures(exposures):
    """
    由三个暴露度生成权重。
    无法计算的指标用已计算指标的均值作为基准，避免权重为 0。
    """
    computed = [e for e in exposures if e is not None]
    if not computed:
        return np.ones(3, dtype=float) / 3.0
    fallback = float(np.mean(computed))
    weights = np.array(
        [e if e is not None else fallback for e in exposures],
        dtype=float,
    )
    total = float(weights.sum())
    if total <= 1e-12:
        return np.ones(3, dtype=float) / 3.0
    return weights / total


# ---------- 主入口 ----------

def suggest_weights(node_path, stress_folder=None, deformation_path=None,
                    stress_files=None):
    """
    根据 ANSYS 力学结果自动推荐 ΔT/ΔB/ΔS 权重。

    stress_files：{分量名: 文件路径} 字典，与 stress_folder 二选一。

    返回：
        weights     归一化权重 [ΔT, ΔB, ΔS]
        exposures   三类暴露度（缺失为 None）
        explanation 权重依据的文字说明
        n_nodes     参与计算的节点数
    """
    nodes = parse_node_file(node_path)

    sigma1 = None
    shear = None
    if stress_files:
        stress = load_stress_components(stress_files)
    elif stress_folder:
        stress = parse_stress_folder(stress_folder)
    else:
        stress = None

    if stress is not None:
        merged = pd.merge(nodes, stress, on="Node", how="inner")
        if merged.empty:
            raise ValueError("节点坐标与六个应力文件没有匹配到共同 Node。")
        sigma1 = _max_principal_stress(merged)
        shear = _interlaminar_shear(merged)
        nodes = merged

    deformation = None
    if deformation_path:
        deform = parse_deformation_file(deformation_path)
        nodes = pd.merge(nodes, deform, on="Node", how="inner")
        if nodes.empty:
            raise ValueError("变形文件与节点坐标没有匹配到共同 Node。")
        deformation = nodes["Deformation"].to_numpy(float)

    tensile = _exposure(np.maximum(sigma1, 0.0)) if sigma1 is not None else None
    shear_exp = _exposure(shear) if shear is not None else None

    if deformation is not None:
        bending = _exposure(deformation)
    elif sigma1 is not None:
        bending = _bending_exposure_from_stress(
            nodes["Z"].to_numpy(float), sigma1
        )
    else:
        bending = None

    weights = _weights_from_exposures([tensile, bending, shear_exp])

    def fmt(value):
        return "未计算" if value is None else f"{value:.4f}"

    explanation = (
        "基于 ANSYS 力学结果的自动权重：\n"
        f"  拉伸暴露度 = {fmt(tensile)}（最大主应力 σ1，来自六应力分量）\n"
        f"  弯曲暴露度 = {fmt(bending)}"
        f"（来自{'变形文件' if deformation is not None else '分层σ1变化（无变形文件，近似）'}）\n"
        f"  层间剪切暴露度 = {fmt(shear_exp)}（τ_xz/τ_yz，来自六应力分量）\n"
        f"→ 权重 ΔT:ΔB:ΔS = "
        f"{weights[0]:.3f} : {weights[1]:.3f} : {weights[2]:.3f}"
    )
    if tensile is None and shear_exp is None:
        explanation += (
            "\n未提供六应力分量，拉伸/层剪沿用等权基准，"
            "建议补充应力文件以提高准确度。"
        )

    return {
        "weights": weights,
        "exposures": {
            "tensile": tensile,
            "bending": bending,
            "shear": shear_exp,
        },
        "explanation": explanation,
        "n_nodes": len(nodes),
    }
