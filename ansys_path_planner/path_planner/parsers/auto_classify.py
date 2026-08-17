"""ANSYS 文件自动分类（与 SATC 优化器一致的导入方式）。"""

from pathlib import Path

from path_planner import config
from path_planner.utils.numeric import numeric_tokens
from path_planner.utils.text import read_text_auto


_STRESS_NAME_MAP = {
    name.lower(): component
    for name, component in config.STRESS_FILE_MAP.items()
}
_LETTER_MAP = {
    "x": "SX",
    "y": "SY",
    "z": "SZ",
    "xy": "SXY",
    "yz": "SYZ",
    "xz": "SXZ",
}


def _component_from_filename(path):
    """根据文件名识别应力分量，兼容大小写、扩展名与首尾空格。"""
    name = Path(path).name.strip().lower()
    if name in _STRESS_NAME_MAP:
        return _STRESS_NAME_MAP[name]
    stem = name.split(".")[0].strip()
    return _LETTER_MAP.get(stem)


def _first_numeric_line(path):
    """返回文件头中第一条含至少两个数字的行（用于内容识别）。

    只读文件头部（8 KB），避免对超大文件做整读，加快导入。
    """
    with open(path, "rb") as f:
        head = f.read(8192)
    text = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "latin1"):
        try:
            text = head.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if text is None:
        text = head.decode("latin1", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        nums = numeric_tokens(line)
        if len(nums) >= 2:
            return nums
    return []


def classify_ansys_files(paths, require_stress=True):
    """
    把一次多选的文件自动分类。

    - 按文件名识别六个应力分量（大小写、扩展名、首尾空格均兼容）；
    - 按内容识别节点坐标（一行 >= 4 个数字）；
    - 仅有两个数字的文件（如变形结果）对路径规划无用，忽略并返回提示。

    require_stress=False 时（相似零件模板映射场景），允许只提供节点坐标文件，
    缺失的应力分量不再报错，由调用方决定是否走模板映射。

    返回：
        node_path    节点坐标文件路径
        stress_files 应力分量文件字典 {分量名: 路径}
        skipped      被忽略的辅助文件路径列表
    """
    stress_files = {}
    node_path = None
    skipped = []

    for path in paths:
        component = _component_from_filename(path)
        if component is not None:
            stress_files[component] = str(path)
            continue

        tokens = _first_numeric_line(path)
        if len(tokens) >= 4:
            if node_path is not None:
                raise ValueError(
                    f"识别到多个节点坐标文件：{node_path} 与 {path}"
                )
            node_path = str(path)
        elif len(tokens) >= 2:
            skipped.append(str(path))
        else:
            raise ValueError(f"无法识别文件：{path}")

    if node_path is None:
        raise ValueError(
            "未识别到节点坐标文件（需包含 Node,X,Y,Z 数据）。"
        )

    missing = [
        comp for comp in config.STRESS_COMPONENTS
        if comp not in stress_files
    ]
    if missing and require_stress:
        raise ValueError(
            "缺少应力分量文件：\n"
            + "\n".join(missing)
            + "\n\n需要：" + "、".join(config.STRESS_FILE_MAP)
        )

    return node_path, stress_files, skipped
