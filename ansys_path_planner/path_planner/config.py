"""全局配置：常量、字体、路径规划参数集中管理。"""

# ---------- 应用 ----------
APP_TITLE = "ANSYS 仿真驱动路径规划系统"
APP_GEOMETRY = "1650x1000"

# ---------- Matplotlib 中文字体 ----------
MATPLOTLIB_FONTS = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]

# ---------- ANSYS 六应力分量文件映射 ----------
STRESS_FILE_MAP = {
    "X.txt": "SX",
    "Y.txt": "SY",
    "Z.txt": "SZ",
    "XY.txt": "SXY",
    "YZ.txt": "SYZ",
    "XZ.txt": "SXZ",
}

STRESS_COMPONENTS = ["SX", "SY", "SZ", "SXY", "SYZ", "SXZ"]

# ---------- 默认保存文件名 ----------
DEFAULT_MERGED_FILE = "ANSYS_merged_result.csv"
DEFAULT_PATH_FILE = "ANSYS_planned_path.csv"

# ---------- 路径规划参数 ----------
# 高优先级区域分位数
DEFAULT_PERCENTILE = 75.0
PERCENTILE_MIN = 50.0
PERCENTILE_MAX = 99.0

# 自动步长：占模型包围盒最大边长的比例（高应力 -> 小步长）
SPACING_MAX_RATIO = 0.045   # 低应力区步长上限（d_max）
SPACING_MIN_RATIO = 0.018   # 高应力区步长下限（d_min）

# 用户指定步长时：d_min = spacing，d_max = spacing * 倍数
SPACING_USER_MAX_MULTIPLIER = 4.0

# 应力 -> 步长映射的幂指数
SPACING_GAMMA = 0.75

# 候选节点搜索半径
SEARCH_RADIUS_SPACING_MULTIPLIER = 2.2
SEARCH_RADIUS_MIN_RATIO = 0.025

# 方向追踪：只保留与主方向夹角足够小的候选（点积下限）
DIRECTION_MIN_DOT = 0.35

# 候选节点打分权重
SCORE_DIRECTION_WEIGHT = 0.60
SCORE_DISTANCE_WEIGHT = 0.25
SCORE_STRESS_WEIGHT = 0.15

# 路径权重
PATH_WEIGHT_BASE = 0.25
PATH_WEIGHT_STRESS_FACTOR = 0.75

# ---------- 3D 打印分层切片参数 ----------
DEFAULT_LAYERS = 20
LAYER_MIN = 2
LAYER_MAX = 200
SLICE_AXIS = "Z"

# ---------- 空区（无节点区域）检测 ----------
VOID_CHECK_RADIUS_RATIO = 0.045        # 检测半径下限 = 特征尺寸 × 该比例
VOID_CHECK_SPACING_MULTIPLIER = 1.0    # 检测半径至少为典型节点间距 × 该倍数
VOID_SAMPLE_MIN = 3                    # 每段路径至少采样点数

# ---------- 路径密度等级配色 ----------
DENSITY_COLORS = ["#4C78A8", "#72B7B2", "#F2CF5B", "#F58518", "#E45756"]
DENSITY_LABELS = ["稀疏路径", "较稀疏", "中等密度", "较密集", "高密度路径"]
