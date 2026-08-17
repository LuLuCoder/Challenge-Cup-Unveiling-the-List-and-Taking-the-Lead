"""全局配置：常量、字体、路径规划参数集中管理。"""

from pathlib import Path

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

# ---------- 相似零件模板库 / 路径映射 ----------
# 模板 = 一次完整的“真实仿真 + 路径规划”结果；相似的新零件可直接复用模板路径
TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "template_library"

# 形状签名：三个主轴坐标直方图 + 径向距离直方图的直方图分箱数
SIGNATURE_AXIS_BINS = 16
SIGNATURE_RADIAL_BINS = 24

# 相似度阈值（≥阈值判定为相似，直接映射；否则要求真实仿真数据）
DEFAULT_SIMILARITY_THRESHOLD = 0.80
SIMILARITY_MIN = 0.50
SIMILARITY_MAX = 0.99

# 路径/应力场映射参数
MAPPING_NEIGHBORS = 3            # 最近邻个数（反距离加权）
MAPPING_PATH_SCALE = 1.0         # 路径映射缩放系数（默认 1.0）
MAPPING_STRESS_SCALE = 1.0       # 应力映射缩放系数（默认 1.0）

# ---------- 大数据可视化 ----------
# 节点云绘图安全上限（几何/应力为静态图，13.8 万点全量绘制仅约 0.03s；
# 超过该上限才降采样，正常零件不会省略任何点）
PLOT_MAX_POINTS = 300000
