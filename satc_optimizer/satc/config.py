"""全局配置：数据、参数水平、GPR 超参数、输出路径。"""

from pathlib import Path

import numpy as np


# ---------- 输出 ----------
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results"

LOG_FILENAME = "SATC_NS_results.log"
LOO_FILENAME = "SATC_NS_LOO_results.csv"
ALL_PREDICTIONS_FILENAME = "SATC_NS_all_predictions.csv"
PARETO_FILENAME = "SATC_NS_Pareto_results.csv"
SUMMARY_FILENAME = "SATC_NS_summary.csv"


def resolve_output_dir(output_dir=None):
    """返回输出目录（默认 results/），不存在则创建。"""
    path = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------- 参数 ----------
# A = Layer Thickness（层厚）
# B = First Layer Thickness（首层层厚）
# C = Nozzle Temperature（喷嘴温度）
# D = Printing Speed（打印速度）
LEVELS = {
    "A": np.array([0.15, 0.20, 0.25], dtype=float),
    "B": np.array([0.15, 0.20, 0.25], dtype=float),
    "C": np.array([180.0, 200.0, 220.0], dtype=float),
    "D": np.array([35.0, 40.0, 45.0], dtype=float),
}

PARAMETER_NAMES = ["A", "B", "C", "D"]
PARAMETER_LABELS = [
    "LayerThickness_mm",
    "FirstLayerThickness_mm",
    "NozzleTemperature_C",
    "PrintingSpeed_mm_s",
]
PARAMETER_UNITS = ["mm", "mm", "°C", "mm/s"]


# ---------- 论文真实实验数据（Table 3 / Table 4） ----------
X_REAL = np.array(
    [
        [0.15, 0.15, 180.0, 35.0],
        [0.15, 0.20, 200.0, 40.0],
        [0.15, 0.25, 220.0, 45.0],
        [0.20, 0.15, 200.0, 45.0],
        [0.20, 0.20, 220.0, 35.0],
        [0.20, 0.25, 180.0, 40.0],
        [0.25, 0.15, 220.0, 40.0],
        [0.25, 0.20, 180.0, 45.0],
        [0.25, 0.25, 200.0, 35.0],
    ],
    dtype=float,
)

Y_REAL = np.array(
    [
        [39.36, 29.21, 30.19],
        [56.25, 30.96, 49.83],
        [41.83, 32.58, 39.52],
        [54.46, 36.66, 12.95],
        [51.80, 44.05, 18.24],
        [58.92, 34.23, 10.05],
        [50.18, 46.87, 9.26],
        [61.37, 35.41, 13.73],
        [53.16, 49.19, 11.37],
    ],
    dtype=float,
)

OBJECTIVE_NAMES = ["ΔT Tensile", "ΔB Bending", "ΔS ILSS"]
DEFAULT_WEIGHTS = (1.0, 1.0, 1.0)


# ---------- 论文最优方案 ----------
PAPER_OPTIMAL = np.array([0.25, 0.20, 180.0, 45.0], dtype=float)
PAPER_OPTIMAL_NAME = "A3B2C1D3"


# ---------- 热约束 ----------
TEMPERATURE_MIN = 180.0
TEMPERATURE_MAX = 220.0


# ---------- GPR 超参数 ----------
GPR_LENGTH_SCALE = 1.0
GPR_NOISE_LEVEL = 1e-3
GPR_EXTRA_JITTER = 1e-8
GPR_JITTER_ATTEMPTS = 8
GPR_JITTER_BASE = 10.0
GPR_JITTER_START_EXP = -10
