"""5.4 在线缺陷识别方法 —— 全局配置。"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# ---------- 缺陷类别（cCFRP 3D 打印） ----------
# 正常（无缺陷）情况不产生检测框，不作为类别
CLASS_NAMES = ["misalignment", "gap", "buildup"]
CLASS_CN = {
    "misalignment": "纤维错位",
    "gap": "间隙",
    "buildup": "堆积",
}

# ---------- 数据集 ----------
DATASET_YAML = DATA_DIR / "dataset.yaml"
IMG_SIZE = 640
SEED = 42

# ---------- 训练 ----------
MODEL_WEIGHTS = "yolov8n.pt"   # 迁移学习基座；None = 从头训练
EPOCHS = 100
BATCH = 16
DEVICE = ""                    # 空 = ultralytics 自动选择

# ---------- 在线检测 ----------
CONF_THRESHOLD = 0.35
FPS_WARMUP = 10                # 在线测速前预热帧数

# ---------- 论文参考（用于结果对照） ----------
# Zubayer et al., 2024, Composites Part C: Open Access
# DOI: 10.1016/j.jcomc.2024.100451
PAPER_REFERENCE = {
    "authors": "Zubayer et al.",
    "year": 2024,
    "title": (
        "Enhancing additive manufacturing precision: Intelligent "
        "inspection and optimization for defect-free continuous carbon "
        "fiber-reinforced polymer"
    ),
    "journal": "Composites Part C: Open Access",
    "doi": "10.1016/j.jcomc.2024.100451",
    "model": "YOLOv8",
    "metrics": {
        "mAP50": [0.88, 0.90, 0.94],
        "misalignment_accuracy": 0.94,
    },
    "notes": (
        "论文对每个参数组采集 50 张逐层图像，错位缺陷识别准确率约 94%，"
        "mAP50 报告值约 0.88/0.90/0.94（以论文原文为准）。"
    ),
}


def ensure_dirs():
    for d in (DATA_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
