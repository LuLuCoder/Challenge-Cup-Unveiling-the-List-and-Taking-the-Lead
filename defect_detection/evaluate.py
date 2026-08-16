"""测试集评估：Precision / Recall / F1 / mAP + 混淆矩阵 + 论文对照表（5.4.5）。

用法：
    python evaluate.py --weights results/train/exp/weights/best.pt
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    CLASS_NAMES,
    CONF_THRESHOLD,
    DATASET_YAML,
    PAPER_REFERENCE,
    RESULTS_DIR,
    ensure_dirs,
)


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 缺陷识别评估")
    parser.add_argument(
        "--weights", type=str,
        default=str(RESULTS_DIR / "train" / "exp" / "weights" / "best.pt"),
    )
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    args = parser.parse_args()

    ensure_dirs()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    metrics = model.val(
        data=str(DATASET_YAML),
        split="test",
        conf=args.conf,
        plots=True,
        project=str(RESULTS_DIR / "eval"),
        name="test",
    )

    box = metrics.box
    ap50 = box.ap50  # 每类 AP@0.5
    if ap50 is None or len(ap50) == 0:
        raise SystemExit(
            "测试集没有检测结果。请检查 --weights 与数据集是否匹配，"
            "或运行 python dataset/prepare.py --check"
        )
    ap_class = box.ap_class_index.astype(int)
    rows = []
    for cls_idx, ap in zip(ap_class, ap50):
        rows.append({
            "类别": CLASS_NAMES[cls_idx],
            "中文": CLASS_NAMES[cls_idx],
            "AP@0.5": float(ap),
        })

    summary = pd.DataFrame(rows)
    summary.loc[len(summary)] = {
        "类别": "全部",
        "中文": "全部",
        "AP@0.5": float(box.map50),
    }
    summary["中文"] = summary["类别"].map(
        {"misalignment": "纤维错位", "gap": "间隙", "buildup": "堆积"}
    ).fillna("全部")

    csv_path = RESULTS_DIR / "eval" / "metrics_summary.csv"
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))

    # ---------- 5.4.5 论文结果对照表 ----------
    paper_metrics = PAPER_REFERENCE["metrics"]
    paper_map50 = "/".join(f"{v:.2f}" for v in paper_metrics["mAP50"])
    md_path = RESULTS_DIR / "eval" / "paper_comparison.md"
    lines = [
        "## 5.4.5 实验验证：缺陷识别指标对照",
        "",
        "| 指标 | 本方法 | 论文参考（Zubayer et al., 2024） |",
        "|---|---|---|",
        f"| mAP50 | {float(box.map50):.4f} | {paper_map50}（三类） |",
        f"| 错位识别准确率 | - | {paper_metrics['misalignment_accuracy']:.2f} |",
        f"| Precision | {float(box.mp):.4f} | - |",
        f"| Recall | {float(box.mr):.4f} | - |",
        f"| mAP50-95 | {float(box.map):.4f} | - |",
        "",
        f"模型：YOLOv8；数据：{DATASET_YAML}",
        f"论文：{PAPER_REFERENCE['authors']}, {PAPER_REFERENCE['year']}, "
        f"{PAPER_REFERENCE['title']}",
        f"DOI：{PAPER_REFERENCE['doi']}",
        "",
        PAPER_REFERENCE["notes"],
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"对照表已生成：{md_path}")


if __name__ == "__main__":
    main()
