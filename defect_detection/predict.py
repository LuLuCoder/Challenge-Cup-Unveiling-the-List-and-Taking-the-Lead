"""批量推理：对图像输出检测框并导出检测明细 CSV。

用法：
    python predict.py --weights results/train/exp/weights/best.pt \
        --source data/images/test
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CLASS_NAMES, CONF_THRESHOLD, RESULTS_DIR, ensure_dirs  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 缺陷识别推理")
    parser.add_argument("--weights", type=str, default=None, required=True)
    parser.add_argument("--source", type=str, default="data/images/test")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--out", type=str, default=str(RESULTS_DIR / "predict"))
    args = parser.parse_args()

    ensure_dirs()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = Path(args.source)

    from ultralytics import YOLO

    model = YOLO(args.weights)
    images = sorted(
        p for p in src_dir.glob("*.*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not images:
        raise SystemExit(f"源目录没有图像：{src_dir}")

    det_rows = []
    for img_path in images:
        results = model.predict(
            str(img_path), conf=args.conf, imgsz=640, verbose=False
        )[0]
        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            for box, conf, cls_id in zip(xyxy, confs, cls_ids):
                x1, y1, x2, y2 = [float(v) for v in box]
                area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
                det_rows.append({
                    "image": img_path.name,
                    "class": CLASS_NAMES[cls_id],
                    "confidence": round(float(conf), 4),
                    "x1": round(x1, 1), "y1": round(y1, 1),
                    "x2": round(x2, 1), "y2": round(y2, 1),
                    "area_ratio": round(area_ratio, 4),
                })

        # 保存带检测框的可视化图（论文配图风格）
        annotated = results.plot()
        cv2.imwrite(str(out_dir / img_path.name), annotated)

    csv_path = out_dir / "detections.csv"
    if det_rows:
        import pandas as pd

        pd.DataFrame(det_rows).to_csv(
            csv_path, index=False, encoding="utf-8-sig"
        )
        print(f"检测到 {len(det_rows)} 个缺陷框，明细：{csv_path}")
    else:
        print("未检测到缺陷框。")
    print(f"可视化结果：{out_dir}")


if __name__ == "__main__":
    main()
