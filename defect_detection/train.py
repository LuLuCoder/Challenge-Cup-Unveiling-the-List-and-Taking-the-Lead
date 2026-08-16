"""YOLOv8 缺陷识别网络训练（5.4.3）。

用法：
    python train.py --weights yolov8n.pt --epochs 100 --imgsz 640

默认使用迁移学习（yolov8n.pt 预训练权重）。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    BATCH,
    DATASET_YAML,
    DEVICE,
    EPOCHS,
    IMG_SIZE,
    MODEL_WEIGHTS,
    RESULTS_DIR,
    SEED,
    ensure_dirs,
)


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 缺陷识别训练")
    parser.add_argument("--weights", type=str, default=MODEL_WEIGHTS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    ensure_dirs()
    if not DATASET_YAML.exists():
        raise SystemExit(
            "未找到 dataset.yaml。请先运行：\n"
            "  python dataset/synthetic.py   （合成数据演示）\n"
            "  或  python dataset/prepare.py --roboflow-zip xxx.zip"
        )

    from ultralytics import YOLO

    model = YOLO(args.weights) if args.weights else YOLO("yolov8n.yaml")
    print(f"开始训练：{args.weights or 'yolov8n.yaml（从头）'}，"
          f"epochs={args.epochs}, imgsz={args.imgsz}")

    model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=str(RESULTS_DIR / "train"),
        name="exp",
        seed=args.seed,
        plots=True,
    )
    print(f"训练完成，最优权重：{RESULTS_DIR / 'train' / 'exp' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
