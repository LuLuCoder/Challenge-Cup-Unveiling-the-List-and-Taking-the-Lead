"""在线缺陷识别（5.4.1 + 5.4.4）：逐层图像流 / 摄像头 + FPS。

用法：
    python online.py --weights results/train/exp/weights/best.pt \
        --source data/images/test            # 逐层图片流（模拟在线）
    python online.py --weights xxx.pt --camera 0   # 摄像头

输出：
    results/online/online_log.csv   每层判定记录
    results/online/online_metrics.txt  FPS 与统计
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    CLASS_NAMES,
    CONF_THRESHOLD,
    FPS_WARMUP,
    RESULTS_DIR,
    ensure_dirs,
)


def _class_name(cls_id):
    return CLASS_NAMES[int(cls_id)] if int(cls_id) < len(CLASS_NAMES) else "unknown"


def main():
    parser = argparse.ArgumentParser(description="在线缺陷识别")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, default=None,
                        help="逐层图像文件夹（模拟在线逐层拍照）")
    parser.add_argument("--camera", type=int, default=None,
                        help="摄像头索引，如 0")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--alert-conf", type=float, default=0.5,
                        help="超过该置信度判定为缺陷（报警）")
    parser.add_argument("--no-display", action="store_true",
                        help="无窗口环境（不调用 cv2.imshow）")
    args = parser.parse_args()

    if not args.source and args.camera is None:
        raise SystemExit("需要 --source 或 --camera。")

    ensure_dirs()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    out_dir = RESULTS_DIR / "online"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "online_log.csv"

    fps_values = []
    rows = []

    def process(frame_id, frame, tag=""):
        t0 = time.perf_counter()
        results = model.predict(
            frame, conf=args.conf, imgsz=640, verbose=False
        )[0]
        elapsed = time.perf_counter() - t0
        fps_values.append(1.0 / max(elapsed, 1e-9))

        defects = []
        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy()
            for conf, cls_id in zip(confs, cls_ids):
                defects.append({
                    "class": _class_name(cls_id),
                    "confidence": round(float(conf), 4),
                })

        has_defect = any(
            d["confidence"] >= args.alert_conf for d in defects
        )
        rows.append({
            "frame": frame_id,
            "tag": tag,
            "defect_count": len(defects),
            "defects": ";".join(
                f"{d['class']}@{d['confidence']:.2f}" for d in defects
            ),
            "verdict": "NG" if has_defect else "OK",
            "fps": round(1.0 / max(elapsed, 1e-9), 2),
        })

        # 实时显示（有窗口环境时）
        annotated = results.plot()
        status = f"Frame {frame_id} [{tag}] -> {'NG' if has_defect else 'OK'}"
        cv2.putText(
            annotated, status, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if has_defect else (0, 200, 0), 2,
        )
        if not args.no_display:
            cv2.imshow("online defect detection", annotated)
        return elapsed

    try:
        if args.source:
            images = sorted(
                p for p in Path(args.source).glob("*.*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            )
            for i, img_path in enumerate(images):
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue
                process(i, frame, tag=img_path.name)
                if not args.no_display and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        else:
            cap = cv2.VideoCapture(args.camera)
            if not cap.isOpened():
                raise SystemExit("无法打开摄像头")
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                process(i, frame, tag=f"cam{args.camera}")
                i += 1
                if not args.no_display and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cap.release()
    finally:
        cv2.destroyAllWindows()

    if fps_values:
        warm = fps_values[FPS_WARMUP:]
        mean_fps = float(sum(warm) / len(warm)) if warm else float(fps_values[0])
    else:
        mean_fps = 0.0

    import pandas as pd

    pd.DataFrame(rows).to_csv(log_path, index=False, encoding="utf-8-sig")
    (out_dir / "online_metrics.txt").write_text(
        f"平均在线检测速度：{mean_fps:.2f} FPS\n"
        f"样本数：{len(rows)}\n"
        f"判定为 NG 的帧数：{sum(1 for r in rows if r['verdict'] == 'NG')}\n",
        encoding="utf-8",
    )
    print(f"在线检测完成：{log_path}")
    print(f"平均 FPS：{mean_fps:.2f}")


if __name__ == "__main__":
    main()
