"""生成合成缺陷数据集（离线演示用）。

在没有真实数据时，生成模拟 cCFRP 打印层的图像：
    misalignment  纤维错位（某条带水平偏移）
    gap           间隙（层间断开条带）
    buildup       堆积（局部材料堆积块）
    normal        正常层（无缺陷，不生成标签）

输出 YOLO 格式：
    data/images/{train,val,test}/xxx.jpg
    data/labels/{train,val,test}/xxx.txt   （class cx cy w h，归一化）

注意：合成数据仅用于流程演示，论文验证请使用真实数据
（见 README 与 dataset/prepare.py）。
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))

from config import CLASS_NAMES, DATA_DIR, IMG_SIZE, SEED  # noqa: E402


def _draw_layer_base(img, rng):
    """深色基底 + 平行纤维条纹，模拟打印层表面。"""
    img[:] = (30, 30, 34)
    n_stripes = rng.randint(18, 26)
    for i in range(n_stripes):
        y = int(i * IMG_SIZE / n_stripes)
        cv2.line(img, (0, y), (IMG_SIZE, y), (45, 46, 52), 2)


def _draw_misalignment(img, rng):
    """纤维错位：把某条带水平平移。"""
    band_h = rng.randint(12, 24)
    y0 = rng.randint(40, IMG_SIZE - 40 - band_h)
    shift = rng.randint(30, 70)

    src = img[y0:y0 + band_h, :].copy()
    img[y0:y0 + band_h, :] = (24, 24, 27)
    if shift < IMG_SIZE:
        img[y0:y0 + band_h, shift:] = src[:, :IMG_SIZE - shift]
    return 0, y0, IMG_SIZE, band_h


def _draw_gap(img, rng):
    """间隙：一条横向空隙。"""
    gap_h = rng.randint(5, 12)
    y0 = rng.randint(50, IMG_SIZE - 50 - gap_h)
    img[y0:y0 + gap_h, :] = (12, 12, 14)
    return 0, y0, IMG_SIZE, gap_h


def _draw_buildup(img, rng):
    """堆积：局部圆形凸起。"""
    cx = rng.randint(60, IMG_SIZE - 60)
    cy = rng.randint(60, IMG_SIZE - 60)
    r = rng.randint(15, 32)
    cv2.circle(img, (cx, cy), r, (68, 70, 80), -1)
    cv2.circle(img, (cx, cy), r, (112, 116, 130), 2)
    cv2.circle(img, (cx + r // 3, cy - r // 3), r // 3, (96, 100, 114), -1)
    return cx - r, cy - r, 2 * r, 2 * r


DRAWERS = {
    "misalignment": _draw_misalignment,
    "gap": _draw_gap,
    "buildup": _draw_buildup,
}


def generate_one(rng, defect):
    """生成一张图，返回 (image, label_line 或 None)。"""
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    _draw_layer_base(img, rng)

    if defect is None:
        return img, None

    x1, y1, w, h = DRAWERS[defect](img, rng)
    cx = (x1 + w / 2.0) / IMG_SIZE
    cy = (y1 + h / 2.0) / IMG_SIZE
    nw = w / IMG_SIZE
    nh = h / IMG_SIZE
    label = f"{CLASS_NAMES.index(defect)} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
    return img, label


def main():
    parser = argparse.ArgumentParser(description="生成合成缺陷数据集")
    parser.add_argument("--n-train", type=int, default=120)
    parser.add_argument("--n-val", type=int, default=30)
    parser.add_argument("--n-test", type=int, default=30)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    splits = [("train", args.n_train), ("val", args.n_val), ("test", args.n_test)]

    for split, count in splits:
        img_dir = DATA_DIR / "images" / split
        lbl_dir = DATA_DIR / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            # 50% 正常层，50% 随机缺陷，保证类别均衡
            defect = None if rng.random() < 0.5 else rng.choice(CLASS_NAMES)
            img, label = generate_one(rng, defect)
            name = f"{split}_{i:04d}"
            cv2.imwrite(str(img_dir / f"{name}.jpg"), img)
            if label is not None:
                (lbl_dir / f"{name}.txt").write_text(label + "\n", encoding="utf-8")

        n_labels = len(list(lbl_dir.glob("*.txt")))
        print(f"{split}: {count} 张，其中含缺陷标注 {n_labels} 张")

    print("合成数据集生成完成。下一步：python train.py")


if __name__ == "__main__":
    main()
