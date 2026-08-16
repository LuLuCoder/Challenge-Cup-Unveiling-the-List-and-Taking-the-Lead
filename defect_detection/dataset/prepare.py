"""真实数据集准备：Roboflow 导出包 / COCO JSON → YOLO 格式 + 数据集划分。

用法示例：
    python dataset/prepare.py --roboflow-zip path/to/roboflow_export.zip
    python dataset/prepare.py --coco-json path/to/annotations.json --image-dir path/to/images
    python dataset/prepare.py --check
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from config import CLASS_NAMES, DATA_DIR, DATASET_YAML, SEED  # noqa: E402


SPLITS = ["train", "val", "test"]


def write_dataset_yaml():
    """写出 ultralytics 需要的 dataset.yaml。"""
    data = {
        "path": str(DATA_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    DATASET_YAML.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"dataset.yaml 已写入：{DATASET_YAML}")


def _collect_yolo_pairs(images_root, labels_root):
    """收集 images/labels 目录下匹配的图像-标签对。"""
    pairs = []
    for img in sorted(images_root.glob("*.*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        lbl = labels_root / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))
        else:
            print(f"警告：{img.name} 没有对应标签，已跳过")
    return pairs


def _copy_split(src_images, src_labels, split, names):
    """把匹配对复制进 YOLO 标准目录。"""
    img_dir = DATA_DIR / "images" / split
    lbl_dir = DATA_DIR / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    pairs = _collect_yolo_pairs(src_images, src_labels)
    rng = __import__("random").Random(SEED)
    rng.shuffle(pairs)
    for img, lbl in pairs[:names]:
        shutil.copy2(img, img_dir / img.name)
        shutil.copy2(lbl, lbl_dir / lbl.name)
    print(f"{split}: 复制 {min(names, len(pairs))} 对")


def from_roboflow_zip(zip_path):
    """解压 Roboflow YOLO 导出包并规整为标准目录。"""
    zip_path = Path(zip_path)
    tmp = DATA_DIR / "_roboflow_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)

    # Roboflow YOLO 导出结构：train/valid/test 下各含 images/ 与 labels/
    alias = {
        "train": ["train"],
        "val": ["valid", "val"],
        "test": ["test"],
    }
    handled = 0
    for split in SPLITS:
        found = False
        for cand in alias[split]:
            imgs = tmp / cand / "images"
            lbls = tmp / cand / "labels"
            if imgs.exists() and lbls.exists():
                _copy_split(imgs, lbls, split, names=10 ** 6)
                handled += 1
                found = True
                break
        if not found:
            print(f"警告：Roboflow 包中未找到 {split} 划分")

    shutil.rmtree(tmp)
    if handled == 0:
        raise RuntimeError(
            "Roboflow 包结构无法识别，请确认导出格式为 YOLOv8"
        )
    write_dataset_yaml()


def from_coco_json(coco_json, image_dir, val_ratio=0.15, test_ratio=0.15):
    """把 COCO JSON 标注转换为 YOLO txt，并按比例划分。"""
    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {im["id"]: im for im in coco["images"]}
    # 类别名 -> id
    cat_names = {cat["id"]: cat["name"] for cat in coco["categories"]}
    cat_to_idx = {
        name: i for i, name in enumerate(CLASS_NAMES)
        if name in CLASS_NAMES
    }

    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    items = []
    for image_id, anns in anns_by_image.items():
        im = images[image_id]
        img_path = Path(image_dir) / im["file_name"]
        if not img_path.exists():
            print(f"跳过缺失图像：{img_path}")
            continue
        lines = []
        for ann in anns:
            cat_name = cat_names.get(ann["category_id"], "")
            cls_idx = cat_to_idx.get(cat_name)
            if cls_idx is None:
                continue
            x, y, w, h = ann["bbox"]
            cx = (x + w / 2) / im["width"]
            cy = (y + h / 2) / im["height"]
            nw = w / im["width"]
            nh = h / im["height"]
            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        if lines:
            items.append((img_path, "\n".join(lines) + "\n"))

    rng = __import__("random").Random(SEED)
    rng.shuffle(items)
    n_val = int(len(items) * val_ratio)
    n_test = int(len(items) * test_ratio)
    assign = (
        [("test", n_test)]
        + [("val", n_val)]
        + [("train", len(items) - n_val - n_test)]
    )

    cursor = 0
    for split, count in assign:
        img_dir = DATA_DIR / "images" / split
        lbl_dir = DATA_DIR / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_path, label in items[cursor:cursor + count]:
            shutil.copy2(img_path, img_dir / img_path.name)
            (lbl_dir / f"{img_path.stem}.txt").write_text(label, encoding="utf-8")
        cursor += count
        print(f"{split}: {count} 张")

    write_dataset_yaml()


def check_dataset():
    """检查数据集完整性并输出类别统计。"""
    from collections import Counter

    write_dataset_yaml()
    counts = Counter()
    for split in SPLITS:
        lbl_dir = DATA_DIR / "labels" / split
        n = 0
        if lbl_dir.exists():
            for lbl in lbl_dir.glob("*.txt"):
                n += 1
                lines = [l for l in lbl.read_text().splitlines() if l.strip()]
                for line in lines:
                    cls = int(line.split()[0])
                    counts[cls] += 1
        print(f"{split}: {n} 个标注")
    print("类别分布：", {CLASS_NAMES[k]: v for k, v in sorted(counts.items())})


def main():
    parser = argparse.ArgumentParser(description="准备 YOLO 格式数据集")
    parser.add_argument("--roboflow-zip", type=str, default=None)
    parser.add_argument("--coco-json", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        check_dataset()
    elif args.roboflow_zip:
        from_roboflow_zip(args.roboflow_zip)
    elif args.coco_json and args.image_dir:
        from_coco_json(args.coco_json, args.image_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
