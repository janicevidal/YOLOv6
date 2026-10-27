import os
import json
import shutil
import argparse
from collections import defaultdict
from tqdm import tqdm

# Objects365 中 Person 的类别 ID
PERSON_ID = 1
# 输出到 YOLO 时的类别 ID（单类固定为 0）
YOLO_CLASS_ID = 0


def convert_person_split(anno_file, imgs_dir, out_img_dir, out_lbl_dir,
                         keep_crowd=False):
    """
    从 Objects365 单个划分（train/val）中提取 person 类，转 YOLO 格式。

    :param anno_file:   train.json / val.json
    :param imgs_dir:    原始图片目录
    :param out_img_dir: 输出图片目录（.../images/train 或 val）
    :param out_lbl_dir: 输出标签目录（.../labels/train 或 val）
    :param keep_crowd:  是否保留 iscrowd=1 的标注（默认丢弃）
    """
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    with open(anno_file, 'r') as f:
        data = json.load(f)

    # 1) 建立「磁盘上实际存在」的图片索引：id -> image_info
    existing = set(os.listdir(imgs_dir))
    img_map = {
        img["id"]: img
        for img in data["images"]
        if img["file_name"] in existing
    }

    # 2) 收集 person 标注：id -> [bbox, ...]
    img_2_bboxes = defaultdict(list)
    skipped_crowd = 0
    for anno in data["annotations"]:
        if anno["image_id"] not in img_map:
            continue
        if anno["category_id"] != PERSON_ID:
            continue
        if not keep_crowd and anno.get("iscrowd", 0) == 1:
            skipped_crowd += 1
            continue
        img_2_bboxes[anno["image_id"]].append(anno["bbox"])

    print(f"[{os.path.basename(anno_file)}] "
          f"images={len(data['images'])}, "
          f"person_images={len(img_2_bboxes)}, "
          f"skipped_crowd={skipped_crowd}")

    # 3) 逐张图片转换并写出
    n_img = n_box = 0
    for img_id, bboxes in tqdm(img_2_bboxes.items(), desc="converting"):
        info = img_map[img_id]
        w_img, h_img = info["width"], info["height"]
        if w_img <= 0 or h_img <= 0:
            continue

        lines = []
        for x, y, w, h in bboxes:
            if w <= 0 or h <= 0:
                continue
            cx = min(max(x + w / 2, 0) / w_img, 1.0)
            cy = min(max(y + h / 2, 0) / h_img, 1.0)
            nw = min(max(w / w_img, 0), 1.0)
            nh = min(max(h / h_img, 0), 1.0)
            if nw <= 0 or nh <= 0:
                continue
            lines.append(f"{YOLO_CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if not lines:  # 清洗后该图无有效框，跳过
            continue

        # 复制图片
        shutil.copy(
            os.path.join(imgs_dir, info["file_name"]),
            os.path.join(out_img_dir, info["file_name"]),
        )
        # 写标签（用 splitext 处理 .jpg/.jpeg/.png）
        stem = os.path.splitext(info["file_name"])[0]
        with open(os.path.join(out_lbl_dir, stem + ".txt"), 'w') as f:
            f.write("\n".join(lines) + "\n")

        n_img += 1
        n_box += len(lines)

    print(f"done: images={n_img}, boxes={n_box} -> "
          f"{out_img_dir} | {out_lbl_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract person class from Objects365 -> YOLO format")
    parser.add_argument("--anno",  required=True,
                        help="Objects365 标注 json，如 train.json / val.json")
    parser.add_argument("--imgs",  required=True,
                        help="Objects365 图片目录")
    parser.add_argument("--out",   required=True,
                        help="输出根目录（自动生成 images/labels 子目录）")
    parser.add_argument("--split", default="train",
                        choices=["train", "val"],
                        help="数据划分，用于组织输出目录（默认 train）")
    parser.add_argument("--keep-crowd", action="store_true",
                        help="保留 iscrowd=1 的标注（默认丢弃）")
    args = parser.parse_args()

    convert_person_split(
        anno_file   = args.anno,
        imgs_dir    = args.imgs,
        out_img_dir = os.path.join(args.out, "images", args.split),
        out_lbl_dir = os.path.join(args.out, "labels", args.split),
        keep_crowd  = args.keep_crowd,
    )


if __name__ == "__main__":
    main()