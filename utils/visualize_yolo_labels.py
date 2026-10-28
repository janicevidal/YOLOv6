#!/usr/bin/env python3
"""Save YOLO detection ground-truth overlays without a GUI or a model.

Examples (run from the YOLOv6 project directory):
    python tools/visualize_yolo_labels.py
    python tools/visualize_yolo_labels.py --image 003799.jpg
    python tools/visualize_yolo_labels.py --num-images 0  # all images

Reads class_id center_x center_y width height, with normalized coordinates.
Image/label paths must follow the images/<split> and labels/<split> layout.
Relative paths inside the YAML use the working directory, as in YOLOv6.
Outputs annotated images and overview.jpg (up to 16 images). Existing output
directories are preserved: a new numbered sibling is used on repeated runs.
Requires opencv-python, numpy and PyYAML; the input dataset is never modified.
"""

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# BGR colors; color by instance, so overlapping people remain distinguishable.
COLORS = [(50, 220, 50), (40, 150, 255), (255, 130, 40), (200, 70, 230),
          (30, 220, 220), (220, 200, 50), (70, 70, 255), (200, 130, 255)]


def label_path_for(image_path):
    """Replace the last /images/ component, without resolving image symlinks."""
    parts = list(Path(image_path).parts)
    if "images" not in parts:
        raise ValueError("Image path must contain an images directory: {}".format(image_path))
    index = len(parts) - 1 - parts[::-1].index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_labels(label_path, num_classes):
    labels = []
    with Path(label_path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                values = [float(value) for value in line.split()]
                if len(values) != 5 or not all(math.isfinite(v) for v in values):
                    raise ValueError("expected five finite numbers")
                cls, cx, cy, width, height = values
                if not cls.is_integer() or not 0 <= cls < num_classes:
                    raise ValueError("class ID is outside the configured classes")
                if not all(0 <= v <= 1 for v in values[1:]) or width <= 0 or height <= 0:
                    raise ValueError("expected normalized coordinates and positive box size")
                if (cx - width / 2 < -1e-6 or cy - height / 2 < -1e-6
                        or cx + width / 2 > 1 + 1e-6 or cy + height / 2 > 1 + 1e-6):
                    raise ValueError("bounding box extends outside the image")
                labels.append((int(cls), cx, cy, width, height))
            except ValueError as error:
                raise ValueError("{}:{}: {}".format(label_path, line_number, error)) from error
    return labels


def pixel_box(label, width, height):
    _, cx, cy, box_width, box_height = label
    corners = ((cx - box_width / 2) * width, (cy - box_height / 2) * height,
               (cx + box_width / 2) * width, (cy + box_height / 2) * height)
    return tuple(max(0, min(limit - 1, int(round(value))))
                 for value, limit in zip(corners, (width, height, width, height)))


def draw_labels(image, labels, names):
    result = image.copy()
    height, width = result.shape[:2]
    thickness = max(2, round(min(height, width) / 400))
    for index, label in enumerate(labels):
        color = COLORS[index % len(COLORS)]
        x1, y1, x2, y2 = pixel_box(label, width, height)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        text = "{} #{}".format(names[label[0]], index + 1)
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        text_x = max(0, min(x1, width - tw - 6))
        text_y = max(th + 6, y1 - 4)
        cv2.rectangle(result, (text_x, text_y - th - 5),
                      (text_x + tw + 6, text_y + baseline), color, -1)
        cv2.putText(result, text, (text_x + 3, text_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return result


def make_tile(image, title, size):
    tile = np.full((size + 28, size, 3), 240, dtype=np.uint8)
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    small = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    x, y = (size - small.shape[1]) // 2, (size - small.shape[0]) // 2 + 28
    tile[y:y + small.shape[0], x:x + small.shape[1]] = small
    cv2.putText(tile, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return tile


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError("Failed to save image: {}".format(path))


def new_output_dir(path):
    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    for index in range(1, 10000):
        candidate = path if index == 1 else path.with_name("{}_{}".format(path.name, index))
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError("Too many existing output directories: {}".format(path))


def visualize(data_path, split, output_dir, num_images=16, seed=42,
              image_name=None, columns=4, tile_size=320):
    if num_images < 0 or columns < 1 or tile_size < 64:
        raise ValueError("num_images >= 0, columns >= 1 and tile_size >= 64 are required")
    with Path(data_path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    names = data["names"]
    if isinstance(names, dict):
        names = [names[index] for index in range(len(names))]
    if not names or len(names) != data["nc"]:
        raise ValueError("YAML nc must match the class names")
    directories = data[split] if isinstance(data[split], list) else [data[split]]
    images = set()
    for directory in directories:
        directory = Path(directory).absolute()
        if not directory.is_dir():
            raise NotADirectoryError(str(directory))
        label_directory = label_path_for(directory / "placeholder.jpg").parent
        for protected in (directory.resolve(), label_directory.resolve()):
            resolved_output = Path(output_dir).resolve()
            if (resolved_output == protected or protected in resolved_output.parents
                    or resolved_output in protected.parents):
                raise ValueError("Output directory must not overlap input images or labels")
        images.update(path for path in directory.rglob("*")
                      if path.suffix.lower() in IMAGE_FORMATS and path.is_file())
    images = sorted(images)
    if image_name:
        images = [path for path in images if path.name == image_name]
        if len(images) != 1:
            raise ValueError("--image must match exactly one filename; matched {}".format(len(images)))
    elif num_images and len(images) > num_images:
        images = sorted(random.Random(seed).sample(images, num_images))
    if not images:
        raise ValueError("No images found for split {}".format(split))
    output = new_output_dir(output_dir)
    tiles, records = [], []
    for index, image_path in enumerate(images, 1):
        label_path = label_path_for(image_path)
        labels = read_labels(label_path, len(names))
        image = cv2.imread(str(image_path))
        if image is None:
            raise OSError("Cannot decode image: {}".format(image_path))
        annotated = draw_labels(image, labels, names)
        filename = "{:05d}_{}.jpg".format(index, image_path.stem)
        save_image(output / filename, annotated)
        if len(tiles) < 16:
            tiles.append(make_tile(annotated, "{} | {} boxes".format(image_path.name, len(labels)), tile_size))
        records.append({"image": str(image_path), "label": str(label_path),
                        "output": filename, "boxes": len(labels)})
        if index % 100 == 0:
            print("Rendered {}/{} images".format(index, len(images)))
    columns = min(columns, len(tiles))
    rows = math.ceil(len(tiles) / columns)
    overview = np.full((rows * (tile_size + 28), columns * tile_size, 3), 240, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        x, y = (index % columns) * tile_size, (index // columns) * (tile_size + 28)
        overview[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
    save_image(output / "overview.jpg", overview)
    summary = {"data": str(Path(data_path).absolute()), "split": split, "seed": seed,
               "images": len(records), "boxes": sum(record["boxes"] for record in records),
               "output_dir": str(output), "overview_images": len(tiles), "files": records}
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/ochuman_person.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/vis/ochuman_person")
    parser.add_argument("--num-images", type=int, default=16, help="Random sample size; 0 means all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image", help="Visualize one exact filename, e.g. 003799.jpg")
    parser.add_argument("--columns", type=int, default=4, help="Overview grid columns")
    parser.add_argument("--tile-size", type=int, default=320, help="Overview tile width in pixels")
    args = parser.parse_args()
    try:
        summary = visualize(args.data, args.split, args.output_dir, args.num_images,
                            args.seed, args.image, args.columns, args.tile_size)
    except (OSError, ValueError, KeyError, TypeError, cv2.error) as error:
        parser.exit(1, "Visualization failed: {}\n".format(error))
    print("Saved {} images / {} boxes to {}".format(summary["images"], summary["boxes"], summary["output_dir"]))
    print("Overview: {}".format(Path(summary["output_dir"]) / "overview.jpg"))


if __name__ == "__main__":
    main()
