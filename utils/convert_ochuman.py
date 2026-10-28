#!/usr/bin/env python3
"""Convert all detection boxes in the original ochuman.json to YOLO labels.

Usage:
    python tools/convert_ochuman.py
    python tools/convert_ochuman.py --source-dir /path/to/OCHuman \
        --output-dir /path/to/ochuman_person --image-mode copy

The output contains images/train, labels/train and conversion_summary.json.
All images are used for training; no validation/test split is invented.
data/ochuman_person.yaml uses the existing COCO person validation set, just
like data/coco_person.yaml. Adjust its train path if --output-dir is changed.

Unlike the official COCO-format subsets, the original JSON also includes
people without keypoints or segmentation. Keep their boxes for detection.
Only Python's standard library is required; the source is never modified.
"""

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path


DEFAULT_SOURCE = Path(
    "/data/xiaoshuai/person/OpenDataLab___OCHuman/raw/OCHuman/OCHuman"
)
DEFAULT_OUTPUT = Path("/data/xiaoshuai/human_detection/ochuman_person")


def normalize_bbox(bbox, width, height):
    """Convert OCHuman's pixel xyxy box (not COCO xywh) to normalized xywh.

    Clip out-of-image coordinates; return None for a zero-area box.
    Malformed or nonfinite coordinates/dimensions are errors, not negatives.
    """
    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError("Image dimensions must be finite and positive")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("A bounding box must contain four xyxy coordinates")
    coordinates = tuple(float(value) for value in bbox)
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError("Bounding box coordinates must be finite")
    x1, y1, x2, y2 = coordinates
    x1, x2 = (min(width, max(0.0, x)) for x in (x1, x2))
    y1, y2 = (min(height, max(0.0, y)) for y in (y1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (
        (x1 + x2) / (2 * width),
        (y1 + y2) / (2 * height),
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def convert_dataset(source_dir, output_dir, image_mode="symlink"):
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).absolute()
    output_resolved = output_dir.resolve()
    if image_mode not in ("symlink", "copy"):
        raise ValueError("image_mode must be symlink or copy")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Refusing to overwrite existing output: {}".format(output_dir))
    if (source_dir == output_resolved or source_dir in output_resolved.parents
            or output_resolved in source_dir.parents):
        raise ValueError("Source and output directories must not overlap")

    annotation_path = source_dir / "ochuman.json"
    with annotation_path.open(encoding="utf-8") as stream:
        images = json.load(stream)["images"]
    if not images:
        raise ValueError("No image records found in ochuman.json")

    # Validate every record before creating output, to catch missing inputs.
    records = []
    label_paths = set()
    counts = Counter(images=0, source_boxes=0, boxes=0, empty_labels=0,
                     skipped_degenerate_boxes=0, skipped_duplicate_boxes=0)
    for image in images:
        relative_path = Path(image["file_name"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("Unsafe image filename: {}".format(relative_path))
        label_path = relative_path.with_suffix(".txt")
        if label_path in label_paths:
            raise ValueError("Duplicate label filename: {}".format(label_path))
        label_paths.add(label_path)
        source_image = source_dir / "images" / relative_path
        if not source_image.is_file():
            raise FileNotFoundError("Image not found: {}".format(source_image))
        width, height = float(image["width"]), float(image["height"])
        if not all(math.isfinite(v) and v > 0 for v in (width, height)):
            raise ValueError("Invalid image dimensions: {}".format(relative_path))
        lines, seen = [], set()
        for annotation in image["annotations"]:
            counts["source_boxes"] += 1
            box = normalize_bbox(annotation["bbox"], width, height)
            if box is None:
                counts["skipped_degenerate_boxes"] += 1
                continue
            line = "0 " + " ".join("{:.10f}".format(value) for value in box)
            if line in seen:
                counts["skipped_duplicate_boxes"] += 1
                continue
            seen.add(line)
            lines.append(line)
        records.append((relative_path, source_image, lines))
        counts["images"] += 1
        counts["boxes"] += len(lines)
        counts["empty_labels"] += not lines

    output_dir.mkdir(parents=True, exist_ok=False)
    for relative_path, source_image, lines in records:
        output_image = output_dir / "images" / "train" / relative_path
        output_label = output_dir / "labels" / "train" / relative_path.with_suffix(".txt")
        output_image.parent.mkdir(parents=True, exist_ok=True)
        output_label.parent.mkdir(parents=True, exist_ok=True)
        if image_mode == "symlink":
            output_image.symlink_to(source_image)
        else:
            shutil.copy2(str(source_image), str(output_image))
        with output_label.open("x", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + ("\n" if lines else ""))

    summary = {
        "source_annotations": str(annotation_path),
        "output_dir": str(output_dir),
        "image_mode": image_mode,
        "source_bbox_format": "xyxy",
        "label_format": "class_id center_x center_y width height (normalized)",
        "names": ["person"],
        "split": "All source images used for training; validation is external.",
        **dict(counts),
    }
    with (output_dir / "conversion_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-mode", choices=("symlink", "copy"), default="symlink",
                        help="Symlink images by default to avoid duplicating source data")
    args = parser.parse_args()
    try:
        summary = convert_dataset(args.source_dir, args.output_dir, args.image_mode)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, "Conversion failed: {}\n".format(error))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
