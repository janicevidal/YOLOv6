#!/usr/bin/env python3
"""Convert Human-Parts COCO JSON annotations to YOLOv6 detection labels.

Usage (from the YOLOv6 project root):
    python utils/convert_human_parts.py
    # Optional: explicitly retain all three original classes in another directory.
    python utils/convert_human_parts.py --classes person face hand \
        --output-dir /data/xiaoshuai/human_detection/human_parts_3classes

Preserves the official train/val split and the source dataset. By default,
only person is retained as YOLO ID 0; face and hand boxes are filtered out.
Use --classes to explicitly select/remap other classes when needed.
Creates images/<split>, labels/<split>, dataset.yaml, conversion_summary.json,
and annotations/human_parts_val.json for evaluation with matching image/class IDs.
Images are symlinked by default; use --image-mode copy for a standalone copy.
Uses the supplied Json_Annos files: their image dimensions were read from
actual images, unlike the sometimes incorrect dimensions in the source XML.
Only Python's standard library is required. Existing output is not overwritten.
"""

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path


DEFAULT_SOURCE = Path("/data/xiaoshuai/body_keypoint/OpenDataLab___Human-Parts/raw")
DEFAULT_OUTPUT = Path("/data/xiaoshuai/human_detection/human_parts")
AVAILABLE_CLASSES = ("person", "face", "hand")
DEFAULT_CLASSES = ("person",)


def normalize_coco_box(bbox, width, height):
    """Convert pixel top-left xywh (not xyxy) to normalized center xywh."""
    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError("Image dimensions must be finite and positive")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("A COCO bounding box must have four xywh coordinates")
    x, y, box_width, box_height = (float(value) for value in bbox)
    if not all(math.isfinite(v) for v in (x, y, box_width, box_height)):
        raise ValueError("Bounding box coordinates must be finite")
    if box_width <= 0 or box_height <= 0:
        return None
    x1, y1 = min(width, max(0.0, x)), min(height, max(0.0, y))
    x2 = min(width, max(0.0, x + box_width))
    y2 = min(height, max(0.0, y + box_height))
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            (x2 - x1) / width, (y2 - y1) / height)


def prepare_split(source_dir, split, classes):
    annotation_path = source_dir / "Json_Annos" / ("privpersonpart_" + split + ".json")
    with annotation_path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    categories = {category["id"]: category["name"] for category in data["categories"]}
    if len(categories) != len(data["categories"]):
        raise ValueError("Duplicate category IDs in {}".format(annotation_path))
    if set(classes) - set(categories.values()):
        raise ValueError("Requested classes are missing from {}".format(annotation_path))
    class_ids = {name: index for index, name in enumerate(classes)}
    records, stems = {}, set()
    counts = Counter(images=0, source_boxes=len(data["annotations"]), boxes=0,
                     empty_labels=0, filtered_boxes=0, skipped_crowd_boxes=0,
                     skipped_degenerate_boxes=0, skipped_duplicate_boxes=0, clipped_boxes=0)
    per_class = Counter({name: 0 for name in classes})
    for image in data["images"]:
        relative_path = Path(image["file_name"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("Unsafe filename: {}".format(relative_path))
        # YOLOv6's non-COCO evaluator uses the filename stem as the image ID.
        if relative_path.stem in stems or image["id"] in records:
            raise ValueError("Duplicate image ID or filename stem in {}".format(split))
        stems.add(relative_path.stem)
        source_image = source_dir / "Images" / relative_path
        if not source_image.is_file():
            raise FileNotFoundError("Image not found: {}".format(source_image))
        width, height = float(image["width"]), float(image["height"])
        if not all(math.isfinite(v) and v > 0 for v in (width, height)):
            raise ValueError("Invalid dimensions: {}".format(relative_path))
        records[image["id"]] = {"relative_path": relative_path, "source_image": source_image,
                                "width": width, "height": height, "lines": [], "seen": set()}
    if not records:
        raise ValueError("No images in {}".format(annotation_path))
    for annotation in data["annotations"]:
        if annotation["image_id"] not in records or annotation["category_id"] not in categories:
            raise ValueError("Annotation references an unknown image or category in {}".format(split))
        name = categories[annotation["category_id"]]
        if name not in class_ids:
            counts["filtered_boxes"] += 1
            continue
        if annotation.get("iscrowd", 0):
            counts["skipped_crowd_boxes"] += 1
            continue
        record = records[annotation["image_id"]]
        box = normalize_coco_box(annotation["bbox"], record["width"], record["height"])
        if box is None:
            counts["skipped_degenerate_boxes"] += 1
            continue
        line = str(class_ids[name]) + " " + " ".join("{:.10f}".format(v) for v in box)
        if line in record["seen"]:
            counts["skipped_duplicate_boxes"] += 1
            continue
        x, y, width, height = (float(value) for value in annotation["bbox"])
        counts["clipped_boxes"] += (x < 0 or y < 0 or x + width > record["width"]
                                    or y + height > record["height"])
        record["seen"].add(line)
        record["lines"].append(line)
        counts["boxes"] += 1
        per_class[name] += 1
    counts["images"] = len(records)
    counts["empty_labels"] = sum(not record["lines"] for record in records.values())
    summary = {"source_annotations": str(annotation_path), **dict(counts),
               "per_class_boxes": dict(per_class)}
    return list(records.values()), summary


def write_validation_annotations(output_dir, records, classes):
    """Write evaluator-compatible COCO JSON without changing the source JSON.

    Non-COCO YOLOv6 predictions use string filename stems and zero-based classes.
    Ground-truth annotation IDs must start at 1: COCOeval uses 0 for no match.
    Keep a separate filename, since YOLOv6 also auto-generates instances_val.json.
    """
    annotation_path = Path(output_dir) / "annotations" / "human_parts_val.json"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = {"info": {"description": "Human-Parts validation detection annotations"},
               "images": [], "annotations": [],
               "categories": [{"id": index, "name": name, "supercategory": name}
                              for index, name in enumerate(classes)]}
    for record in records:
        image_id = record["relative_path"].stem
        width, height = record["width"], record["height"]
        dataset["images"].append({"id": image_id, "file_name": str(record["relative_path"]),
                                  "width": int(width), "height": int(height)})
        for line in record["lines"]:
            cls, cx, cy, bw, bh = (float(value) for value in line.split())
            bbox = [(cx - bw / 2) * width, (cy - bh / 2) * height, bw * width, bh * height]
            dataset["annotations"].append({"id": len(dataset["annotations"]) + 1,
                                           "image_id": image_id, "category_id": int(cls),
                                           "bbox": bbox, "area": bbox[2] * bbox[3],
                                           "iscrowd": 0, "segmentation": []})
    with annotation_path.open("x", encoding="utf-8") as stream:
        json.dump(dataset, stream)
    return annotation_path


def convert_dataset(source_dir, output_dir, classes=DEFAULT_CLASSES, image_mode="symlink"):
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).absolute()
    classes = tuple(classes)
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("Select at least one class, without duplicates")
    if image_mode not in ("symlink", "copy"):
        raise ValueError("image_mode must be symlink or copy")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Refusing to overwrite existing output: {}".format(output_dir))
    output_dir = output_dir.resolve()
    if (source_dir == output_dir or source_dir in output_dir.parents
            or output_dir in source_dir.parents):
        raise ValueError("Source and output directories must not overlap")

    # Validate both splits before creating any output files.
    prepared = {split: prepare_split(source_dir, split, classes) for split in ("train", "val")}
    train_names = {record["relative_path"] for record in prepared["train"][0]}
    val_names = {record["relative_path"] for record in prepared["val"][0]}
    if train_names & val_names:
        raise ValueError("Train and validation splits contain overlapping images")

    output_dir.mkdir(parents=True, exist_ok=False)
    for split, (records, _) in prepared.items():
        for record in records:
            relative_path = record["relative_path"]
            output_image = output_dir / "images" / split / relative_path
            output_label = output_dir / "labels" / split / relative_path.with_suffix(".txt")
            output_image.parent.mkdir(parents=True, exist_ok=True)
            output_label.parent.mkdir(parents=True, exist_ok=True)
            if image_mode == "symlink":
                output_image.symlink_to(record["source_image"])
            else:
                shutil.copy2(str(record["source_image"]), str(output_image))
            lines = record["lines"]
            with output_label.open("x", encoding="utf-8") as stream:
                stream.write("\n".join(lines) + ("\n" if lines else ""))

    annotation_path = write_validation_annotations(output_dir, prepared["val"][0], classes)
    # JSON-quoted strings/lists are valid YAML; no PyYAML dependency is needed.
    with (output_dir / "dataset.yaml").open("x", encoding="utf-8") as stream:
        stream.write("# Human-Parts: official train/val split; zero-based YOLO classes.\n")
        for split in ("train", "val"):
            stream.write("{}: {}\n".format(split, json.dumps(str(output_dir / "images" / split))))
        stream.write("anno_path: {}\n".format(json.dumps(str(annotation_path))))
        stream.write("nc: {}\nnames: {}\nis_coco: False\n".format(len(classes), json.dumps(classes)))
        stream.write("# Evaluation uses the supplied JSON with positive annotation IDs.\n")

    summary = {"source_dir": str(source_dir), "output_dir": str(output_dir),
               "image_mode": image_mode, "names": list(classes),
               "source_bbox_format": "COCO top-left xywh in pixels",
               "label_format": "class_id center_x center_y width height (normalized)",
               "splits": {split: stats for split, (_, stats) in prepared.items()}}
    with (output_dir / "conversion_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--classes", nargs="+", choices=AVAILABLE_CLASSES, default=DEFAULT_CLASSES,
                        help="Retain these classes in this order (default: person only)")
    parser.add_argument("--image-mode", choices=("symlink", "copy"), default="symlink")
    args = parser.parse_args()
    try:
        summary = convert_dataset(args.source_dir, args.output_dir, args.classes, args.image_mode)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, "Conversion failed: {}\n".format(error))
    print(json.dumps(summary, indent=2))
    print("YOLOv6 data config: {}".format(Path(summary["output_dir"]) / "dataset.yaml"))


if __name__ == "__main__":
    main()
