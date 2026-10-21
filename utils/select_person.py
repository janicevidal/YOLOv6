import os
import shutil


def select_person(source_label_path, output_label_path, source_images_path, output_images_path):
    """Keep person labels (class 0) and copy each matching image once per run."""
    os.makedirs(output_label_path, exist_ok=True)
    os.makedirs(output_images_path, exist_ok=True)
    for txt_file in os.listdir(source_label_path):
        txt_name, extension = os.path.splitext(txt_file)
        label_path = os.path.join(source_label_path, txt_file)
        if extension.lower() != ".txt" or not os.path.isfile(label_path):
            continue

        person_lines = []
        with open(label_path, "r") as f:
            for line in f:
                fields = line.split()
                if fields and fields[0] == "0":
                    person_lines.append(line)

        if not person_lines:
            continue

        with open(os.path.join(output_label_path, txt_file), "w") as outfile:
            outfile.writelines(person_lines)

        src = os.path.join(source_images_path, txt_name + ".jpg")
        dst = os.path.join(output_images_path, txt_name + ".jpg")
        shutil.copy(src, dst)
        print("已完成", dst)


if __name__ == '__main__':
    train_source_labels = "/data/xiaoshuai/human_detection/coco/labels/train2017"
    train_output_labels = "/data/xiaoshuai/human_detection/coco_person/labels/train2017"
    train_source_images = "/data/xiaoshuai/human_detection/coco/images/train2017"
    train_output_images = "/data/xiaoshuai/human_detection/coco_person/images/train2017"
    select_person(train_source_labels, train_output_labels, train_source_images, train_output_images)
    valid_source_labels = "/data/xiaoshuai/human_detection/coco/labels/val2017"
    valid_output_labels = "/data/xiaoshuai/human_detection/coco_person/labels/val2017"
    valid_source_images = "/data/xiaoshuai/human_detection/coco/images/val2017"
    valid_output_images = "/data/xiaoshuai/human_detection/coco_person/images/val2017"
    select_person(valid_source_labels, valid_output_labels, valid_source_images, valid_output_images)
