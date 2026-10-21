import os

if __name__ == '__main__':
    train_images = '/data/xiaoshuai/human_detection/coco_person/images/train2017'
    valid_images = '/data/xiaoshuai/human_detection/coco_person/images/val2017'
    train_txt = '/data/xiaoshuai/human_detection/coco_person/train2017.txt'
    valid_txt = '/data/xiaoshuai/human_detection/coco_person/val2017.txt'
    
    train_file_list = os.listdir(train_images)
    val_file_list = os.listdir(valid_images)
    train_file = open(train_txt, 'a')
    valid_file = open(valid_txt, 'a')

    for file_obj in val_file_list:
        file_path = os.path.join(valid_images, file_obj)

        file_name, file_extend = os.path.splitext(file_obj)
        print(file_name)
        file_num = int(file_name)
        valid_file.write(file_path + '\n')
        
    valid_file.close()
    
    for file_obj in train_file_list:
        file_path = os.path.join(train_images, file_obj)

        file_name, file_extend = os.path.splitext(file_obj)
        file_num = int(file_name)
        train_file.write(file_path + '\n')

    train_file.close()