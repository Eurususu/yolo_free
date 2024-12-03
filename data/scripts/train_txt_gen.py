#coding=utf-8
import os


def create_yolo_data_list(images_dir, output_file):
    # 获取所有图片文件的路径
    images = [os.path.join(images_dir, img) for img in os.listdir(images_dir) if
              img.endswith(('.jpg', '.png', '.jpeg'))]
    images.sort()  # 可选，排序一下方便查看

    # 将图片路径写入到文件
    with open(output_file, 'w') as f:
        for image_path in images:
            f.write(f"{image_path}\n")
    print(f"{output_file} 已生成，共 {len(images)} 张图片。")


# 设置图片文件夹路径和输出文件路径
train_images_dir = '/workspace/project/multispectral-object-detection/bird/images/train'
val_images_dir = '/workspace/project/multispectral-object-detection/bird/images/val'
train_output_file = '/workspace/project/multispectral-object-detection/bird/train.txt'
val_output_file = '/workspace/project/multispectral-object-detection/bird/val.txt'

# 生成 train.txt 和 val.txt
create_yolo_data_list(train_images_dir, train_output_file)
create_yolo_data_list(val_images_dir, val_output_file)
