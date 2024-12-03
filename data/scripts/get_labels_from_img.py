#coding=utf-8
import os
import shutil

def copy_yolo_labels(image_dir, label_dir, target_label_dir):
    """
    根据图片文件名复制对应的 YOLO 标签文件到目标文件夹，如果没有标签则跳过。

    Args:
        image_dir (str): 图片所在目录路径。
        label_dir (str): 标签文件所在目录路径。
        target_label_dir (str): 目标标签文件夹路径。
    """
    # 确保目标文件夹存在
    os.makedirs(target_label_dir, exist_ok=True)

    for image_name in os.listdir(image_dir):
        if image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # 检查图片文件格式
            base_name = os.path.splitext(image_name)[0]  # 去掉扩展名
            label_path = os.path.join(label_dir, f"{base_name}.txt")  # 对应的标签路径

            if os.path.exists(label_path):  # 检查标签文件是否存在
                target_path = os.path.join(target_label_dir, f"{base_name}.txt")  # 目标路径
                shutil.copy(label_path, target_path)  # 复制文件
                print(f"复制标签文件: {label_path} -> {target_path}")
            else:
                print(f"标签文件不存在，跳过: {label_path}")


# 示例使用
image_directory = "/home/jia/bird/images/train"  # 替换为实际的图片目录路径
label_directory = "/home/jia/PycharmProjects/YOLOv8-Multi-Modal-Fusion-Network-RGB-IR/bird/labels/train"  # 替换为实际的标签目录路径
target_directory = "/home/jia/bird/labels/train"  # 替换为目标标签文件夹路径

copy_yolo_labels(image_directory, label_directory, target_directory)
