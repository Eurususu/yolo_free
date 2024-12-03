#coding=utf-8
import cv2
import os


def extract_frames(video_path, output_dir):
    """
    从视频中提取所有帧并保存到指定目录。

    Args:
        video_path (str): 输入视频文件路径。
        output_dir (str): 输出帧文件保存的目录。
    """
    # 检查输出目录是否存在，不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    frame_count = 0

    while True:
        # 读取一帧
        ret, frame = cap.read()

        if not ret:
            print("视频读取完毕或出错")
            break

        # 生成帧的文件名
        frame_filename = os.path.join(output_dir, f"{frame_count:06d}.jpg")

        # 保存帧到文件
        cv2.imwrite(frame_filename, frame)

        print(f"保存帧: {frame_filename}")

        frame_count += 1

    # 释放视频文件
    cap.release()
    print(f"帧提取完成，共提取 {frame_count} 帧")


# 使用示例
video_path = "/home/jia/football_video/20240808-182936_003749.mp4"  # 替换为视频文件路径
output_dir = "/home/jia/football_video/20240808-182936_003749"  # 替换为保存帧的目录
extract_frames(video_path, output_dir)
