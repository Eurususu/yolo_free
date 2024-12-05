#coding=utf-8
import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
import os
from utils.general import scale_coords
from utils.ops import non_max_suppression_v11
from utils.torch_utils import time_synchronized
from utils.datasets import letterbox
from utils.plots import colors, plot_one_box
from models.experimental import attempt_load


def adaptive_threshold(diff_image, max_val=225):
    # 自适应阈值，可以更好地检测慢速运动物体的轮廓
    adaptive_thresh = cv2.adaptiveThreshold(
        diff_image,
        max_val,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 使用高斯加权
        cv2.THRESH_BINARY,
        5,  # 阈值计算时邻域的大小，可以根据需求调整
        5    # 常量C，调节阈值敏感度
    )
    return adaptive_thresh


def process_video(opt, video_path, model):
    # 打开视频文件或摄像头
    cap = cv2.VideoCapture(video_path)  # 可以改成 0 来使用摄像头

    # 读取第一帧作为参考帧
    ret, first_frame = cap.read()
    while not ret:
        ret, first_frame = cap.read()
    # 将第一帧转换为灰度图
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    # 读取当前帧
    ret, second_frame = cap.read()
    while not ret:
        ret, second_frame = cap.read()
    # 将当前帧转换为灰度图
    gray = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)
    # 计算当前帧与上一帧的差分
    diff1 = cv2.absdiff(gray, prev_gray)
    # 累积初始化
    accumulated_diff = np.zeros_like(gray, dtype=np.float32)

    # 设置累积的衰减参数
    alpha = 0.5  # 衰减系数，用于控制累积的效果
    thresh = 0  # 根据噪声情况调整
    frame_count = 0
    curr_frame = second_frame
    names = ['bird']
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # 将下一帧转换为灰度图
        next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 计算下一帧与当前帧的差分
        diff2 = cv2.absdiff(next_gray, gray)
        # 取与运算
        diff = cv2.bitwise_and(diff1, diff2)
        # 去掉残影
        diff = adaptive_threshold(diff)
        diff = cv2.bitwise_not(diff)
        # 更新累积图像 并做归一化
        accumulated_diff = cv2.addWeighted(accumulated_diff, alpha, diff1.astype(np.float32), 1 - alpha, 0)
        normalized_accumulated_diff_ = cv2.normalize(accumulated_diff, None, 0, 255, cv2.NORM_MINMAX)
        normalized_accumulated_diff_ = normalized_accumulated_diff_.astype(np.uint8)
        diff_expanded_ = np.expand_dims(normalized_accumulated_diff_, axis=2)
        # _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        # accumulated_diff += diff.astype(np.float32)

        # 将累积图像进行归一化
        # normalized_accumulated_diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        # normalized_accumulated_diff = normalized_accumulated_diff.astype(np.uint8)
        diff = diff.astype(np.uint8)
        diff_expanded = np.expand_dims(diff, axis=2)
        img = letterbox(curr_frame, opt.img_size, stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        gray = np.expand_dims(gray, axis=2)
        two_channel_image = np.concatenate((gray, diff_expanded_), axis=2)
        three_channel_image = np.concatenate((two_channel_image, diff_expanded), axis=2)
        img1 = letterbox(three_channel_image, opt.img_size, stride=32)[0]
        img1 = img1[:, :, ::-1].transpose(2, 0, 1)
        img1 = np.ascontiguousarray(img1)
        # four_channel_image = np.concatenate((frame_rgb, diff_expanded_), axis=2)
        # five_channel_image = np.concatenate((four_channel_image, diff_expanded), axis=2)
        img = torch.from_numpy(img).cuda()
        img1 = torch.from_numpy(img1).cuda()

        img = img.half() # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        img1 = img1.half()  # uint8 to fp16/32
        img1 /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img1.ndimension() == 3:
            img1 = img1.unsqueeze(0)
        t1 = time_synchronized()
        pred = model(img, img1, augment=False)
        # Apply NMS
        pred = non_max_suppression_v11(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, max_det=opt.max_det)
        t2 = time_synchronized()
        # Print time (inference + NMS)
        print(f'({t2 - t1:.6f}s, {1 / (t2 - t1):.6f}Hz)')
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], curr_frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                    plot_one_box(xyxy, curr_frame, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
        cv2.imshow('YOLOv8 Inference', curr_frame)
        # 显示结果
        # cv2.imshow("Accumulated Frame Difference", diff)

        video_name = Path(video_path).name
        video_name = Path(video_name).stem
        # cv2.imwrite(f'pictures/{video_name}_{frame_count:06}.jpg', curr_frame)
        # cv2.imwrite(f'/home/jia/bird/images/val/{video_name}_{frame_count:06}.tiff', four_channel_image)
        # tiff.imwrite(f'/home/jia/bird/images/train/{video_name}_{frame_count:06}.tiff', five_channel_image)
        # cv2.imwrite(f'/home/jia/bird/images/train/{video_name}_{frame_count:06}.jpg', curr_frame)
        # cv2.imwrite(f'/home/jia/bird/image/train/{video_name}_{frame_count:06}.jpg', three_channel_image)
        # 更新前一帧
        gray = next_gray
        diff1 = diff2
        curr_frame = frame

        # 按下 'q' 键退出循环
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='object confidence threshold')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--img-size', nargs='+', type=int, default=[736,1280], help='inference size (pixels)')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=True, action='store_true', help='hide confidences')
    parser.add_argument('--max-det',type=int, default=300, help='the max count detected')
    opt = parser.parse_args()
    model = attempt_load(opt.weights, map_location='cuda')
    model.half()
    video_dir = '/home/jia/anktechDrive/09_dataset/FBD-SV-2024/videos/val'
    for video in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video)
        with torch.no_grad():
            process_video(opt, video_path, model)
    # video_path = '/home/jia/anktechDrive/12_原始数据/我奥赛事/20230912-175555/0.mp4'
    # process_video(video_path, model)
