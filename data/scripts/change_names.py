#coding=utf-8
import os

input_ = '/home/jia/football_video'
for i in os.listdir(input_):
    input_dir = os.path.join(input_, i)
    if os.path.isdir(input_dir):
        for input in sorted(os.listdir(input_dir)):
            old_path = os.path.join(input_dir, input)
            num = int(input.split('.')[0]) - 1
            new_path = os.path.join(input_dir, f'{num:06d}.txt')
            os.rename(old_path, new_path)