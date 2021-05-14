import os
import cv2
import imageio
import skimage
import numpy as np


data_root='/home/insomnia/Video_Position/data/'
label_file1 = os.path.join(data_root, 'data_A/train/必选数据.txt')
label_file2 = os.path.join(data_root, 'data_B/补充数据.txt')
with open(label_file1, 'r') as file:
    lines = file.readlines()
    label_1 = []
    for line in lines:
        label_1.append([os.path.join('data_A/train/', line.split(' ')[0]+'.mp4'), float(line.split(' ')[1])])
with open(label_file2, 'r') as file:
    lines = file.readlines()
    label_2 = []
    for line in lines:
        label_2.append([os.path.join('data_B/', line.split(' ')[0]+'.mp4'), float(line.split(' ')[1])])

label = label_1 + label_2

max_time = 0
min_time = 180
for l in label:
    if 'start' in l[0]:
        max_time = max(l[1], max_time)
        min_time = min(l[1], min_time)
print(min_time, max_time)

