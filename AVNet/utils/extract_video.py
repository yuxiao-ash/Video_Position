import cv2
import os
import multiprocessing
import imageio
import time
import glob
import numpy as np


#============================================================================================================================
#路径设置
data_root='/home/insomnia/Video_Position/data/'                  # 数据集所在根目录          
save_root = '/home/insomnia/Video_Position/data/process_data/'   # 处理后数据集保存的目录

#============================================================================================================================
# 功能函数

def read_video(filename):
    vid = imageio.get_reader(filename, 'ffmpeg')
    video = []
    for im in vid:
        video.append(im)
    return video

def write_video(images, save_path, image_size, fps=25):
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    vw = cv2.VideoWriter(save_path, fourcc, fps, image_size)
    for image in images:
        vw.write(image)
    vw.release()
#============================================================================================================================
# 训练集部分

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

labels = label_1 + label_2

def work1(work_id, datas):
    print('work{} start!'.format(work_id))
    for i, data in enumerate(datas):
        t_start = time.time()
        video_path = os.path.join(data_root, data[0])
        save_path = os.path.join(save_root, data[0])
        
        if not os.path.exists('/'.join(save_path.split('/')[:-1])):
            os.makedirs('/'.join(save_path.split('/')[:-1]))
        
        images = read_video(video_path)
        images = images[::25]
        for _ in range(200 - len(images)):
            images.append(np.zeros((images[0].shape[0], images[0].shape[1], 3), dtype=np.uint8))
        assert len(images) == 200
        image_size = (images[0].shape[1], images[0].shape[0])
        write_video(images, save_path, image_size, 1)
        t_end = time.time()
        if (i+1) % 10 == 0 or i == len(datas) - 1:
            print('work_id:{}, {:.2f}%, eta:{:.2f}h'.format(work_id, (i + 1) / len(datas) * 100, (t_end - t_start)*(len(datas)-i-1)/3600))


num_workers = 8
step = len(labels) // num_workers
process = []
for i in range(num_workers):
    if i == num_workers - 1:
        datas = labels[i*step:]
    else:
        datas = labels[i*step: (i+1)*step]
    p = multiprocessing.Process(target = work1, args = (i,datas,))
    p.daemon = True
    p.start()
    process.append(p)

for p in process:
    p.join()



#===========================================================================================================================================
#测试集部分

data_root = os.path.join(data_root, 'data_A/test/')
save_root = os.path.join(save_root, 'data_A/test/')
files = glob.glob(data_root + '*.mp4')

def work2(work_id, datas):
    print('work{} start!'.format(work_id))
    for i, data in enumerate(datas):
        t_start = time.time()
        video_path = data
        save_path = os.path.join(save_root, data.split('/')[-1])
        
        if not os.path.exists('/'.join(save_path.split('/')[:-1])):
            os.makedirs('/'.join(save_path.split('/')[:-1]))
        
        images = read_video(video_path)
        images = images[::25]
        for _ in range(200 - len(images)):
            images.append(np.zeros((images[0].shape[0], images[0].shape[1], 3), dtype=np.uint8))
        assert len(images) == 200
        image_size = (images[0].shape[1], images[0].shape[0])
        write_video(images, save_path, image_size, 1)
        t_end = time.time()
        if (i+1) % 10 == 0 or i == len(datas) - 1:
            print('work_id:{}, {:.2f}%, eta:{:.2f}h'.format(work_id, (i + 1) / len(datas) * 100, (t_end - t_start)*(len(datas)-i-1)/3600))


num_workers = 8
step = len(files) // num_workers
process = []
for i in range(num_workers):
    if i == num_workers - 1:
        datas = files[i*step:]
    else:
        datas = files[i*step: (i+1)*step]
    p = multiprocessing.Process(target = work2, args = (i, datas,))
    p.daemon = True
    p.start()
    process.append(p)

for p in process:
    p.join()