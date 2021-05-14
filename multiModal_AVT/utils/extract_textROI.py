import json
import os
import cv2
import glob
import torch
import random
import imageio
import librosa
import numpy as np
from torch.utils.data import Dataset
from moviepy.editor import AudioFileClip
from utils import read_video
from dataloader import load_json, sort_two_num, trans_bbox_to_roi


class AVTDataset(Dataset):
    def __init__(self, data_root='/home/insomnia/Video_Position/data/process_data/',
                 mode='train',
                 feature_type='mfcc',
                 feature_lenght=8616,
                 text_path='./datatools/ocr_text_trn.json',
                 ROI_size=(27, 36),
                 img_size=(66, 88)):
        self.data_root = data_root
        self.mode = mode
        self.feature_lenght = feature_lenght
        self.feature_type = feature_type
        self.img_size = img_size
        self.text = load_json(text_path)  # dict
        self.ROI_size = ROI_size

        self.save_dir = self.data_root + 'ROIs'
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        if mode == 'train' or mode == 'val':
            label_file1 = os.path.join(data_root, 'data_A/train/必选数据.txt')
            label_file2 = os.path.join(data_root, 'data_B/补充数据.txt')
            with open(label_file1, 'r') as file:
                lines = file.readlines()
                data_1 = []
                for line in lines:
                    data_1.append([os.path.join('data_A/train/', line.split(' ')[0]), float(line.split(' ')[1])])
            with open(label_file2, 'r') as file:
                lines = file.readlines()
                data_2 = []
                for line in lines:
                    data_2.append([os.path.join('data_B/', line.split(' ')[0]), float(line.split(' ')[1])])
            self.data = data_1 + data_2

            if mode == 'val':
                self.data = self.data[::7]
            elif mode == 'train':
                del self.data[::7]
        elif mode == 'test':
            self.data = glob.glob(self.data_root + '*.mp4')
        else:
            pass

    def __getitem__(self, item):
        if self.mode == 'train' or self.mode == 'val':
            data = self.data[item]
            audio_path = os.path.join(self.data_root, data[0] + '_{}.npy'.format(self.feature_type))
            video_path = os.path.join(self.data_root, data[0] + '.mp4')
            audio = np.load(audio_path)
            audio = np.pad(audio, ((0, 0), (0, self.feature_lenght - audio.shape[1])), 'constant')
            video = read_video(video_path, self.img_size)
            video = np.stack(video, axis=0)
            return torch.FloatTensor(video), torch.from_numpy(audio), torch.FloatTensor([data[1] / 200]), data[
                0] + '.mp4', self.gen_offline_ROI(data[0].split('/')[-1])
        else:
            video_path = self.data[item]
            audio_path = self.data[item][:-4] + '_{}.npy'.format(self.feature_type)
            audio = np.load(audio_path)
            audio = np.pad(audio, ((0, 0), (0, self.feature_lenght - audio.shape[1])), 'constant')
            video = read_video(video_path, self.img_size)
            video = np.stack(video, axis=0)
            return torch.FloatTensor(video), torch.from_numpy(audio), video_path.split('/')[-1], self.gen_offline_ROI(
                video_path.split('/')[-1].split('.')[0])

    def __len__(self):
        return len(self.data)

    def gen_offline_ROI(self, name):
        """
        离线生成ROI
        """
        t_sample = self.text[name]
        ROI = []
        for i in range(200):
            t_res = t_sample[str(i)]
            # 当前时刻的ROI
            t_ROI = np.zeros(self.ROI_size)
            # ocr的结果是否为空列表
            if len(t_res) != 0:
                for j in range(len(t_res)):
                    bbox = trans_bbox_to_roi([t_res[j][0][0], t_res[j][0][2]], t_res[j][1], self.ROI_size)
                    x1, y1, x2, y2 = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
                    # 筛去太小的框
                    if x1 == x2 or y1 == y2:
                        continue
                    # 斜框处理
                    x1, x2 = sort_two_num(x1, x2)
                    y1, y2 = sort_two_num(y1, y2)
                    # 边界处理（easyocr识别的倾斜文本框能识别到图像外面）
                    try:
                        t_ROI[y1:y2, x1:x2] = np.ones((y2 - y1, x2 - x1))
                    except:
                        # print('错误样本： ' + name + ' ，第 {} 帧, t_res和bbox分别为'.format(i))
                        # print(t_res)
                        # print(bbox)
                        continue
            ROI.append(t_ROI)
        ROI = np.stack(ROI, axis=0)

        roi_path = os.path.join(self.save_dir, name + '.npy')
        np.save(roi_path, ROI)

        return torch.from_numpy(ROI).float()


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import time

    # dataset = AVTDataset(data_root='/home/insomnia/Video_Position/data/process_data/',
    #                      mode='train',
    #                      img_size=(84, 112),
    #                      text_path='../datatools/ocr_text_trn_full.json')
    dataset = AVTDataset(data_root='/home/insomnia/Video_Position/data/process_data/',
                         mode='val',
                         img_size=(84, 112),
                         text_path='../datatools/ocr_text_trn_full.json')
    # dataset = AVTDataset(data_root='/home/insomnia/Video_Position/data/process_data/data_A/test/',
    #                      mode='test',
    #                      img_size=(84, 112),
    #                      text_path='../datatools/ocr_text_tst.json')
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    print(len(data_loader))

    s = time.time()
    for batch_idx, (video, audio, target, name, text_ROI) in enumerate(data_loader):
        print(time.time() - s)
        print(text_ROI.size())
        # print(targets)
        # time.sleep(1)
        s = time.time()
    print('success')
