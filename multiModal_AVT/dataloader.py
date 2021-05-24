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
import torchvision.transforms as transforms


class VideoDataset(Dataset):
    def __init__(self, data_root='/home/insomnia/Video_Position/data/process_data/', mode='train', img_size=(66, 88)):
        self.data_root = data_root
        self.mode = mode
        self.img_size = img_size

        if mode == 'train' or mode == 'val':
            label_file1 = os.path.join(data_root, 'data_A/train/必选数据.txt')
            label_file2 = os.path.join(data_root, 'data_B/补充数据.txt')
            with open(label_file1, 'r') as file:
                lines = file.readlines()
                data_1 = []
                for line in lines:
                    data_1.append(
                        [os.path.join('data_A/train/', line.split(' ')[0] + '.mp4'), float(line.split(' ')[1])])
            with open(label_file2, 'r') as file:
                lines = file.readlines()
                data_2 = []
                for line in lines:
                    data_2.append([os.path.join('data_B/', line.split(' ')[0] + '.mp4'), float(line.split(' ')[1])])
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
            video_path = os.path.join(self.data_root, data[0])
            video = read_video(video_path, self.img_size)
            label = float(data[1]) / 200
            video = np.stack(video, axis=0)
            return torch.FloatTensor(video), torch.FloatTensor([label]), data[0]
        elif self.mode == 'test':
            video_path = self.data[item]
            video = read_video(video_path, self.img_size)
            video = np.stack(video, axis=0)
            return torch.FloatTensor(video), video_path.split('/')[-1]

    def __len__(self):
        return len(self.data)


class AudioDataset(Dataset):
    def __init__(self, data_root='/home/insomnia/Video_Position/data/', mode='train', feature_type='mfcc',
                 feature_lenght=8616):
        self.data_root = data_root
        self.mode = mode
        self.feature_lenght = feature_lenght

        if mode == 'train' or mode == 'val':
            label_file1 = os.path.join(data_root, 'data_A/train/必选数据.txt')
            label_file2 = os.path.join(data_root, 'data_B/补充数据.txt')
            with open(label_file1, 'r') as file:
                lines = file.readlines()
                data_1 = []
                for line in lines:
                    data_1.append([os.path.join('data_A/train/', line.split(' ')[0] + '_{}.npy'.format(feature_type)),
                                   float(line.split(' ')[1])])
            with open(label_file2, 'r') as file:
                lines = file.readlines()
                data_2 = []
                for line in lines:
                    data_2.append([os.path.join('data_B/', line.split(' ')[0] + '_{}.npy'.format(feature_type)),
                                   float(line.split(' ')[1])])
            self.data = data_1 + data_2

            if mode == 'val':
                self.data = self.data[::7]
            elif mode == 'train':
                del self.data[::7]
        elif mode == 'test':
            self.data = glob.glob(self.data_root + '*_{}.npy'.format(feature_type))
        else:
            pass

    def __getitem__(self, item):
        if self.mode == 'train' or self.mode == 'val':
            data = self.data[item]
            audio_path = os.path.join(self.data_root, data[0])
            feature = np.load(audio_path)
            feature = np.pad(feature, ((0, 0), (0, self.feature_lenght - feature.shape[1])), 'constant')
            label = data[1] / 200

            argument = True
            if argument and self.mode == 'train':
                raito = random.random() * 2 - 1
                if raito < 0 and -raito < label:
                    label = label + raito
                    feature = feature[:, int(-raito * self.feature_lenght):]
                    feature = np.pad(feature, ((0, 0), (0, self.feature_lenght - feature.shape[1])), 'constant')
                elif raito > 0 and raito < (1 - label):
                    label = label + raito
                    feature = feature[:, :int((1 - raito) * self.feature_lenght)]
                    feature = np.pad(feature, ((0, 0), (self.feature_lenght - feature.shape[1], 0)), 'constant')

            return torch.FloatTensor(feature), torch.FloatTensor([label]), data[0][:-9] + '.mp4'
        else:
            data = self.data[item]
            feature = np.load(data)
            feature = np.pad(feature, ((0, 0), (0, self.feature_lenght - feature.shape[1])), 'constant')
            return torch.FloatTensor(feature), data.split('/')[-1][:-9] + '.mp4'

    def __len__(self):
        return len(self.data)


class AVDataset(Dataset):
    def __init__(self, data_root='/home/insomnia/Video_Position/data/process_data/', mode='train', feature_type='mfcc',
                 feature_lenght=8616, img_size=(66, 88)):
        self.data_root = data_root
        self.mode = mode
        self.feature_lenght = feature_lenght
        self.feature_type = feature_type
        self.img_size = img_size

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
                0] + '.mp4'
        else:
            video_path = self.data[item]
            audio_path = self.data[item][:-4] + '_{}.npy'.format(self.feature_type)
            audio = np.load(audio_path)
            audio = np.pad(audio, ((0, 0), (0, self.feature_lenght - audio.shape[1])), 'constant')
            video = read_video(video_path, self.img_size)
            video = np.stack(video, axis=0)
            return torch.FloatTensor(video), torch.from_numpy(audio), video_path.split('/')[-1]

    def __len__(self):
        return len(self.data)


def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def trans_bbox_to_roi(box, img_size, tgt_size):
    """
    :param box: [[x1, y1], [x2, y2]]
    :param img_size: 原图像尺寸
    :param tgt_size: 目标尺寸
    :return: 在目标尺寸上的ROI框
    """
    h_t, w_t = tgt_size[0], tgt_size[1]
    h_s, w_s = img_size[0], img_size[1]
    ratio = min(float(h_t / h_s), float(w_t / w_s))
    new_size = tuple([int(i * ratio) for i in img_size])
    pad_w = (w_t - new_size[1]) // 2
    pad_h = (h_t - new_size[0]) // 2
    roi = []
    for cord in box:
        roi_cord = [0, 0]
        roi_cord[0] = int(cord[0] * ratio + pad_w)
        roi_cord[1] = int(cord[1] * ratio + pad_h)
        roi.append(roi_cord)
    return roi


def sort_two_num(num1, num2):
    if num1 < num2:
        return num1, num2
    else:
        return num2, num1


class AVTDataset(Dataset):
    def __init__(self, data_root='/home/insomnia/Video_Position/data/process_data/',
                 mode='train',
                 feature_type='mfcc',
                 feature_lenght=8616,
                 ROI_size=(27, 36),
                 img_size=(66, 88),
                 if_normal=False,
                 seed=0):
        self.data_root = data_root
        self.mode = mode
        self.feature_lenght = feature_lenght
        self.feature_type = feature_type
        self.img_size = img_size
        self.ROI_size = ROI_size
        self.if_normal = if_normal
        self.seed = int(seed) % 7

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

            while self.seed > 0:
                temp = self.data.pop(0)
                self.data.append(temp)
                self.seed -= 1

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
            name = data[0].split('/')[-1]
            ROI_path = os.path.join(self.data_root, 'ROIs/' + name + '.npy')
            audio = np.load(audio_path)
            audio = np.pad(audio, ((0, 0), (0, self.feature_lenght - audio.shape[1])), 'constant')
            video = read_video(video_path, self.img_size)

            if self.if_normal:
                normal_video = []
                for v in video:
                    result = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])(v)
                    normal_video.append(result)
                video = torch.stack(normal_video, dim=0).float()
            else:
                video = np.stack(video, axis=0).transpose((0, 3, 1, 2))
            ROI = np.load(ROI_path)
            return torch.FloatTensor(video), torch.from_numpy(audio), torch.FloatTensor([data[1] / 200]), data[
                0] + '.mp4', torch.from_numpy(ROI).float()
        else:
            video_path = self.data[item]
            audio_path = self.data[item][:-4] + '_{}.npy'.format(self.feature_type)
            name = video_path.split('/')[-1].split('.')[0]
            ROI_path = os.path.join(self.data_root, 'ROIs/' + name + '.npy')
            audio = np.load(audio_path)
            audio = np.pad(audio, ((0, 0), (0, self.feature_lenght - audio.shape[1])), 'constant')
            video = read_video(video_path, self.img_size)
            if self.if_normal:
                normal_video = []
                for v in video:
                    result = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])(v)
                    normal_video.append(result)
                video = torch.stack(normal_video, dim=0).float()
            else:
                video = np.stack(video, axis=0).transpose((0, 3, 1, 2))
            ROI = np.load(ROI_path)
            return torch.FloatTensor(video), torch.from_numpy(audio), video_path.split('/')[-1], torch.from_numpy(ROI).float()

    def __len__(self):
        return len(self.data)



# if __name__ == '__main__':
#     from torch.utils.data import DataLoader
#     import time
#     import matplotlib.pyplot as plt
#
#     data_root = '/home/insomnia/Video_Position/data/process_data/'
#     text_path = data_root + 'ocr_text.json'
#     s = time.time()
#     text = load_json(text_path)
#     # print(time.time() - s)# 3.224s
#     # print(len(text))#4100
#     size = (27, 36)
#     name = '0Gv5nPPa_start'
#     t_sample = text[name]
#     for i in range(200):
#         t_res = t_sample[str(i)]
#         t_ROI = np.zeros(size, dtype='uint8')
#         # ocr的结果是否为空列表
#         if len(t_res) == 0:
#             plt.imshow(t_ROI)
#             plt.show()
#         else:
#             for j in range(len(t_res)):
#                 bbox = box_invTrans(t_res[j][0], size)
#                 x1, y1, x2, y2 = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
#                 zone = np.ones((y2-y1, x2-x1), dtype='uint8')
#                 t_ROI[y1:y2, x1:x2] = zone
#                 plt.imshow(t_ROI)
#                 plt.show()
#     pass


# if __name__=='__main__':
#     data_root = '/home/insomnia/Video_Position/data/process_data/'
#     text_path = data_root + 'ocr_text.json'
#     text = load_json(text_path)
#     size = (27, 36)
#     for name in text.keys():
#         t_sample = text[name]
#         for i in range(200):
#             t_res = t_sample[str(i)]
#             # ocr的结果是否为空列表
#             if len(t_res) == 0:
#                 continue
#             else:
#                 for j in range(len(t_res)):
#                     bbox = box_invTrans(t_res[j][0], size)
#                     x1, y1, x2, y2 = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
#                     if (y2-y1) < 0 or (x2-x1) < 0:
#                         print(name + '\t第 {} 张 第 {} 个框：'.format(i, j))
#                         print('x1 : {}, x2 : {}, y1 : {}, y2 : {}'.format(x1, x2, y1, y2))
#                         print('====='*20)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import time
    import matplotlib.pyplot as plt

    dataset = AVTDataset(data_root='/home/insomnia/Video_Position/data/process_data',
                         mode='val',
                         img_size=(84, 112))
    # dataset = AVTDataset(data_root='/home/insomnia/Video_Position/data/process_data/data_A/test/',
    #                      mode='test',
    #                      img_size=(84, 112))
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    print(len(data_loader))

    s = time.time()
    for batch_idx, (video, audio, target, name, text_ROI) in enumerate(data_loader):
        print(time.time() - s)
        for i in range(video.size(0)):
            for j in range(0, 200, 10):
                img = video[i][j].numpy()
                plt.imshow(img)
                plt.show()
                # input()
                plt.imshow(text_ROI[i][j].numpy())
                plt.show()
                input()
        # print(text_ROI.size())
        # print(type(audio))
        # print(type(text_ROI))
        # print(targets)
        # time.sleep(1)
        s = time.time()
    print('success')
