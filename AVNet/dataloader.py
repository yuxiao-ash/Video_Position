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
                    data_1.append([os.path.join('data_A/train/', line.split(' ')[0]+'.mp4'), float(line.split(' ')[1])])
            with open(label_file2, 'r') as file:
                lines = file.readlines()
                data_2 = []
                for line in lines:
                    data_2.append([os.path.join('data_B/', line.split(' ')[0]+'.mp4'), float(line.split(' ')[1])])
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
    def __init__(self, data_root='/home/insomnia/Video_Position/data/', mode='train', feature_type='mfcc', feature_lenght=8616):
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
                    data_1.append([os.path.join('data_A/train/', line.split(' ')[0]+'_{}.npy'.format(feature_type)), float(line.split(' ')[1])])
            with open(label_file2, 'r') as file:
                lines = file.readlines()
                data_2 = []
                for line in lines:
                    data_2.append([os.path.join('data_B/', line.split(' ')[0]+'_{}.npy'.format(feature_type)), float(line.split(' ')[1])])
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
            feature = np.pad(feature, ((0,0), (0, self.feature_lenght-feature.shape[1])), 'constant')
            label = data[1] / 200
            
            argument = False
            if argument and self.mode == 'train':
                raito = random.random()*2 - 1
                if raito < 0 and -raito < label:
                    label = label + raito
                    feature = feature[:, int(-raito*self.feature_lenght):]
                    feature = np.pad(feature, ((0,0), (0, self.feature_lenght-feature.shape[1])), 'constant')
                elif raito > 0 and raito < (1 - label):
                    label = label + raito
                    feature = feature[:, :int((1-raito)*self.feature_lenght)]
                    feature = np.pad(feature, ((0,0), (self.feature_lenght-feature.shape[1], 0)), 'constant')
            
            return torch.FloatTensor(feature), torch.FloatTensor([label]), data[0][:-9] + '.mp4'
        else:
            data = self.data[item]
            feature = np.load(data)
            feature = np.pad(feature, ((0,0), (0, self.feature_lenght-feature.shape[1])), 'constant')
            return torch.FloatTensor(feature), data.split('/')[-1][:-9] + '.mp4'

    def __len__(self):
        return len(self.data)


class AVDataset(Dataset):
    def __init__(self, data_root='/home/insomnia/Video_Position/data/process_data/', mode='train', feature_type='mfcc', feature_lenght=8616, img_size=(66,88)):
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
            audio = np.pad(audio, ((0,0), (0, self.feature_lenght-audio.shape[1])), 'constant')
            video = read_video(video_path, self.img_size)
            video = np.stack(video, axis=0)
            return torch.FloatTensor(video), torch.FloatTensor(audio), torch.FloatTensor([data[1] / 200]), data[0] + '.mp4'
        else:
            video_path = self.data[item]
            audio_path = self.data[item][:-4] + '_{}.npy'.format(self.feature_type)
            audio = np.load(audio_path)
            audio = np.pad(audio, ((0,0), (0, self.feature_lenght-audio.shape[1])), 'constant')
            video = read_video(video_path, self.img_size)
            video = np.stack(video, axis=0)
            return torch.FloatTensor(video), torch.FloatTensor(audio), video_path.split('/')[-1]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import time
    dataset = AudioDataset(data_root='/home/insomnia/Video_Position/data/', mode='train', feature_type='mfcc', feature_lenght=8616)
    data_loader =  DataLoader(dataset, batch_size=24, shuffle=True, num_workers=4)

    s = time.time()
    for batch_idx, (inputs, targets, _) in enumerate(data_loader):
        print(time.time() - s)
        print(inputs.size())
        #print(inputs[0][0][0])
        # print(targets)
        time.sleep(1)
        s = time.time()
    print('success')

# if __name__ == '__main__':
#     from torch.utils.data import DataLoader
#     import time
#     dataset = VideoDataset(data_root='/home/insomnia/Video_Position/data/process_data', mode='train', img_size=(84, 112))
#     data_loader =  DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

#     s = time.time()
#     for batch_idx, (inputs, targets) in enumerate(data_loader):
#         print(time.time() - s)
#         print(inputs.size())
#         print(targets)
#         time.sleep(1)
#         s = time.time()
#     print('success')

