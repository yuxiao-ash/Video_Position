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
    def __init__(self, data_root='/home/insomnia/Video_Position/data/process_data/', mode='train', img_size=(84, 112)):
        self.data_root = data_root
        self.mode = mode
        self.img_size = img_size
        self.step = 2

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
            pass
        else:
            pass

    def __getitem__(self, item):
        data = self.data[item]
        video_path = os.path.join(self.data_root, data[0])

        # video = read_video(video_path, self.img_size)[::self.step]
        # label = 0.5
        # mov_step = int((random.random() * 2 - 1) * (200 // self.step))
        # if mov_step > 0:
        #     video = video[:len(video)-mov_step]
        #     for i in range(mov_step):
        #         video.insert(0, np.zeros((self.img_size[0], self.img_size[1], 3), dtype=float))
        # elif mov_step < 0:
        #     video = video[-mov_step:]
        #     for i in range(-mov_step):
        #         video.append(np.zeros((self.img_size[0], self.img_size[1], 3), dtype=float))
        # assert len(video) == (500 // self.step)
        # video = np.stack(video, axis=0)
        # label = int(0.5 * (500 // self.step) + mov_step) / (500 // self.step)
        # return torch.FloatTensor(video), torch.FloatTensor([label])

        video = read_video(video_path, self.img_size)
        label = float(data[1]) / 200
        video = np.stack(video, axis=0)
        return torch.FloatTensor(video), torch.FloatTensor([label]), data[0][:-9] + '.mp4'

    def __len__(self):
        return len(self.data)


class AudioDataset(Dataset):
    def __init__(self, data_root='/home/insomnia/Video_Position/data/', mode='train', feature_type='mfcc',
                 feature_lenght=8616, img_size=(66, 88)):
        if data_root is not None:
            self.data_root = data_root
        else:
            self.data_root = '/home/insomnia/Video_Position/data/'
        self.mode = mode
        self.feature_lenght = feature_lenght
        self.img_size = img_size

        if mode == 'train' or mode == 'val':
            label_file1 = os.path.join(self.data_root, 'data_A/train/必选数据.txt')
            label_file2 = os.path.join(self.data_root, 'data_B/补充数据.txt')
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
            return torch.from_numpy(feature), torch.FloatTensor([data[1] / 200]), data[0][:-9] + '.mp4'
        else:
            data = self.data[item]
            feature = np.load(data)
            feature = np.pad(feature, ((0, 0), (0, self.feature_lenght - feature.shape[1])), 'constant')
            return torch.FloatTensor(feature), data.split('/')[-1][:-9] + '.mp4'

    def __len__(self):
        return len(self.data)


class AVDataset(Dataset):
    def __init__(self, data_root='/home/insomnia/Video_Position/data/', mode='train', feature_type='mfcc',
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
            self.data = glob.glob(self.data_root + '*_{}.npy'.format(feature_type))
        else:
            pass

    def __getitem__(self, item):
        if self.mode == 'train' or self.mode == 'val':
            data = self.data[item]
            audio_path = os.path.join(self.data_root, data[0] + '_{}.npy'.format(self.feature_type))
            video_path = os.path.join(self.data_root, 'process_data', data[0] + '.mp4')
            feature = np.load(audio_path)
            feature = np.pad(feature, ((0, 0), (0, self.feature_lenght - feature.shape[1])), 'constant')
            video = read_video(video_path, self.img_size)
            video = np.stack(video, axis=0)
            return torch.FloatTensor(video), torch.from_numpy(feature), torch.FloatTensor([data[1] / 200]), data[0][
                                                                                                            :-9] + '.mp4'
        else:
            data = self.data[item]
            feature = np.load(data)
            feature = np.pad(feature, ((0, 0), (0, self.feature_lenght - feature.shape[1])), 'constant')
            return torch.FloatTensor(feature), data.split('/')[-1][:-9] + '.mp4'

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import time

    dataset = AudioDataset(data_root='/home/insomnia/Video_Position/data/', mode='train', feature_type='mfcc')
    data_loader = DataLoader(dataset, batch_size=24, shuffle=True, num_workers=4)

    s = time.time()
    for batch_idx, (inputs, targets, name) in enumerate(data_loader):
        print(time.time() - s)
        print(inputs.size())#24 20 8616
        print(inputs[0][0][0])
        print(targets[0])
        print(name[0])
        time.sleep(1)
        s = time.time()
    print('success')

# if __name__ == '__main__':
#     from torch.utils.data import DataLoader
#     import time
#     dataset = VideoDataset(data_root='/home/insomnia/Video_Position/data/process_data', mode='train', img_size=(88, 88))
#     data_loader =  DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
#
#     s = time.time()
#     for batch_idx, (inputs, targets) in enumerate(data_loader):
#         print(time.time() - s)
#         print(inputs.size())
#         print(targets)
#         time.sleep(1)
#         s = time.time()
#     print('success')
