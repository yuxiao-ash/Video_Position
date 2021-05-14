import os
import cv2
import glob

import imageio
import torch
import numpy as np
from torch.utils.data import Dataset
import random


def resize_img_keep_ratio(image, target_size):
    old_size = image.shape[0:2]
    ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))
    new_size = tuple([int(i * ratio) for i in old_size])
    image = cv2.resize(image, (new_size[1], new_size[0]))
    pad_w = target_size[1] - new_size[1]
    pad_h = target_size[0] - new_size[0]
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    img_new = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
    return img_new


def read_video(filename, img_size):
    vid = imageio.get_reader(filename, 'ffmpeg')
    video = []
    for im in vid:
        video.append(resize_img_keep_ratio(im, img_size) / 255)
    return video


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
            self.data = glob.glob(self.data_root + 'data_A/test/*.mp4')
        else:
            pass

    def __getitem__(self, item):
        if self.mode == 'train' or self.mode == 'val':
            data = self.data[item]
            audio_path = os.path.join(self.data_root, data[0] + '_{}.npy'.format(self.feature_type))
            video_path = os.path.join(self.data_root, data[0] + '.mp4')
            feature = np.load(audio_path)
            feature = np.pad(feature, ((0, 0), (0, self.feature_lenght - feature.shape[1])), 'constant')
            video = read_video(video_path, self.img_size)
            video = np.stack(video, axis=0)
            return torch.FloatTensor(video).permute(3, 0, 1, 2), torch.FloatTensor(feature), torch.FloatTensor([data[1] / 200]), data[0][:-9] + '.mp4'
        else:
            data = self.data[item]
            audio_path = self.data[item][:-4] + '_{}.npy'.format(self.feature_type)
            video_path = self.data[item]
            # print(video_path)
            feature = np.load(audio_path)
            feature = np.pad(feature, ((0, 0), (0, self.feature_lenght - feature.shape[1])), 'constant')
            video = read_video(video_path, self.img_size)
            video = np.stack(video, axis=0)
            return torch.FloatTensor(video).permute(3, 0, 1, 2), torch.FloatTensor(feature), video_path.split('/')[-1]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from models.utils import Loss_v2

    loss = Loss_v2()

    test_dataset = AVDataset(mode='test')
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=8,
                                             shuffle=False,
                                             num_workers=8,
                                             drop_last=False)
    for i_batch, (image, audio, image_name) in enumerate(test_loader):
        print(image.shape)
        print(audio.shape)
        print(image_name)
        break
        pass

    # val_dataset = AVDataset(mode='val')
    # val_loader = torch.utils.data.DataLoader(val_dataset,
    #                                          batch_size=8,
    #                                          shuffle=False,
    #                                          num_workers=8,
    #                                          drop_last=False)
    # for i_batch, (image, audio, label, image_name) in enumerate(val_loader):
    #     pass
    #
    # trn_dataset = AVDataset(mode='train')
    # trn_loader = torch.utils.data.DataLoader(trn_dataset,
    #                                          batch_size=8,
    #                                          shuffle=True,
    #                                          num_workers=8)
    # print(len(trn_loader))
    # for i_iter, (img_batch, audio, label, _) in enumerate(trn_loader):
    #     print(img_batch.shape)  # ([8, 200, 66, 88, 3])
    #     img = img_batch[0].permute(1, 2, 3, 0)
    #     index = label[0].item()
    #     print('label position: ' + str(index))
    #     for k in range(int(index - 3), int(index + 4)):
    #         print(k)
    #         sub_img = img[k].numpy()
    #         plt.imshow(sub_img)
    #         plt.show()
    #
    #     input = torch.randn(8, 210, requires_grad=True)
    #     cost = loss(input, label)
    #     print(cost)
    #
    #     if i_iter >= 0:
    #         break
