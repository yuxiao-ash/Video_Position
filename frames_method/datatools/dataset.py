import os
import cv2
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
import random


class FrameDataset(Dataset):
    def __init__(self,
                 data_root='/home/insomnia/Video_Position/code/frames_method/datatools/frames_data',
                 label_path='/home/insomnia/Video_Position/code/frames_method/datatools',
                 mode='train', img_size=(84, 112), seq_len=210):
        self.data_root = data_root
        self.mode = mode
        self.img_size = img_size
        self.seq_len = seq_len
        self.label = []
        if mode == 'train':
            file = open(os.path.join(label_path, 'train.txt'), 'r', encoding='utf-8')
            label_lines = file.readlines()
            for line in label_lines:
                self.label.append({'image': line.split(' ')[0].strip(),
                                   'label': float(line.split(' ')[1].strip())})
        elif mode == 'val':
            file = open(os.path.join(label_path, 'val.txt'), 'r', encoding='utf-8')
            label_lines = file.readlines()
            for line in label_lines:
                self.label.append({'image': line.split(' ')[0].strip(),
                                   'label': float(line.split(' ')[1].strip())})
        else:
            pass

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        if self.mode == 'train' or self.mode == 'val':
            img_dir = self.data_root + '/train/' + self.label[item]['image']
            img_path_list = os.listdir(img_dir)
            img_path_list.sort(key=lambda x: int(x[:-4]))

            frames_ = []
            for i, img_path in enumerate(img_path_list):
                if i >= self.seq_len:
                    break
                image = cv2.imread(img_dir + '/' + img_path)
                image = self.resize_img_keep_ratio(image, self.img_size)
                frames_.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.)
            frames_ = np.stack(frames_, axis=0)
            if self.mode == 'train':
                frames_ = self.HorizontalFlip(frames_)

            vlm = torch.zeros((3, self.seq_len, self.img_size[0], self.img_size[1]))
            for i in range(frames_.shape[0]):
                temp = frames_[i]
                result = torch.tensor(temp).permute(2, 0, 1).float()
                vlm[:, i] = result
            label = torch.tensor(self.label[item]['label']).long()
            return vlm, label, self.label[item]['image']

    @staticmethod
    def img_noise(image_in, noise_sigma=0):
        temp_image = image_in

        h = temp_image.shape[0]
        w = temp_image.shape[1]
        noise = np.random.randn(h, w) * noise_sigma * 255

        temp_image[:, :, 0] = temp_image[:, :, 0] + noise
        temp_image[:, :, 1] = temp_image[:, :, 1] + noise
        temp_image[:, :, 2] = temp_image[:, :, 2] + noise
        temp_image[temp_image < 0] = 0
        temp_image[temp_image > 255] = 255

        return temp_image

    @staticmethod
    def HorizontalFlip(batch_img):
        if random.random() > 0.5:
            batch_img = np.ascontiguousarray(batch_img[:, :, ::-1])
        return batch_img

    @staticmethod
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from models.utils import FrameLoss
    loss = FrameLoss()

    val_dataset = FrameDataset(mode='val')
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=8,
                                             shuffle=False,
                                             num_workers=8,
                                             drop_last=False)
    for i_batch, (image, label, image_name) in enumerate(val_loader):
        pass

    trn_dataset = FrameDataset()
    trn_loader = torch.utils.data.DataLoader(trn_dataset,
                                             batch_size=8,
                                             shuffle=True,
                                             num_workers=8)
    print(len(trn_loader))
    for i_iter, (img_batch, label, _) in enumerate(trn_loader):
        print(img_batch.shape)
        img = img_batch[0].permute(1, 2, 3, 0)
        index = label[0].item()
        print('label position: ' + str(index))
        for k in range(int(index - 3), int(index + 4)):
            print(k)
            sub_img = img[k].numpy()
            plt.imshow(sub_img)
            plt.show()

        input = torch.randn(8, 210, requires_grad=True)
        cost = loss(input, label)
        print(cost)

        if i_iter >= 0:
            break
