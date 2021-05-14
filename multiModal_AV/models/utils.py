import csv
from abc import ABC

import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


import os
import datetime


class LogTxt(object):
    ### train logger
    def __init__(self, path, description):
        if not os.path.exists(path):
            os.mkdir(path)
        self.file = os.path.join(path, 'train_log.txt')
        logger = open(self.file, 'a')
        logger.write('\n' + description + '\n')
        logger.close()

    def log_train(self, epoch, loss, optimizer):
        curr_time = datetime.datetime.now()
        curr_time = datetime.datetime.strftime(curr_time, '%Y-%m-%d %H:%M:%S')
        logger = open(self.file, 'a')
        logger.write('\n' + '--'*40 +
            '\n {Time}\tEpoch: {epoch} \tLoss: {loss:0.4f}\tLR: {lr:0.6f}'.format(Time=curr_time, epoch=epoch, loss=loss, lr=optimizer.param_groups[0]['lr'])
                     )
        logger.close()

    def log_val(self, acc, best_acc):
        logger = open(self.file, 'a')
        logger.write('\n \t \t  Validate Score : {acc}\t  Best Score : {best}'.format(acc=acc, best=best_acc))
        logger.close()


from torch.optim.lr_scheduler import _LRScheduler
class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math


class FrameLoss(nn.Module):
    def __init__(self):
        super(FrameLoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, input, label):
        target = torch.zeros_like(input)
        N, T = input.size()
        for i in range(N):
            position = label[i].item() - 1
            floor, ceil = math.floor(position), math.ceil(position)
            target[i][floor] = 1
            target[i][ceil] = 1
            if floor >= 1:
                target[i][floor - 1] = 0.5
            if ceil < T - 1:
                target[i][ceil + 1] = 0.5
        if input.is_cuda:
            target = target.cuda()
        return self.loss(input, target)


class Loss_v2(nn.Module):
    def __init__(self):
        super(Loss_v2, self).__init__()

    def forward(self, input, label):
        """
        :param input: 模型的输出结果
        :param label: 标签位置比例, 0~1
        :return: 标签转换为热图平滑标签后,计算交叉熵
        """
        target = torch.zeros_like(input)
        N, T = input.size()
        for i in range(N):
            position = label[i].item() * T - 1
            floor, ceil = math.floor(position), math.ceil(position)
            # 热图平滑标签
            if floor >= 1:
                target[i][floor - 1] = 0.1
                target[i][floor] = 0.4
            else:
                target[i][floor] = 0.5
            if ceil < T - 1:
                target[i][ceil + 1] = 0.1
                target[i][ceil] = 0.4
            else:
                target[i][ceil] = 0.5

        if input.is_cuda:
            target = target.cuda()

        log_likelihood = - torch.log_softmax(input, dim=1)

        return torch.sum(torch.mul(log_likelihood, target), dim=1).mean()


if __name__ == '__main__':
    loss = nn.MSELoss(reduction='sum')
    input = torch.randn(4, 210, requires_grad=True)
    target = torch.zeros_like(input)
    output = loss(input, target)
    print(output)
