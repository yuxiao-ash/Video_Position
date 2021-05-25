import math
import torch
import torch.nn as nn


class Reg_Loss(nn.Module):
    def __init__(self, length=200):
        super(Reg_Loss, self).__init__()
        self.criterion = nn.L1Loss()
        self.length = length

    def forward(self, x, y):
        x = x * self.length
        y = y * self.length
        return self.criterion(x, y)


class Huber_Loss(nn.Module):
    def __init__(self, delta=10.0, length=200):
        super(Huber_Loss, self).__init__()
        self.delta = delta
        self.length = length

    def forward(self, x, y):
        x = x * self.length
        y = y * self.length
        residual = torch.abs(x - y)
        large_loss = 0.5 * torch.square(residual)
        small_loss = self.delta * residual - 0.5 * self.delta * self.delta
        huber_loss = torch.where(residual < self.delta, large_loss, small_loss)
        return huber_loss.mean()


class Smooth_Reg_Loss(nn.Module):
    def __init__(self, length=200, scale=1):
        super(Smooth_Reg_Loss, self).__init__()
        self.length = length
        self.criterion = nn.SmoothL1Loss()
        self.scale = scale

    def forward(self, x, y, filters=False):
        if filters:
            filter_mask = torch.abs(x - y) * self.length < 10
            x = x[filter_mask]
            y = y[filter_mask]
        x = x * self.length
        y = y * self.length
        return self.criterion(x, y) * self.scale


class Reg_Loss3(nn.Module):
    def __init__(self, length=200, scale=1, grid_num=20):
        super(Reg_Loss3, self).__init__()
        self.length = length
        self.criterion = nn.L1Loss()
        self.scale = scale
        self.grid_num = grid_num

    def forward(self, pred1, pred2, target, filters=False):
        # if filters:
        #     filter_mask = torch.abs(pred1 - target) * self.length < 10
        #     pred1 = pred1[filter_mask]
        #     pred2 = pred2[filter_mask]
        #     target = target[filter_mask]

        # 大回归损失函数
        loss1 = self.criterion(pred1 * self.length, target * self.length)

        # 小回归损失函数
        target_index = torch.floor(target * self.grid_num).long()
        pred_index = torch.floor(pred1 * self.grid_num).long()
        mask = pred_index == target_index
        small_pred = pred2[mask]
        small_target = target[mask]
        small_index = target_index[mask]
        new_target = self.grid_num * small_target - small_index
        loss2 = self.criterion(self.length / self.grid_num * small_pred, self.length / self.grid_num * new_target)
        
        if small_pred.size(0) > 0:
            loss = (loss1 + loss2) * self.scale
        else:
            loss = loss1 * self.scale
        return loss


if __name__ == '__main__':
    loss = Huber_Loss(delta=0.5, length=200)
    input = torch.randn(8, 200, requires_grad=True)#.cuda()
    target = torch.zeros_like(input)#.cuda()
    output = loss(input, target)
    print(output)
    reg = Reg_Loss()
    print(reg(input, target))
    l1 = nn.L1Loss()
    print(l1(input, target))
    l2 = nn.MSELoss()
    print(l2(input, target))
