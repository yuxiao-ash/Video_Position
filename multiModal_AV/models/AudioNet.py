import math

import torch
import torch.nn as nn


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class SElayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SElayer, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        hidden_size = max(16, channel // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class TempConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TempConv, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(in_channel, out_channel, 3, 2, 1, 1), nn.BatchNorm1d(out_channel), Mish())
        self.conv2 = nn.Sequential(nn.Conv1d(in_channel, out_channel, 3, 2, 2, 2), nn.BatchNorm1d(out_channel), Mish())
        self.conv3 = nn.Sequential(nn.Conv1d(out_channel * 2, out_channel, 1), nn.BatchNorm1d(out_channel), Mish())
        self.downsample = nn.Conv1d(in_channel, out_channel, 1, 2)
        self.se = SElayer(out_channel)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = self.conv3(torch.cat([x1, x2], dim=1))
        out = self.se(out)
        out = self.downsample(x) + out
        return out


class AudioNet(nn.Module):
    def __init__(self):
        super(AudioNet, self).__init__()
        self.conv1 = TempConv(20, 64)
        self.conv2 = TempConv(64, 128)
        self.conv3 = TempConv(128, 256)
        self.conv4 = TempConv(256, 512)
        self.gru = nn.GRU(512, 1024, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.conv5 = nn.Sequential(nn.Conv1d(2048, 1, 1, 1), Mish())
        self.fc = nn.Sequential(nn.Linear(539, 256), Mish(), nn.Linear(256, 64), Mish(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        self.gru.flatten_parameters()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        x = x.transpose(1, 2)
        x = self.conv5(x).squeeze(1)
        x = self.fc(x)
        return x


class AudioNet_v2(nn.Module):
    def __init__(self):
        super(AudioNet_v2, self).__init__()
        self.conv1 = TempConv(20, 64)
        self.conv2 = TempConv(64, 128)
        self.conv3 = TempConv(128, 256)
        self.conv4 = TempConv(256, 512)
        self.conv5 = nn.Sequential(nn.Conv1d(512, 512, 3, 2, 1), nn.BatchNorm1d(512), Mish())
        self.gru = nn.GRU(512, 1024, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(2048, 1)

    def forward(self, x):
        self.gru.flatten_parameters()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.transpose(1, 2)  # N T C
        x, _ = self.gru(x)
        # x = torch.sigmoid(self.fc(x))#N 270 1
        x = self.fc(x)
        return x.squeeze(2)


# class Loss_v2(nn.Module):
#     def __init__(self):
#         super(Loss_v2, self).__init__()
#         self.loss = nn.MSELoss(reduction='sum')
#
#     def forward(self, input, label):
#         """
#         :param input: 模型的输出结果
#         :param label: 标签位置比例, 0~1
#         :return: 对270个时刻的预测做l2损失计算
#         """
#         target = torch.zeros_like(input)
#         N, T = input.size()
#         for i in range(N):
#             position = label[i].item() * T - 1
#             floor, ceil = math.floor(position), math.ceil(position)
#             target[i][floor] = 1
#             target[i][ceil] = 1
#             if floor >= 1:
#                 target[i][floor - 1] = 0.5
#             if ceil < T - 1:
#                 target[i][ceil + 1] = 0.5
#         if input.is_cuda:
#             target = target.cuda()
#         return self.loss(input, target)


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


class Reg_Loss(nn.Module):
    def __init__(self, length=200):
        super(Reg_Loss, self).__init__()
        self.criterion = nn.L1Loss()
        self.length = length

    def forward(self, x, y):
        x = x * self.length
        y = y * self.length
        return self.criterion(x, y)


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


if __name__ == '__main__':
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # torch.backends.cudnn.benchmark = True
    # model = AudioNet().cuda()
    # model = nn.DataParallel(model)
    # a = torch.zeros((24, 20, 8616)).cuda()
    # print(model(a).size())

    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # # 124.9MB
    net = AudioNet_v2()  # .cuda()
    input = torch.FloatTensor(24, 20, 8616)  # .cuda()
    output = net(input)
    print(output.shape)  # 24 270
    print_size_of_model(net)

    loss = Loss_v2()
    label = torch.rand(24)
    cost = loss(output, label)
    print(cost)
