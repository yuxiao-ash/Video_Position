import math
import torch.nn as nn
import torch
from models.AudioNet import TempConv, SElayer, Mish, Reg_Loss
from models.ResNet_GRU import ResNet, BasicBlock


class AudioNet(nn.Module):
    def __init__(self):
        super(AudioNet, self).__init__()
        self.conv1 = TempConv(20, 64)
        self.conv2 = TempConv(64, 128)
        self.conv3 = TempConv(128, 256)
        self.conv4 = TempConv(256, 512)
        self.pool = nn.AdaptiveAvgPool1d(output_size=200)
        self.gru = nn.GRU(512, 1024, 2, batch_first=True, bidirectional=True, dropout=0.3)

    def forward(self, x):
        self.gru.flatten_parameters()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)  # N C T
        x = self.pool(x)
        x = x.transpose(1, 2)  # N T C
        x, _ = self.gru(x)
        return x


class VideoNet(nn.Module):
    def __init__(self, inputDim=512, hiddenDim=1024, nClasses=1, frameLen=200):
        super(VideoNet, self).__init__()
        # self.mode = mode
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.nClasses = nClasses
        self.frameLen = frameLen
        self.nLayers = 2
        # frontend3D
        self.frontend3D = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        # resnet

        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=self.inputDim)
        self.dropout = nn.Dropout(0.5)

        self.gru = nn.GRU(self.inputDim, self.hiddenDim, 2, batch_first=True, bidirectional=True, dropout=0.2)
        # initialize
        self._initialize_weights()

    def forward(self, x):
        self.gru.flatten_parameters()
        x = self.frontend3D(x)
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(-1, 64, x.size(3), x.size(4))  # 卷积部分共享权重
        x = self.resnet18(x)

        x = self.dropout(x)

        x = x.view(-1, self.frameLen, self.inputDim)
        x, _ = self.gru(x)  # N T=200 C=2048

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class AVNet(nn.Module):
    def __init__(self, use_se=True):
        super(AVNet, self).__init__()
        self.audioNet = AudioNet()
        self.videoNet = VideoNet()
        self.fc = nn.Linear(4096, 1)
        self.se = SElayer(4096)
        self.use_se = use_se

    def forward(self, frames, audios):
        x1 = self.videoNet(frames)
        x2 = self.audioNet(audios)
        x = torch.cat([x1, x2], dim=2)  # N 200 4096
        if self.use_se:
            x = x.transpose(1, 2)  # N 4096 200
            x = self.se(x)
            x = x.transpose(1, 2)
        x = self.fc(x).squeeze(2)  # N 200

        return x


class AVNet_Reg(nn.Module):
    def __init__(self, use_se=True):
        super(AVNet_Reg, self).__init__()
        self.audioNet = AudioNet()
        self.videoNet = VideoNet()
        self.fc = nn.Linear(200, 1)
        self.conv = nn.Sequential(nn.Conv1d(4096, 1, 1, 1), Mish())
        self.se = SElayer(4096)
        self.use_se = use_se

    def forward(self, frames, audios):
        x1 = self.videoNet(frames)
        x2 = self.audioNet(audios)
        x = torch.cat([x1, x2], dim=2)  # N 200 4096
        if self.use_se:
            x = x.transpose(1, 2)  # N 4096 200
            x = self.se(x)
        x = self.conv(x).squeeze(1)  # N 200
        x = torch.sigmoid(self.fc(x))  # N 1

        return x


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


if __name__ == '__main__':
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # # 121.8MB
    net = AudioNet()  # .cuda()
    input = torch.FloatTensor(12, 20, 8616)
    output = net(input)
    print(output.shape)  # 12, 200, 2048
    print_size_of_model(net)

    # 158.3MB
    net = VideoNet()
    input = torch.FloatTensor(12, 3, 200, 84, 112)  # .cuda()
    output = net(input)
    print(output.shape)  # 12 200 2048
    print_size_of_model(net)

    # 313.6MB
    net = AVNet_Reg()
    input1 = torch.FloatTensor(12, 3, 200, 84, 112)
    input2 = torch.FloatTensor(12, 20, 8616)
    output = net(input1, input2)
    print(output.shape)  # 12, 200
    print_size_of_model(net)
