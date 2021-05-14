import torch
import torch.nn as nn
from .resnet import resnet18
import torch.nn.functional as F

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x

class VideoNet(nn.Module):
    def __init__(self):
        super(VideoNet, self).__init__()
        self.front_end = resnet18()
        self.back_end = nn.GRU(512, 1024, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.conv = nn.Sequential(nn.Conv1d(2048, 1, 1, 1), Mish())
        self.fc = nn.Sequential(nn.Linear(200, 128), Mish()) 
    
    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.size()
        x = x.view(B*T, C, H, W).contiguous()
        x = self.front_end(x)
        x = x.view(B, T, -1)
        x, _ = self.back_end(x)
        x = x.transpose(1,2)
        x = self.conv(x).squeeze(1)
        x = self.fc(x)
        return x

class SElayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SElayer,self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        hidden_size = max(16, channel // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden_size, bias=False),
            nn.ReLU(inplace = True),
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
        self.fc = nn.Sequential(nn.Linear(539, 256), Mish(), nn.Linear(256, 128), Mish())

    def forward(self, x):
        self.gru.flatten_parameters()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.transpose(1,2)
        x, _ = self.gru(x)
        x = x.transpose(1,2)
        x = self.conv5(x).squeeze(1)
        x = self.fc(x)
        return x

class AVNet(nn.Module):
    def __init__(self):
        super(AVNet, self).__init__()
        self.videoNet = VideoNet()
        self.audioNet = AudioNet()
        self.weight = nn.Parameter(torch.Tensor(128, 1))
        self.fc = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        
    def forward(self, video, audio):
        video_feature = self.videoNet(video)
        audio_feature = self.audioNet(audio)
        s = F.linear(self.weight, torch.cat([video_feature, audio_feature], axis=0))
        print(s.size())
        out = self.fc(video_feature + audio_feature)
        return out

