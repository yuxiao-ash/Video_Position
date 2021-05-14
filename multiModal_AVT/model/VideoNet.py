import torch
import torch.nn as nn
# from .resnet import resnet18
from torchvision.models.resnet import resnet18


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x

class VideoNet(nn.Module):
    def __init__(self):
        super(VideoNet, self).__init__()
        resnet = resnet18(pretrained=True)
        self.front_end = nn.Sequential(*list(resnet.children())[:-1])
        self.dropout = nn.Dropout(0.5)
        self.back_end = nn.GRU(512, 1024, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.conv = nn.Sequential(nn.Conv1d(2048, 1, 1, 1), Mish())
        self.fc = nn.Sequential(nn.Linear(200, 64), Mish(), nn.Linear(64,1), nn.Sigmoid())
    
    def forward(self, x):
        self.back_end.flatten_parameters()
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        B, T, C, H, W = x.size()
        x = x.view(-1, C, H, W).contiguous()
        x = self.front_end(x)
        x = self.dropout(x)
        x = x.view(B, T, -1).contiguous()
        # x = x.float()
        x, _ = self.back_end(x)
        x = x.transpose(1,2).contiguous()
        x = self.conv(x).squeeze(1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.backends.cudnn.benchmark = True
    model = VideoNet().cuda()
    model = nn.DataParallel(model)
    a = torch.zeros((6, 250, 66, 88, 3)).cuda()
    print(model(a).size())

