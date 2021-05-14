import argparse

import torch
import random
import numpy as np
import torch.optim as optim
from datatools.dataset import AVDataset
from models.AVNet_v2 import AVNet
# from models.AVNet_v2 import AVNet_Reg as AVNet
from torch.utils.data import DataLoader
from train import train_epoch
from train import val, test
# from train import val_reg as val
# from train import test_reg as test
from models.utils import LogTxt, WarmUpLR
from models.utils import Loss_v2 as Loss
# from models.AudioNet import Reg_Loss as Loss


def parse_args():
    parse = argparse.ArgumentParser(description='图片+音频，特征向量SE融合，softmax+cls && sigmoid+回归')
    parse.add_argument('--validate', action='store_true', help='only evaluate the checkpoint')
    parse.add_argument('--test', action='store_true', help='train or test')
    parse.add_argument('--work_dir', default='./work_dir/AV_se_1_1', help='the dir to save logs and models')
    parse.add_argument('--resume_from', help='the dir of the pretrained model')
    parse.add_argument('--seq_len', type=int, default=200)
    parse.add_argument('--use_se', type=bool, default=True)

    parse.add_argument('--epoch_size', type=int, default=65)
    parse.add_argument('--lr', type=float, default=0.0003)
    parse.add_argument('--bs', type=int, default=16)
    parse.add_argument('--milestone', type=int, default=30)
    parse.add_argument('--warm', default=1)

    parse.add_argument('--show_frequent', type=int, default=50)
    parse.add_argument('--best_score', type=int, default=999)

    parse.add_argument('--gpus', type=str, default='1,2')
    return parse.parse_args()


if __name__ == '__main__':
    # 读取命令行参数
    args = parse_args()
    print('training config:', args)
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # 读取配置文件
    if not os.path.exists(args.work_dir):
        os.mkdir(args.work_dir)

    # 设置随机种子
    manualSeed = 10
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    torch.backends.cudnn.benchmark = True

    # 配置模型和loss
    model = AVNet(use_se=args.use_se)
    criterion = Loss()

    model = torch.nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0001)

    # 打开日志文件
    log = LogTxt(path=args.work_dir, description='=='*10)

    # 加载预训练参数
    start_epoch = 0
    if args.resume_from is not None:
        print('loading pretrained model from %s' % args.resume_from)
        checkpoint = torch.load(args.resume_from)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        start_epoch = checkpoint['epoch']

    if args.test:
        # 测试模式
        test_dataset = AVDataset(mode='test')
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.bs,
                                 shuffle=False,
                                 num_workers=4)
        test(model, test_loader, args)
        pass
    else:
        # 训练和验证模式
        train_dataset = AVDataset(mode='train')
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.bs,
                                  shuffle=True,
                                  num_workers=8)

        val_dataset = AVDataset(mode='val')
        val_loader = DataLoader(val_dataset,
                                batch_size=args.bs * 2,
                                shuffle=False,
                                num_workers=8,
                                drop_last=False)

        # 配置训练策略
        iter_per_epoch = len(train_loader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch)
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[args.milestone],
                                                         gamma=0.1)

        for i in range(0, args.epoch_size):
            if args.validate:
                val(model, val_loader, args, log, i)
                break
            else:
                train_epoch(i, model, train_loader, criterion, optimizer, args, log, warmup_scheduler, train_scheduler)
                val(model, val_loader, args, log, i)


