import argparse
import shutil

import torch
import random
import numpy as np
import torch.optim as optim
from models.ResNet_GRU import FrameCls_v2 as FrameCls
from datatools.dataset import FrameDataset
from torch.utils.data import DataLoader
from train import train_epoch, val
from models.utils import LogTxt, WarmUpLR#, FrameLoss
from models.utils import FrameLoss_v2 as FrameLoss


def parse_args():
    parse = argparse.ArgumentParser(description='对每一帧识别其为片头/片尾帧的概率')
    parse.add_argument('--validate', action='store_true', help='only evaluate the checkpoint')
    parse.add_argument('--test', default=False, help='train or test')
    parse.add_argument('--work_dir', default='./work_dir/version_ce', help='the dir to save logs and models')
    parse.add_argument('--resume_from', help='the dir of the pretrained model')
    parse.add_argument('--seq_len', type=int, default=210)

    parse.add_argument('--epoch_size', type=int, default=60)
    parse.add_argument('--lr', type=float, default=0.0003)
    parse.add_argument('--bs', type=int, default=8)
    parse.add_argument('--milestone', type=int, default=30)
    parse.add_argument('--warm', default=1)

    parse.add_argument('--show_frequent', type=int, default=50)
    parse.add_argument('--best_score', type=int, default=999)

    parse.add_argument('--gpus', type=str, default='1')
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
    model = FrameCls(frameLen=args.seq_len)
    criterion = FrameLoss()

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
        # test_dataset = FrameDataset(mode='test')
        # test_loader = DataLoader(test_dataset,
        #                          batch_size=cfg.test_batch_size,
        #                          shuffle=False,
        #                          num_workers=4)
        # test(model, test_loader, cfg)
        pass
    else:
        # 训练和验证模式
        train_dataset = FrameDataset(mode='train', seq_len=args.seq_len)
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.bs,
                                  shuffle=True,
                                  num_workers=8)

        val_dataset = FrameDataset(mode='val', seq_len=args.seq_len)
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


