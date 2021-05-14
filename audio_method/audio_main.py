import os
import time
import torch
import random
import shutil
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import _LRScheduler

from dataloader import AudioDataset
from prefetcher import data_prefetcher
# from model import AudioNet, Reg_Loss
from model import AudioNet_v2, Loss_v2


# =======================================================================================================================
# =======================================================================================================================
# function
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parser = argparse.ArgumentParser()
    # train opts
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--max_epoch', type=int, default=64)
    parser.add_argument('--warm', type=int, default=1)
    parser.add_argument('--accumulation_step', type=int, default=1)
    parser.add_argument('--show_freq', type=int, default=40)
    parser.add_argument('--gpus', type=str, default='2')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--test', action='store_true')
    # load opts
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--data_root', type=str, default='/home/insomnia/Video_Position/data/')
    parser.add_argument('--weights', type=str, required=False, default=None)
    parser.add_argument('--save_prefix', type=str, default='version_2')

    args = parser.parse_args()
    print(args)
    return args


# =======================================================================================================================
# =======================================================================================================================
# train && test
def train(model, criterion, train_loader, valid_loader, args):
    best_score = 10000000000000
    for epoch in range(args.max_epoch):
        model.train()
        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        losses = []

        prefetcher = data_prefetcher(train_loader)
        batch_idx = -1
        tic = time.time()
        while batch_idx < len(train_loader) - 1:
            if epoch < args.warm:
                warmup_scheduler.step()
            batch_idx += 1
            audios, targets, _ = prefetcher.next()

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    out = model(audios)
                    loss = criterion(out, targets)
                scaler.scale(loss).backward()
                if ((batch_idx + 1) % args.accumulation_step == 0) or ((batch_idx + 1) == len(train_loader)):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                out = model(audios)
                loss = criterion(out, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            toc = time.time()
            losses.append(loss.item())
            smooth_loss = np.mean(losses[-30:])
            if batch_idx % args.show_freq == 0:
                print('epoch:{}\tloss:{:.5f}\tsmth:{:.5f}\tETA:{:.2f} minutes'.format(epoch, loss.item(), smooth_loss,
                                                                              (toc - tic) / (batch_idx + 1) * (len(
                                                                                  train_loader) - batch_idx) / 60.0))
        scheduler.step()
        torch.save({'model': model.module.state_dict()}, '{}/audionet_epoch{}.pth'.format(args.save_prefix, epoch))

        score = val(model, valid_loader, args)

        if score < best_score:
            torch.save({'model': model.module.state_dict()}, '{}/audionet_bestmodel.pth'.format(args.save_prefix))
            best_score = score


def val(model, valid_loader, args):
    model.eval()

    labels = []
    predictions = []
    names = []
    with torch.no_grad():
        prefetcher = data_prefetcher(valid_loader)
        batch_idx = -1
        while batch_idx < len(valid_loader) - 1:
            batch_idx += 1
            audios, targets, name = prefetcher.next()
            preds = model(audios)
            _, out = torch.max(preds, 1)
            out = torch.true_divide(out, preds.size(1)).unsqueeze(1)
            # print(targets)
            # print(targets.size())
            # print(out)
            # print(out.size())
            predictions.append(out.cpu().numpy())
            labels.append(targets.cpu().numpy())
            names.extend(list(name))
        predictions = np.concatenate(predictions, axis=0)
        labels = np.concatenate(labels, axis=0)
        score = get_score(predictions, labels)
        get_error_file(names, predictions, labels, os.path.join(args.save_prefix, 'error.txt'))
        print('val score: {}'.format(score))
    return score


def get_error_file(names, prediction, label, path):
    errors = np.abs(prediction - label) * 200 - 1
    files = open(path, 'w')
    for i, error in enumerate(errors):
        if error > 10:
            files.write('{},{},{}\n'.format(names[i], label[i][0] * 200, prediction[i][0] * 200))
    files.close()


def get_score(prediction, label):
    cut = np.abs(prediction - label) * 200 - 1
    cut[cut < 0] = 0
    score = np.mean(cut)
    return score


def test(model, test_loader, args):
    model.eval()

    predictions = []
    all_name = []
    with torch.no_grad():
        prefetcher = data_prefetcher(test_loader)
        batch_idx = -1
        while batch_idx < len(test_loader) - 1:
            batch_idx += 1
            audios, names = prefetcher.next()
            preds = model(audios)
            _, out = torch.max(preds, 1)
            out = torch.true_divide(out, preds.size(1)).unsqueeze(1)
            predictions.append(out.cpu().numpy())
            all_name.extend(list(names))
        predictions = np.concatenate(predictions, axis=0)

    with open(os.path.join(args.save_prefix, 'submission.csv'), 'w') as file:
        for i in range(len(all_name)):
            if 'start' in all_name[i]:
                types = 's'
            elif 'end' in all_name[i]:
                types = 'e'
            file.write('{},{},{:.3f}\n'.format(all_name[i], types, predictions[i][0] * 200))


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


# =======================================================================================================================
# =======================================================================================================================
if (__name__ == '__main__'):
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    torch.backends.cudnn.benchmark = True
    args = parse_args()

    if (not os.path.exists(args.save_prefix)):
        os.makedirs(args.save_prefix)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    dataset_train = AudioDataset(args.data_root, 'train', feature_type='mfcc', feature_lenght=8616)
    dataset_valid = AudioDataset(args.data_root, 'val', feature_type='mfcc', feature_lenght=8616)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.num_workers, pin_memory=True)
    if args.test:
        dataset_test = AudioDataset(args.data_root + 'data_A/test/', 'test')
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)

    model = AudioNet_v2().cuda()
    criterion = Loss_v2()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch - 1)
    warmup_scheduler = WarmUpLR(optimizer, len(train_loader))

    # load weight
    if args.weights is not None:
        model.load_state_dict(torch.load(args.weights)['model'])

    model = nn.DataParallel(model)

    if args.test:
        # val(model, valid_loader, args)
        test(model, test_loader, args)
    else:
        train(model, criterion, train_loader, valid_loader, args)
