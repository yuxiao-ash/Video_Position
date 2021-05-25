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

from dataloader import AVTDataset
from prefetcher import data_prefetcher
from model import AVTNet, Reg_Loss, Smooth_Reg_Loss, AVTNet2, AVTNet_YOLT, Reg_Loss3
from warmup import GradualWarmupScheduler


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
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--max_epoch', type=int, default=64)
    parser.add_argument('--use_amp', type=str2bool, default='False')
    parser.add_argument('--accumulation_step', type=int, default=1)
    parser.add_argument('--show_freq', type=int, default=30)
    parser.add_argument('--gpus', type=str, default='1,2')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--testB', action='store_true')
    parser.add_argument('--if_normal', action='store_true')
    parser.add_argument('--data_seed', type=int, default=0)
    # load opts
    parser.add_argument('--data_root', type=str, default='/home/insomnia/Video_Position/data/process_data/')
    parser.add_argument('--weights', type=str, required=False, default=None)
    parser.add_argument('--save_prefix', type=str, required=True)

    args = parser.parse_args()
    # print(args)
    return args


class Log():
    def __init__(self, file_path):
        self.file = open(file_path, 'a')

    def prints(self, inputs):
        print(inputs)
        print(inputs, file=self.file)


# =======================================================================================================================
# =======================================================================================================================
# train && test
def train(model, criterion, train_loader, valid_loader, optimizer, scheduler, args):
    best_score = 10000000000000
    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        losses = []

        prefetcher = data_prefetcher(train_loader)
        batch_idx = -1
        tic = time.time()
        while batch_idx < len(train_loader) - 1:
            batch_idx += 1
            videos, audios, targets, _, text_ROI = prefetcher.next()

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    out = model(videos, audios, text_ROI)
                    loss = criterion(out, targets)
                scaler.scale(loss).backward()
                if ((batch_idx + 1) % args.accumulation_step == 0) or ((batch_idx + 1) == len(train_loader)):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                out = model(videos, audios, text_ROI)
                loss = criterion(out, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            toc = time.time()
            losses.append(loss.item())
            smooth_loss = np.mean(losses[-30:])
            if batch_idx % args.show_freq == 0:
                log.prints(
                    'epoch:{}\tloss:{:.5f}\tsmth:{:.5f}\tETA:{:.5f}\tlr:{:.6f}'.format(epoch, loss.item(), smooth_loss,
                                                                                       (toc - tic) / (batch_idx + 1) * (
                                                                                                   len(
                                                                                                       train_loader) - batch_idx) / 3600.0,
                                                                                       optimizer.param_groups[0]['lr']))
        scheduler.step()
        if epoch > int(0.6 * args.max_epoch):
            torch.save({'epoch': epoch + 1, 'model': model.module.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()}, '{}/avnet_epoch{}.pth'.format(args.save_prefix, epoch))

        score = val(model, valid_loader, args)

        if score < best_score:
            torch.save({'model': model.module.state_dict()}, '{}/avnet_bestmodel.pth'.format(args.save_prefix))
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
            videos, audios, targets, name, text_ROI = prefetcher.next()
            out = model(videos, audios, text_ROI)
            predictions.append(out.cpu().numpy())
            labels.append(targets.cpu().numpy())
            names.extend(list(name))
        predictions = np.concatenate(predictions, axis=0)
        labels = np.concatenate(labels, axis=0)
        score = get_score(predictions, labels)
        get_error_file(names, predictions, labels, os.path.join(args.save_prefix, 'error.txt'))
        log.prints('val score: {}'.format(score))
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
            videos, audios, names, text_ROI = prefetcher.next()
            out = model(videos, audios, text_ROI)
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


# =======================================================================================================================
# =======================================================================================================================
if __name__ == '__main__':
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    torch.cuda.manual_seed_all(10)
    np.random.seed(10)
    random.seed(10)
    torch.manual_seed(10)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args = parse_args()

    if (not os.path.exists(args.save_prefix)):
        os.makedirs(args.save_prefix)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    log = Log(os.path.join(args.save_prefix, 'log.txt'))
    log.prints(args)

    dataset_train = AVTDataset(args.data_root, 'train', feature_type='mfcc', feature_lenght=8616, if_normal=args.if_normal, seed=args.data_seed)
    dataset_valid = AVTDataset(args.data_root, 'val', feature_type='mfcc', feature_lenght=8616, if_normal=args.if_normal, seed=args.data_seed)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.num_workers, pin_memory=True)
    if args.test:
        dataset_test = AVTDataset(args.data_root + 'data_A/test/', 'test', if_normal=args.if_normal)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)
    elif args.testB:
        dataset_test = AVTDataset(args.data_root + 'data_C/', 'test', if_normal=args.if_normal)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)

    # model = AVTNet().cuda()
    model = AVTNet2().cuda()
    criterion = Smooth_Reg_Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch - 1 - 5)
    warm_scheduler = GradualWarmupScheduler(optimizer, 5, scheduler)

    # load weight
    start_epoch = 0
    if args.weights is not None:
        resume_dict = torch.load(args.weights)
        model.load_state_dict(resume_dict['model'])
        if not args.test and not args.testB:
            optimizer.load_state_dict(resume_dict['optimizer'])
            warm_scheduler.load_state_dict(resume_dict['scheduler'])
            start_epoch = resume_dict['epoch']

    model = nn.DataParallel(model)

    if args.test or args.testB:
        # val(model, valid_loader, args)
        test(model, test_loader, args)
    else:
        train(model, criterion, train_loader, valid_loader, optimizer, warm_scheduler, args)
