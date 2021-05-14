from models.utils import AverageMeter
import time
import datetime
import torch
import os
from tqdm import tqdm
import numpy as np


def train_epoch(epoch_i, model, train_loader, criterion, optimizer, args, log, warmup_scheduler,
                train_scheduler):
    print('train the {} epoch---------'.format(epoch_i))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end_time = time.time()
    for i_batch, (image, audio, label, _) in enumerate(train_loader):
        if epoch_i < args.warm:
            warmup_scheduler.step()

        data_time.update(time.time() - end_time)
        curr_time = datetime.datetime.now()
        curr_time = datetime.datetime.strftime(curr_time, '%Y-%m-%d %H:%M:%S')
        image = image.cuda()
        audio = audio.cuda()
        label = label.cuda()
        preds = model(image, audio)
        cost = criterion(preds, label)
        model.zero_grad()
        cost.backward()
        optimizer.step()

        losses.update(cost.item(), image.size(0))
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if i_batch % args.show_frequent == 0:
            print('{0}\t'
                  'Epoch: [{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'lr {4:0.6f}'.format(curr_time,
                                       epoch_i,
                                       i_batch + 1,
                                       len(train_loader),
                                       optimizer.param_groups[0]['lr'],
                                       batch_time=batch_time,
                                       data_time=data_time,
                                       loss=losses))
    if (epoch_i + 1) % 5 == 0:
        save_file_path = os.path.join(args.work_dir, 'model')
        if not os.path.exists(save_file_path):
            os.mkdir(save_file_path)
        states = {
            'epoch': epoch_i + 1,
            'state_dict': model.state_dict()
        }
        torch.save(states, os.path.join(save_file_path, 'AV_model_epoch{}.pth'.format(epoch_i + 1)))

    train_scheduler.step()
    log.log_train(epoch_i + 1, losses.avg, optimizer)


def val(model, val_loader, args, log, i=1):
    """
    softmax 方式的 验证函数
    """
    print('validate on the Val dataset-------')

    model.eval()
    val_flie = open(os.path.join(args.work_dir, 'val_error.txt'), 'w')
    n_err = 0
    with torch.no_grad():
        for i_batch, (image, audio, label, image_name) in enumerate(val_loader):
            image = image.cuda()
            audio = audio.cuda()
            label = label.cuda()
            preds = model(image, audio)
            _, prediction = torch.max(preds, 1)
            for i in range(label.size(0)):
                err = max(abs(prediction[i].item() - label[i].item()*200) - 1, 0)
                n_err += err
                if err > 10:
                    val_flie.write(image_name[i] + '\tlabel: ' + str(label[i].item()) + '\tpred: ' + str(
                        prediction[i].item()) + '\n')

        score = n_err / float(len(val_loader.dataset))
        print('the validate score is {}'.format(score))

        if score < args.best_score:
            save_file_path = os.path.join(args.work_dir, 'model')
            if not os.path.exists(save_file_path):
                os.mkdir(save_file_path)
            states = {
                'epoch': i,
                'state_dict': model.state_dict()
            }
            torch.save(states, os.path.join(save_file_path, 'best_model.pth'))
            args.best_score = score

    log.log_val(score, args.best_score)
    val_flie.close()


def test(model, test_loader, args):
    """
    softmax方式的测试函数
    """
    model.eval()

    predictions = []
    all_name = []
    with torch.no_grad():
        for i_batch, (image, audio, names) in enumerate(test_loader):
            image = image.cuda()
            audio = audio.cuda()
            preds = model(image, audio)
            _, prediction = torch.max(preds, 1)
            # print(prediction.shape)
            predictions.append(prediction.unsqueeze(1).cpu().numpy())
            all_name.extend(list(names))
        predictions = np.concatenate(predictions, axis=0)

    with open(os.path.join(args.work_dir, 'submission.csv'), 'w') as file:
        for i in range(len(all_name)):
            if 'start' in all_name[i]:
                types = 's'
            elif 'end' in all_name[i]:
                types = 'e'
            file.write('{},{},{:.3f}\n'.format(all_name[i], types, predictions[i][0]))


def val_reg(model, val_loader, args, log, i=1):
    """
    回归 方式的 验证函数
    """
    print('validate on the Val dataset-------')

    model.eval()
    val_flie = open(os.path.join(args.work_dir, 'val_error.txt'), 'w')
    n_err = 0
    with torch.no_grad():
        for i_batch, (image, audio, label, image_name) in enumerate(val_loader):
            image = image.cuda()
            audio = audio.cuda()
            label = label.cuda()
            preds = model(image, audio)
            for i in range(label.size(0)):
                err = max(abs(preds[i].item()*200 - label[i].item()*200) - 1, 0)
                n_err += err
                if err > 10:
                    val_flie.write(image_name[i] + '\tlabel: ' + str(label[i].item()) + '\tpred: ' + str(
                        preds[i].item()) + '\n')

        score = n_err / float(len(val_loader.dataset))
        print('the validate score is {}'.format(score))

        if score < args.best_score:
            save_file_path = os.path.join(args.work_dir, 'model')
            if not os.path.exists(save_file_path):
                os.mkdir(save_file_path)
            states = {
                'epoch': i,
                'state_dict': model.state_dict()
            }
            torch.save(states, os.path.join(save_file_path, 'best_model.pth'))
            args.best_score = score

    log.log_val(score, args.best_score)
    val_flie.close()


def test_reg(model, test_loader, args):
    """
    回归方式的测试函数
    """
    model.eval()

    predictions = []
    all_name = []
    with torch.no_grad():
        for i_batch, (image, audio, names) in enumerate(test_loader):
            image = image.cuda()
            audio = audio.cuda()
            preds = model(image, audio)
            # print(preds.shape)
            predictions.append(preds.cpu().numpy())
            all_name.extend(list(names))
        predictions = np.concatenate(predictions, axis=0)

    with open(os.path.join(args.work_dir, 'submission.csv'), 'w') as file:
        for i in range(len(all_name)):
            if 'start' in all_name[i]:
                types = 's'
            elif 'end' in all_name[i]:
                types = 'e'
            file.write('{},{},{:.3f}\n'.format(all_name[i], types, predictions[i][0]*200))
