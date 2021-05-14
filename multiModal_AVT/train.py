from models.utils import AverageMeter
import time
import datetime
import torch
import os
from tqdm import tqdm


def train_epoch(epoch_i, model, train_loader, criterion, optimizer, args, log, warmup_scheduler,
                train_scheduler):
    print('train the {} epoch---------'.format(epoch_i))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end_time = time.time()
    for i_batch, (image, label, _) in enumerate(train_loader):
        if epoch_i < args.warm:
            warmup_scheduler.step()

        data_time.update(time.time() - end_time)
        curr_time = datetime.datetime.now()
        curr_time = datetime.datetime.strftime(curr_time, '%Y-%m-%d %H:%M:%S')
        image = image.cuda()
        label = label.cuda()
        preds = model(image)
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
    if (epoch_i + 1) % 10 == 0:
        save_file_path = os.path.join(args.work_dir, 'model')
        if not os.path.exists(save_file_path):
            os.mkdir(save_file_path)
        states = {
            'epoch': epoch_i + 1,
            'state_dict': model.state_dict()
        }
        torch.save(states, os.path.join(save_file_path, 'lips_model_epoch{}.pth'.format(epoch_i + 1)))

    train_scheduler.step()
    log.log_train(epoch_i + 1, losses.avg, optimizer)


def val(model, val_loader, args, log, i=1):
    print('validate on the Val dataset-------')

    model.eval()
    val_flie = open(os.path.join(args.work_dir, 'val_error.txt'), 'w')
    n_err = 0
    with torch.no_grad():
        for i_batch, (image, label, image_name) in enumerate(val_loader):
            image = image.cuda()
            label = label.cuda()
            preds = model(image)
            _, prediction = torch.max(preds, 1)
            for i in range(label.size(0)):
                err = max(abs(prediction[i] - label[i]) - 1, 0)
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
