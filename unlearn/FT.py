import time
import torch
import utils
from .impl import iterative_unlearn


def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def FT_iter(data_loaders, model, criterion, optimizer, epoch, args, with_l1=False):
    train_loader = data_loaders["retain"]

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    for i, (inp, mask, label) in enumerate(train_loader):
        inp = inp.cuda()
        mask = mask.cuda()
        label = label.cuda()

        optimizer.zero_grad()
        # compute output
        output_clean = model(inp, mask)
        loss = criterion(output_clean, label)
        if with_l1:
            loss += args.alpha * l1_regularization(model)

        loss.backward()
        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = utils.accuracy(output.data, label)[0]

        losses.update(loss.item(), label.size(0))
        top1.update(prec1.item(), label.size(0))

        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Time {3:.2f}'.format(
                      epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


@iterative_unlearn
def FT(data_loaders, model, criterion, optimizer, epoch, args):
    return FT_iter(data_loaders, model, criterion, optimizer, epoch, args)


@iterative_unlearn
def FT_l1(data_loaders, model, criterion, optimizer, epoch, args):
    return FT_iter(data_loaders, model, criterion, optimizer, epoch, args, with_l1=True)
