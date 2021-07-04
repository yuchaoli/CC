# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import pdb
from functools import reduce
import operator
import numpy as np
from math import cos, pi

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(train_loader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda()
        target = target.cuda()
        input.requires_grad_()
        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            writer.add_scalar('loss', losses.val,
                              i + epoch * len(train_loader))
            writer.add_scalar('acc', top1.val, i + epoch * len(train_loader))
            writer.add_scalar('top5-acc', top5.val,
                              i + epoch * len(train_loader))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1,
                      top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            input.requires_grad_()

            # compute output
            torch.cuda.synchronize()
            end = time.time()
            output = model(input)
            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)

            loss = criterion(output, target)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            if i % 100 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          i,
                          len(val_loader),
                          batch_time=batch_time,
                          loss=losses,
                          top1=top1,
                          top5=top5))

        print(
            ' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {batch_time.avg:.3f}'
            .format(top1=top1, top5=top5, batch_time=batch_time))

    return top1.avg, top5.avg


def eval_loss(val_loader, model, criterion):
    losses = AverageMeter()

    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            input.requires_grad_()

            output = model(input)

            loss = criterion(output, target)
            losses.update(loss.data.item(), input.size(0))

    return losses.avg


def eval_grad(val_loader, model, criterion, parameters):
    losses = AverageMeter()

    model.eval()
    grad_one = [torch.zeros(var.size()).cuda() for var in parameters]

    for i, (input, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        input.requires_grad_()

        output = model(input)
        loss = criterion(output, target)
        grad_params_1 = torch.autograd.grad(loss, parameters)

        for j, gp in enumerate(grad_params_1):
            grad_one[j] += gp

    grad_one = [g / len(val_loader) for g in grad_one]

    return grad_one