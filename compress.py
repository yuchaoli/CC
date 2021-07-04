# coding: utf-8

import os
import torch
import torch.nn as nn
import argparse
import importlib
from tensorboardX import SummaryWriter
import numpy as np
import pickle

from utils import base, models

parser = argparse.ArgumentParser(description='Weight Decay Experiments')
parser.add_argument('--dataset',
                    dest='dataset',
                    help='training dataset',
                    default='cifar10',
                    type=str)
parser.add_argument('--net',
                    dest='net',
                    help='training network',
                    default='resnet20',
                    type=str)
parser.add_argument('--pretrained',
                    dest='pretrained',
                    help='whether use pretrained model',
                    default=False,
                    type=bool)
parser.add_argument('--checkpoint',
                    dest='checkpoint',
                    help='checkpoint dir',
                    default=None,
                    type=str)
parser.add_argument('--train_dir',
                    dest='train_dir',
                    help='training data dir',
                    default='tmp',
                    type=str)
parser.add_argument('--save_best',
                    dest='save_best',
                    help='whether only save best model',
                    default=False,
                    type=bool)
parser.add_argument('--save_interval',
                    dest='save_interval',
                    help='save interval',
                    default=10,
                    type=int)

parser.add_argument('--train_batch_size',
                    dest='train_batch_size',
                    help='training batch size',
                    default=64,
                    type=int)
parser.add_argument('--test_batch_size',
                    dest='test_batch_size',
                    help='test batch size',
                    default=50,
                    type=int)

parser.add_argument('--optimizer',
                    dest='optimizer',
                    help='optimizer',
                    default='sgd',
                    type=str)
parser.add_argument('--learning_rate',
                    dest='learning_rate',
                    help='learning rate',
                    default=0.1,
                    type=float)
parser.add_argument('--momentum',
                    dest='momentum',
                    help='momentum',
                    default=0.9,
                    type=float)
parser.add_argument('--weight_decay',
                    dest='weight_decay',
                    help='weight decay',
                    default=1e-4,
                    type=float)
parser.add_argument('--epochs',
                    dest='epochs',
                    help='epochs',
                    default=300,
                    type=int)
parser.add_argument('--schedule',
                    dest='schedule',
                    help='Decrease learning rate',
                    default=[150, 225],
                    type=int,
                    nargs='+')
parser.add_argument('--gamma',
                    dest='gamma',
                    help='gamma',
                    default=0.1,
                    type=float)

parser.add_argument('--com_ratio',
                    dest='com_ratio',
                    help='compression ratio',
                    default=0,
                    type=float)
parser.add_argument('--com_gamma',
                    dest='com_gamma',
                    help='compression gamma',
                    default=0.9,
                    type=float)
parser.add_argument('--com_ops',
                    dest='com_ops',
                    help='compression operation',
                    nargs='+',
                    default=['pruning', 'td'],
                    type=str)
parser.add_argument('--com_max',
                    dest='com_max',
                    help='compression maximum',
                    default=1,
                    type=float)
parser.add_argument('--com_min',
                    dest='com_min',
                    help='compression min',
                    default=0,
                    type=float)

args = parser.parse_args()

if __name__ == '__main__':
    print(args)
    model = importlib.import_module('model.model_deploy').__dict__[args.net](
        args.pretrained, args.checkpoint)
    compress_model = importlib.import_module('model.model_deploy').__dict__[
        args.net + '_compress'](args.pretrained, args.checkpoint)

    train_loader, val_loader, test_loader = importlib.import_module(
        'dataset.' + args.dataset).__dict__['load_data'](args.train_batch_size,
                                                         args.test_batch_size)

    writer = SummaryWriter(args.train_dir)
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        compress_model = compress_model.cuda()
        criterion = criterion.cuda()

    if args.net == 'resnet56':
        input_size = 32
    elif args.net == 'vggnet' or args.net == 'vggnet16':
        input_size = 32
    elif args.net == 'densenet40':
        input_size = 32
    elif args.net == 'densenet_bc_100':
        input_size = 32
    elif args.net == 'googlenet':
        input_size = 32
    elif args.net == 'vgg16':
        input_size = 224
    elif args.net == 'resnet50':
        input_size = 224

    layer_flops, model_flop, layer_inputsize = models.measure_flop_model(
        model, input_size, input_size)

    parameter = []
    parameter_name = []
    org_layers = []
    compress_layers = []
    if args.net == 'resnet56':
        for k in range(1, 4):
            for j in range(9):
                for t in range(1, 3):
                    weight = model.__getattr__(
                        'layer' + str(k))[j].__getattr__('conv' +
                                                         str(t)).weight
                    parameter.append(weight)
                    parameter_name.append('block{}-unit{}-conv{}'.format(
                        k, j + 1, t))

                    org_layer = model.__getattr__(
                        'layer' + str(k))[j].__getattr__('conv' + str(t))
                    compress_layer = compress_model.__getattr__(
                        'layer' + str(k))[j].__getattr__('conv' + str(t))
                    org_layers.append(org_layer)
                    compress_layers.append(compress_layer)

        layer_flops = layer_flops[1:-1]
        layer_inputsize = layer_inputsize[1:-1]

    elif args.net == 'vggnet':
        index = 2
        for k in range(20):
            if isinstance(model.features[k], nn.MaxPool2d) or k == 0:
                continue
            weight = model.features[k].conv1.weight
            parameter.append(weight)
            parameter_name.append('conv{}'.format(index))
            org_layers.append(model.features[k].conv1)
            compress_layers.append(model.features[k].conv1)
            index += 1
        layer_flops = layer_flops[1:-1]
        layer_inputsize = layer_inputsize[1:-1]

    elif args.net == 'vggnet16':
        index = 2
        for k in range(17):
            if isinstance(model.features[k], nn.MaxPool2d) or k == 0:
                continue
            weight = model.features[k].conv1.weight
            parameter.append(weight)
            parameter_name.append('conv{}'.format(index))
            org_layers.append(model.features[k].conv1)
            compress_layers.append(compress_model.features[k].conv1)
            index += 1

        for k in [0, 3, 6]:
            weight = model.classifier[k].weight
            parameter.append(weight)
            parameter_name.append('fc{}'.format(index))
            org_layers.append(model.classifier[k])
            compress_layers.append(compress_model.classifier[k])
            index += 1
        
        layer_flops = layer_flops[1:-1]
        layer_inputsize = layer_inputsize[1:-1]

    elif args.net == 'vgg16':
        index = 2
        for k in range(len(model.features)):
            if isinstance(model.features[k], nn.MaxPool2d) or isinstance(
                    model.features[k], nn.ReLU) or k == 0:
                continue
            weight = model.features[k].weight
            parameter.append(weight)
            parameter_name.append('conv{}'.format(index))
            org_layers.append(model.features[k])
            compress_layers.append(compress_model.features[k])
            index += 1
        layer_flops = layer_flops[1:-1]
        layer_inputsize = layer_inputsize[1:-1]

    elif args.net == 'googlenet':
        layer = ['a3', 'b3', 'a4', 'b4', 'c4', 'd4', 'e4', 'a5', 'b5']
        module = [('b1', 0), ('b2', 0), ('b2', 3), ('b3', 0), ('b3', 3), ('b3', 6), ('b4', 1)]
        for k in layer:
            for j in module:
                weight = model.__getattr__(k).__getattr__(j[0])[j[1]].weight
                parameter.append(weight)
                parameter_name.append('block[{}]-unit[{}]-conv[{}]'.format(k, j[0], j[1]))
                org_layers.append(model.__getattr__(k).__getattr__(j[0])[j[1]])
                compress_layers.append(
                    compress_model.__getattr__(k).__getattr__(j[0])[j[1]])

        layer_flops = layer_flops[1:-1]
        layer_inputsize = layer_inputsize[1:-1]

    elif args.net == 'densenet40':
        for k in range(1, 4):
            for j in range(12):
                weight = model.__getattr__('dense' + str(k))[j].conv1.weight
                parameter.append(weight)
                parameter_name.append('block{}-dense{}-conv'.format(k, j + 1))
                org_layers.append(model.__getattr__('dense' + str(k))[j].conv1)
                compress_layers.append(
                    compress_model.__getattr__('dense' + str(k))[j].conv1)
            if k <= 2:
                weight = model.__getattr__('trans' + str(k)).conv1.weight
                parameter.append(weight)
                parameter_name.append('block{}-trans{}-conv'.format(k, j + 1))
                org_layers.append(model.__getattr__('trans' + str(k)).conv1)
                compress_layers.append(
                    compress_model.__getattr__('trans' + str(k)).conv1)

        layer_flops = layer_flops[1:-1]
        layer_inputsize = layer_inputsize[1:-1]

    elif args.net == 'densenet_bc_100':
        for k in range(1, 4):
            for j in range(16):
                weight = model.__getattr__('dense' + str(k))[j].conv1.weight
                parameter.append(weight)
                parameter_name.append('block{}-dense{}-conv1'.format(k, j + 1))
                org_layers.append(model.__getattr__('dense' + str(k))[j].conv1)
                compress_layers.append(
                    compress_model.__getattr__('dense' + str(k))[j].conv1)

                weight = model.__getattr__('dense' + str(k))[j].conv2.weight
                parameter.append(weight)
                parameter_name.append('block{}-dense{}-conv2'.format(k, j + 1))
                org_layers.append(model.__getattr__('dense' + str(k))[j].conv1)
                compress_layers.append(
                    compress_model.__getattr__('dense' + str(k))[j].conv1)

            if k <= 2:
                weight = model.__getattr__('trans' + str(k)).conv1.weight
                parameter.append(weight)
                parameter_name.append('block{}-trans{}-conv'.format(k, j + 1))
                org_layers.append(model.__getattr__('trans' + str(k)).conv1)
                compress_layers.append(
                    compress_model.__getattr__('trans' + str(k)).conv1)

        layer_flops = layer_flops[1:-1]
        layer_inputsize = layer_inputsize[1:-1]

    elif args.net == 'resnet50':
        block_list = [3, 4, 6, 3]
        for k in range(1, 5):
            for j in range(block_list[k - 1]):
                for t in range(1, 4):
                    weight = model.__getattr__(
                        'layer' + str(k))[j].__getattr__('conv' +
                                                         str(t)).weight
                    parameter.append(weight)
                    parameter_name.append('block{}-unit{}-conv{}'.format(
                        k, j + 1, t))
                    org_layers.append(
                        model.__getattr__('layer' +
                                          str(k))[j].__getattr__('conv' +
                                                                 str(t)))
                    compress_layers.append(
                        compress_model.__getattr__('layer' + str(k))
                        [j].__getattr__('conv' + str(t)))

                if model.__getattr__('layer' +
                                     str(k))[j].downsample is not None:
                    weight = model.__getattr__('layer' +
                                               str(k))[j].downsample[0].weight
                    parameter.append(weight)
                    parameter_name.append('block{}-unit{}-conv{}'.format(
                        k, j + 1, 4))
                    org_layers.append(
                        model.__getattr__('layer' + str(k))[j].downsample[0])
                    compress_layers.append(
                        compress_model.__getattr__('layer' +
                                                   str(k))[j].downsample[0])
        layer_flops = layer_flops[1:-1]
        layer_inputsize = layer_inputsize[1:-1]

    grads = base.eval_grad(val_loader, model, criterion, parameter)

    # Global Compression Rate Optimization
    now_target_compress_flop = models.cal_compress_ratio(
        org_layers, grads, args.com_ratio, layer_inputsize, layer_flops, model_flop, 
        args.com_ops, parameter_name)
    
    # Multi-Step Heuristic Compression
    models.model_compress(org_layers, compress_layers, grads,
                          now_target_compress_flop, layer_inputsize,
                          args.com_ops, parameter_name, args.com_gamma,
                          compress_model, args.train_dir)

    print('---------------- Compress Finish! ------------------')

    del model
    model = compress_model

    if args.net in ['vgg16', 'resnet50']:
        model = torch.nn.DataParallel(model, [0, 1, 2, 3]).cuda()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda i: i.requires_grad,
                                           model.parameters()),
                                    args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.schedule, gamma=args.gamma)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(filter(lambda i: i.requires_grad,
                                            model.parameters()),
                                     args.learning_rate,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = None

    torch.save(model.state_dict(), args.train_dir + '/model_0.pth')

    best_acc = 0
    for i in range(args.epochs):
        base.train(train_loader, model, criterion, optimizer, i, writer)

        top1_acc, top5_acc = base.validate(test_loader, model, criterion)

        if args.optimizer == 'sgd':
            lr_scheduler.step()

        if best_acc < top1_acc:
            torch.save(model.state_dict(),
                        args.train_dir + '/model_best.pth')
            best_acc = top1_acc
        if not args.save_best:
            if (i + 1) % args.save_interval == 0 and i != 0:
                torch.save(model.state_dict(),
                           args.train_dir + '/model_{}.pth'.format(i + 1))

        writer.add_scalar('val-acc', top1_acc, i)
        writer.add_scalar('val-top5-acc', top5_acc, i)

    print('best acc: {:.2f}'.format(best_acc))
