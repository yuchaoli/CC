# coding: utf-8

import pdb
from operator import mul
from functools import reduce
import operator
from collections import OrderedDict, namedtuple
import functools
import itertools
import random
import math
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import multiprocessing as mp
from multiprocessing import Pool, Manager

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import leastsq

_PER_FLOPS = 10**7

count_ops = []
layer_inputsize = []
flop_ops = 0

def get_num_gen(gen):
    return sum(1 for x in gen)


def is_leaf(model):
    return get_num_gen(model.children()) == 0


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name


def compress(model, com_ratio=0):
    for child in model.children():
        if is_leaf(child):
            if get_layer_info(child) in [
                    'Conv2d_Pruning', 'Conv2d_TD', 'Conv2d_Quant'
            ]:
                child.compress(com_ratio)
                print(child, 'Compression finish!')
        else:
            compress(child, com_ratio=com_ratio)

def cal_metric(org_weight,
                now_weight,
                grad,
                com_ops,
                pruning_index=[],
                eigenvalue_zero_num=[],
                com_gamma=0.9):

    num = now_weight.shape[1] - len(pruning_index) + min(
        now_weight.shape[0], reduce(
            mul, now_weight.shape[1:])) - eigenvalue_zero_num

    metrics = []
    for j in range(len(com_ops)):
        if com_ops[j] == 'pruning':
            for k in range(now_weight.shape[1]):
                if k not in pruning_index:
                    keep_index = [
                        i for i in range(org_weight.shape[1])
                        if i not in pruning_index and i != k
                    ]

                    this_weight_k = now_weight[:, k].data.cpu().numpy()
                    now_weight_k = now_weight
                    now_weight_k[:, k] = 0
                    backward_metric = torch.sum(
                        torch.pow(grad * (now_weight_k - org_weight), 2))

                    forward_metric = torch.sum(
                        torch.pow(grad * (now_weight_k - org_weight), 2) +
                        2 / (num - 1) * (2 * (grad * grad *
                                              (now_weight_k - org_weight) *
                                              -now_weight_k)))

                    try:
                        u, s, v = torch.svd(now_weight_k.cpu().reshape(
                            now_weight_k.shape[0], -1))
                        u = torch.pow(u, 2).cuda()
                        s = torch.pow(s, 2).cuda()
                        v = torch.pow(v, 2).cuda()
                        new_weight = torch.mm(torch.mm(u, torch.diag(s)),
                                              v.t())
                        new_weight = new_weight.reshape(now_weight.size())
                        forward_metric += torch.sum(
                            (torch.pow(grad * now_weight_k, 2) +
                             (torch.pow(grad, 2) *
                              new_weight))) / (num - 1)
                    except:
                        forward_metric += torch.sum(
                            (torch.pow(grad * now_weight_k,
                                       2))) * 2 / (num - 1)

                    forward_metric = com_gamma * forward_metric
    
                    metrics.append(
                        ('pruning', k,
                         backward_metric.item() + forward_metric.item()))
                    now_weight[:, k] = torch.from_numpy(this_weight_k).cuda()
                    del this_weight_k

        elif com_ops[j] == 'td':
            try:
                u, s, v = torch.svd(now_weight.cpu().reshape(
                    now_weight.shape[0], -1))
            except:
                continue
            u = u.cuda()
            s = s.cuda()
            v = v.cuda()
            use_rank_num = min(
                now_weight.shape[0], reduce(
                    mul, now_weight.shape[1:])) - eigenvalue_zero_num
            for k in range(use_rank_num):
                org_value = s[k].item()
                s[k] = 0
                now_weight_k = torch.mm(torch.mm(u, torch.diag(s)), v.t())
                now_weight_k = now_weight_k.reshape(now_weight.size())
                backward_metric = torch.sum(
                    torch.pow(grad * (now_weight_k - org_weight), 2))

                keep_index = [
                    i for i in range(org_weight.shape[1])
                    if i not in pruning_index
                ]

                forward_metric = torch.sum(
                    torch.pow(grad *
                              (now_weight_k - org_weight), 2) +
                    2 / (num - 1) * (2 * (grad * grad *
                                          (now_weight_k - org_weight) *
                                          -now_weight_k)))

                try:
                    un = torch.pow(u, 2)
                    sn = torch.pow(s, 2)
                    vn = torch.pow(v, 2)
                    new_weight = torch.mm(torch.mm(un, torch.diag(sn)), vn.t())
                    new_weight = new_weight.reshape(now_weight.size())
                    forward_metric += torch.sum(
                        (torch.pow(grad * now_weight_k, 2) +
                         (torch.pow(grad, 2) * new_weight))) / (
                             num - 1)
                except:
                    forward_metric += torch.sum(
                        (torch.pow(grad * now_weight_k,
                                   2))) * 2 / (num - 1)

                forward_metric = com_gamma * forward_metric

                metrics.append(
                    ('td', k, backward_metric.item() + forward_metric.item()))
                s[k] = org_value

    return metrics

def get_layer_param(model):
    return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])


def get_conv_flop(layer, x):
    out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                layer.stride[0] + 1)
    out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                layer.stride[1] + 1)
    delta_params = get_layer_param(layer)
    delta_ops = delta_params * out_h * out_w
    return delta_ops


def get_fc_flop(layer, x):
    delta_params = get_layer_param(layer)
    delta_ops = x.size()[0] * delta_params
    return delta_ops


def measure_flop_layer(layer, x):
    global count_ops, flop_ops, layer_inputsize
    delta_ops = 0
    type_name = get_layer_info(layer)

    if type_name in ['Conv2d']:
        delta_ops = get_conv_flop(layer, x)
        print(layer, delta_ops)

    elif type_name in ['Linear']:
        delta_ops = get_fc_flop(layer, x)
        print(layer, delta_ops)

    elif type_name in ['BasicBlock_Compress']:
        pruning_ratio = len(
            layer.conv2.keep_index) / layer.conv2.input_channels
        delta_ops = get_conv_flop(layer.conv1, x) * pruning_ratio
        delta_ops += get_conv_flop(layer.conv2, x)
        print(delta_ops)

    elif type_name in [
            'ReLU', 'ReLU6', 'Sigmoid', 'AvgPool2d', 'MaxPool2d',
            'AdaptiveAvgPool2d', 'BatchNorm2d', 'Dropout2d', 'DropChannel',
            'Dropout', 'Sequential', 'LambdaLayer'
    ]:
        pass

    # unknown layer type
    else:
        raise TypeError('unknown layer type: %s' % type_name)

    if delta_ops != 0:
        flop_ops += delta_ops / _PER_FLOPS
        if type_name in ['Conv2d'] and layer.groups != 1:
            return
        count_ops.append(delta_ops / _PER_FLOPS)
        if len(x.shape) == 4:
            layer_inputsize.append((x.shape[2], x.shape[3]))
        else:
            layer_inputsize.append((x.shape[1]))
    return


def measure_flop_model(model, H, W):
    global count_ops, flop_ops, layer_inputsize
    count_ops = []
    layer_inputsize = []
    flop_ops = 0
    data = torch.zeros(1, 3, H, W).cuda()

    def should_measure(x):
        return is_leaf(x)

    def modify_forward(model):
        for child in model.children():
            type_name = get_layer_info(child)
            if should_measure(child) or type_name in ['BasicBlock_Compress']:

                def new_forward(m):
                    def lambda_forward(x):
                        measure_flop_layer(m, x)
                        return m.old_forward(x)

                    return lambda_forward

                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            # leaf node
            type_name = get_layer_info(child)
            if (should_measure(child)
                    or type_name in ['BasicBlock_Compress']) and hasattr(
                        child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data)
    restore_forward(model)

    return count_ops, flop_ops, layer_inputsize

def param_all_compress(layer,
                       weight,
                       grad,
                       target_compress_flop,
                       out_h,
                       out_w,
                       org_flops,
                       com_ops,
                       com_gamma=0.9):

    pruning_index = []
    eigenvalue_zero_num = 0
    rank = min(weight.shape[0], reduce(mul, weight.shape[1:]))
    new_weight = torch.clone(weight)
    use_ops = 'all'

    interval = int((weight.shape[1] + rank) * 0.01)
    if interval == 0:
        interval = 1
    
    last_flops = org_flops
    while True:
        if target_compress_flop == 0:
            break
        
        # calculate metric
        compress_metrics = cal_metric(
            weight, new_weight, grad, com_ops, pruning_index,
            eigenvalue_zero_num, com_gamma)

        com_metrics = []
        for i in range(len(compress_metrics)):
            com_metrics.append(compress_metrics[i][2])
        indexs = np.argsort(com_metrics)

        is_stop = False
        eigen_value_index = [j for j in range(rank)]
        except_num = 0
        for i in range(interval):
            i = i - except_num
            index = indexs[i]

            if compress_metrics[index][0] == 'pruning':
                pruning_index.append(compress_metrics[index][1])
                keep_index = [
                    i for i in range(new_weight.shape[1])
                    if i not in pruning_index
                ]

                new_weight[:, compress_metrics[index][1]] = 0

            elif compress_metrics[index][0] == 'td':
                compress_index = compress_metrics[index][1]
                now_shape = new_weight.shape
                try:
                    u, s, v = torch.svd(new_weight.cpu().reshape(
                        new_weight.shape[0], -1))
                    s[eigen_value_index[compress_index]] = 0
                    new_weight = torch.mm(torch.mm(u, torch.diag(s)), v.t())
                    new_weight = new_weight.reshape(now_shape)
                    new_weight = new_weight.cuda()
                    eigenvalue_zero_num += 1
                    for k in range(compress_index + 1, rank):
                        eigen_value_index[k] = eigen_value_index[k - 1]
                except:
                    now_i = i
                    while now_i < len(indexs) and compress_metrics[
                            index][0] != 'pruning':
                        now_i += 1
                        index = indexs[now_i]

                    if now_i >= len(indexs):
                        print("error can not svd!")
                        is_stop = True
                        break
                    
                    pruning_index.append(compress_metrics[index][1])
                    keep_index = [
                        j for j in range(new_weight.shape[1])
                        if j not in pruning_index
                    ]
                    new_weight[:, compress_metrics[index][1]] = 0
                    indexs = np.delete(indexs, [now_i], 0)
                    except_num += 1

            if eigenvalue_zero_num == 0:
                now_flops = (reduce(operator.mul, new_weight.size(), 1) -
                             len(pruning_index) * new_weight.shape[0] *
                             new_weight.shape[2] *
                             new_weight.shape[3]) * out_h * out_w / _PER_FLOPS
            else:
                now_flops = ((new_weight.shape[1] - len(pruning_index)) *
                             new_weight.shape[2] * new_weight.shape[3] *
                             (rank - eigenvalue_zero_num) +
                             (rank - eigenvalue_zero_num) *
                             new_weight.shape[0]) * out_h * out_w / _PER_FLOPS

            last_flops = now_flops

            if org_flops - now_flops >= target_compress_flop:
                is_stop = True
                break

            if len(pruning_index) != 0 and eigenvalue_zero_num != 0:
                pruning_flops = (reduce(operator.mul, new_weight.size(), 1) -
                                 len(pruning_index) * new_weight.shape[0] *
                                 new_weight.shape[2] * new_weight.shape[3]
                                 ) * out_h * out_w / _PER_FLOPS

                if org_flops - pruning_flops >= target_compress_flop:
                    use_ops = 'pruning'
                    is_stop = True
                    break

        if is_stop:
            break

    new_weight = torch.clone(weight)
    new_weight[:, pruning_index] = 0
    new_weight = torch.Tensor(
        np.delete(new_weight.data.cpu().numpy(), pruning_index, 1)).cuda()
    keep_index = [i for i in range(weight.shape[1]) if i not in pruning_index]

    if eigenvalue_zero_num == 0 or use_ops == 'pruning':
        return [new_weight], keep_index
    else:
        u, s, v = torch.svd(new_weight.reshape(new_weight.shape[0], -1))
        weight_1 = torch.mm(
            torch.sqrt(torch.diag(s))[:rank - eigenvalue_zero_num],
            v.t()).reshape(-1, new_weight.shape[1], new_weight.shape[2],
                        new_weight.shape[3])
        weight_2 = torch.mm(
            u,
            torch.sqrt(torch.diag(s))[:, :rank - eigenvalue_zero_num]).reshape(
                new_weight.shape[0], -1, 1, 1)
        return [weight_1, weight_2], keep_index


def param_compress(layer,
                   weight,
                   grad,
                   target_compress_flop,
                   input_size,
                   com_ops,
                   com_gamma=0.9):
    if len(weight.shape) == 4:
        out_h = int((input_size[0] + 2 * layer.padding[0] -
                     layer.kernel_size[0]) / layer.stride[0] + 1)
        out_w = int((input_size[1] + 2 * layer.padding[1] -
                     layer.kernel_size[1]) / layer.stride[1] + 1)
        delta_params = reduce(operator.mul, weight.size(), 1)
        org_flops = delta_params * out_h * out_w / _PER_FLOPS

    new_weights_all, keep_index_all = param_all_compress(
        layer, weight, grad, target_compress_flop, out_h, out_w, org_flops,
        com_ops, com_gamma)

    return new_weights_all, keep_index_all


def model_compress(org_layers,
                   compress_layers,
                   grads,
                   target_compress_flops,
                   layer_input_sizes,
                   com_ops,
                   layer_names,
                   com_gamma=0.9,
                   compress_model=None,
                   save_dir=None):

    for i in range(len(org_layers)):
        org_layer = org_layers[i]
        compress_layer = compress_layers[i]
        grad = grads[i]
        target_compress_flop = target_compress_flops[i]
        layer_input_size = layer_input_sizes[i]
        layer_name = layer_names[i]

        new_weights, keep_index = param_compress(
            org_layer, org_layer.weight, grad, target_compress_flop,
            layer_input_size, com_ops, com_gamma)
        compress_layer.compress(new_weights, keep_index)

        print(layer_name, [w.shape for w in new_weights])


def cal_compress_ratio(org_layers, grads, compress_ratio, layer_input_sizes,
                       layer_flops, network_flops, com_ops, layer_names):

    layers_compress_reduncy = []
    layers_reduncy_model = []
    layers_reduncy_model_param = []
    for i in range(len(org_layers)):
        weight = org_layers[i].weight
        grad = grads[i]
        layer = org_layers[i]
        input_size = layer_input_sizes[i]
        layer_flop = layer_flops[i]
        channel_num = weight.shape[1]
        rank_num = min(weight.shape[0], reduce(mul, weight.shape[1:]))

        if len(weight.shape) == 4:
            out_h = int((input_size[0] + 2 * layer.padding[0] -
                         layer.kernel_size[0]) / layer.stride[0] + 1)
            out_w = int((input_size[1] + 2 * layer.padding[1] -
                         layer.kernel_size[1]) / layer.stride[1] + 1)
            delta_params = reduce(operator.mul, weight.size(), 1)
            org_flops = delta_params * out_h * out_w / _PER_FLOPS
        elif len(weight.shape) == 2:
            delta_params = reduce(operator.mul, weight.size(), 1)
            org_flops = delta_params / _PER_FLOPS

        now_metrics = []
        # Pruning
        for j in range(channel_num):
            metric = torch.sum(torch.pow(grad[:, j] * weight[:, j], 2))
            now_metrics.append(('pruning', j, metric.item()))

        # TD
        u, s, v = torch.svd(weight.reshape(weight.shape[0], -1))
        for j in range(rank_num):
            org_value = s[j].item()
            s[j] = 0
            new_weight = torch.mm(torch.mm(u, torch.diag(s)), v.t())
            new_weight = new_weight.reshape(weight.size())
            metric = torch.sum(torch.pow(grad * (new_weight - weight), 2))
            now_metrics.append(('td', j, metric.item()))
            s[j] = org_value

        # sort compression units
        metrics_sort = sorted(now_metrics, key=lambda x: x[2])

        now_weight = weight.clone()
        metrics = []
        crs = []
        eigenvalue_zero_num = 0
        channel_zero_num = 0
        eigen_value_index = [j for j in range(rank_num)]
        for j, ms in enumerate(metrics_sort):
            if ms[0] == 'pruning':
                now_weight[:, ms[1]] = 0
                channel_zero_num += 1
            elif ms[0] == 'td':
                try:
                    u, s, v = torch.svd(now_weight.cpu().reshape(
                        now_weight.shape[0], -1))
                except:
                    continue
                s[eigen_value_index[ms[1]]] = 0
                for k in range(ms[1] + 1, rank_num):
                    eigen_value_index[k] -= 1
                now_weight = torch.mm(torch.mm(u, torch.diag(s)), v.t())
                now_weight = now_weight.reshape(weight.size())
                now_weight = now_weight.cuda()
                eigenvalue_zero_num += 1

            metric = torch.sum(torch.pow(grad * (now_weight - weight), 2))
            metrics.append(metric.item())

            if len(weight.shape) == 4:
                if eigenvalue_zero_num == 0:
                    now_flops = (
                        reduce(operator.mul, weight.size(), 1) -
                        channel_zero_num * weight.shape[0] * weight.shape[2] *
                        weight.shape[3]) * out_h * out_w / _PER_FLOPS
                else:
                    now_flops = ((weight.shape[1] - channel_zero_num) *
                                weight.shape[2] * weight.shape[3] *
                                (rank_num - eigenvalue_zero_num) +
                                (rank_num - eigenvalue_zero_num) *
                                weight.shape[0]) * out_h * out_w / _PER_FLOPS
            else:
                if eigenvalue_zero_num == 0:
                    now_flops = (
                        reduce(operator.mul, weight.size(), 1) -
                        channel_zero_num * weight.shape[0])/ _PER_FLOPS
                else:
                    now_flops = ((weight.shape[1] - channel_zero_num) *
                                (rank_num - eigenvalue_zero_num) +
                                (rank_num - eigenvalue_zero_num) *
                                weight.shape[0])/ _PER_FLOPS
            crs.append((org_flops - now_flops) / org_flops)

        indexs = np.argsort(crs)
        crs = np.sort(np.array(crs))
        metrics = np.array(metrics)[indexs]

        metrics = metrics[crs > 0]
        crs = crs[crs > 0]
        metrics = metrics / np.max(metrics)

        layers_compress_reduncy.append((crs, metrics))

        # fit by exponential model
        LR_model = LinearRegression()
        LR_model.fit(crs.reshape(-1, 1),
                     np.log(metrics.reshape(-1, 1) + 1e-10), np.sqrt(metrics))

        layers_reduncy_model.append(LR_model)

        p1 = math.exp(LR_model.intercept_[0])
        p2 = LR_model.coef_[0][0]
        layers_reduncy_model_param.append((p1, p2))

        print("{}:{}exp^({}x)".format(layer_names[i], p1, p2))

    def predict_flop(p, x):
        out = 0
        for i in range(len(p)):
            if p[i][1] * x >= 1:
                if p[i][0] * math.log(p[i][1] * x) < 0:
                    out += 0
                elif p[i][0] * math.log(p[i][1] * x) > 0.95 * p[i][2]:
                    out += 0.9 * p[i][2]
                else:
                    out += p[i][0] * math.log(p[i][1] * x)
            else:
                out += 0
        return out

    def grad_flop(p, x, y):
        p_y = predict_flop(p, x)
        out = 2 * (p_y - y)
        s = 0
        for i in range(len(p)):
            if p[i][1] * x >= 1:
                if p[i][0] * math.log(p[i][1] * x) < 0:
                    s += 0
                elif p[i][0] * math.log(p[i][1] * x) > 0.95 * p[i][2]:
                    s += 0
                else:
                    s += p[i][0]
            else:
                s += 0
        out = out * s
        return out

    def sgd_learning(p, x, y, learning_rate=1e-6, iters=1000000):
        for i in range(iters):
            grad = grad_flop(p, x, y)
            x = x - learning_rate * grad
            ny = predict_flop(p, x)
            if i % 100 == 0:
                print("[{}] Loss:{}".format(i, pow(ny - y, 2)))
            if pow(ny - y, 2) < 1e-10:
                break
        print("sgd final loss:{}".format(pow(ny - y, 2)))
        return x

    # compression rate decision algorithm
    params = []
    for i in range(len(org_layers)):
        p1 = layers_reduncy_model_param[i][0]
        p2 = layers_reduncy_model_param[i][1]
        params.append((layer_flops[i] / p2, 1 / (p1 * p2), layer_flops[i]))

    reduncy_ans = sgd_learning(params, 0.1, network_flops * compress_ratio)

    layer_compress_flops = []
    for i in range(len(org_layers)):
        p1 = layers_reduncy_model_param[i][0]
        p2 = layers_reduncy_model_param[i][1]
        cf = layer_flops[i] / p2 * math.log(1 / (p1 * p2) * reduncy_ans)
        if cf < 0:
            cf = 0
        elif cf > layer_flops[i] * 0.95:
            cf = layer_flops[i] * 0.95
        print("{}:{}".format(layer_names[i], cf))
        layer_compress_flops.append(cf)

    return layer_compress_flops