# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Conv2d_compress(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,
                 groups=1):
        super(Conv2d_compress, self).__init__()

        self.output_channels = output_channels
        self.input_channels = input_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        self.isbias = bias
        self.groups = groups
        self.input_mask = None
        self.keep_index = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_channels))
        else:
            self.register_parameter('bias', None)

    def __repr__(self):
        return self.__class__.__name__ \
               + "({" + str(self.input_channels) \
               + "}, {" + str(self.output_channels) \
               + "}, kernel_size={" + str(self.kernel_size) + "}, stride={" + \
               str(self.stride) + "}, padding={" + str(self.padding) + "})"

    def forward(self, input):
        if self.input_mask is not None:
            input = input*self.input_mask
        elif self.keep_index is not None:
            input = input[:, self.keep_index]
        if self.weight_num[0] == 1:
            output = F.conv2d(input,
                              self.weight,
                              self.bias,
                              stride=self.stride,
                              padding=self.padding)
        elif self.weight_num[0] == 2:
            out1 = F.conv2d(input,
                            self.weight_1,
                            None,
                            stride=self.stride,
                            padding=self.padding)
            output = F.conv2d(out1, self.weight_2, self.bias)
        else:
            output = None
            print('Error weight number: {}'.format(self.weight_num))

        return output

    def compress(self, weights, keep_index=None):
        self.weight_num = nn.Parameter(torch.Tensor([len(weights)]).cuda(),
                                       requires_grad=False)
        self.keep_index = nn.Parameter(torch.LongTensor(keep_index).cuda(),
                                       requires_grad=False)
        if len(weights) == 1:
            self.weight = nn.Parameter(weights[0])
        elif len(weights) == 2:
            self.weight_1 = nn.Parameter(weights[0])
            self.weight_2 = nn.Parameter(weights[1])
