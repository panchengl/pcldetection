# import math
# import time
# import torch
# import torch.nn as nn
# from torch.autograd import Function
# import torch.nn.functional as F
#
# class ScaleSinger(Function):
#     """
#     take a real value x, output sign(X)*E(|x|)
#     """
#     @staticmethod
#     def forward(ctx, input):
#         return torch.sign(input) * torch.mean(torch.abs(input))
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output
#
# def scale_sign(input):
#     return ScaleSinger.apply(input)
#
#
# class Quantizer(Function):
#     @staticmethod
#     def forward(ctx, input, nbit):
#         scale = 2**nbit -1
#         return torch.round(input*scale)/scale
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output, None
#
# def quantize(input, nbit):
#     return Quantizer.apply(input, nbit)
#
#
# def dorefa_w(w, nbit_w):
#     if nbit_w == 1:
#         w = scale_sign(w)
#     else:
#         w = torch.tanh(w) #make weights in [0, 1]
#         # in here, make weights quantied
#         w = w / (2 * torch.max(torch.abs(w))) + 0.5
#         w = 2 * quantize(w, nbit_w) - 1
#     return w
#
# def dorefa_a(input, nbit_a):
#     return quantize(torch.clamp(0.1*input, 0, 1), nbit_a)  # 特征A截断前先进行缩放（* 0.1），以减小截断误差
#
# class QuanConv(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size,  nbit_w=8, nbit_a=8,
#                  stride=1,padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
#         super(QuanConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
#             groups, bias)
#         self.nbit_w = nbit_w
#         self.nbit_a = nbit_a
#         self.quan_w = dorefa_w
#         self.quan_a = dorefa_a
#
#     def forward(self, input):
#         if self.nbit_w <= 32:
#             w = self.quan_w(self.weight, self.nbit_w)
#         else:
#             w = self.weight
#
#         if self.nbit_a <= 32:
#             x = self.quan_a(input, self.nbit_a)
#         else:
#             x = input
#
#         output = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
#         return output
#
#
# class Linear_Q(nn.Linear):
#     def __init__(self, in_features, out_features, bias=True, nbit_w=32, nbit_a=32):
#         super(Linear_Q, self).__init__(in_features, out_features, bias)
#         self.nbit_w = nbit_w
#         self.nbit_a = nbit_a
#         self.quan_w = dorefa_w
#         self.quan_a = dorefa_a
#
#     def forward(self, input):
#         if self.nbit_w < 32:
#             w = self.quan_w(self.weight, self.nbit_w)
#         else:
#             w = self.weight
#
#         if self.nbit_a < 32:
#             x = self.quan_a(input, self.nbit_a)
#         else:
#             x = input
#
#         output = F.linear(x, w, self.bias)
#         return output
