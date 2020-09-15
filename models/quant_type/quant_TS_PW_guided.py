# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np


# 量化比特
QUANTIZE_BIT = 8

class QuantizeTanh(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        n = math.pow(2.0, QUANTIZE_BIT) - 1
        return torch.round(i * n) / n

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs

class QuantizeGEMM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        n = math.pow(2.0, QUANTIZE_BIT) - 1
        v_max = torch.max(i)
        v_min = torch.min(i)
        scale = (v_max - v_min)/n
        scale = max(scale, 1e-8)
        zero_point = torch.round(torch.clamp(-v_min/scale, 0, n))
        quantize_val = torch.clamp(torch.round(i/scale + zero_point), 0, n)
        return (quantize_val-zero_point) * scale

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs


quantize_tanh = QuantizeTanh.apply
quantize_gemm = QuantizeGEMM.apply


def quantize_weights_bias_tanh(weight):
    tanh_w = torch.tanh(weight)
    """
    torch 关于 y = w/max(|w|) 函数在max(|w|)处梯度行为怪异该如何解释?
    tensor w ([[ 0.1229,  0.2390],
             [ 0.8703,  0.6368]])

    tensor y ([[ 0.2873,  0.2873],
             [-0.3296,  0.2873]])
    由于没有搞清楚 torch 在 max(|w|) 处如何处理的, 
    不过, 从上面看出梯度为负数, y = w/max(|w|) w>0时, 梯度为负数, 我认为是不正确的.
    为了便于处理, 这里求梯度过程中, 我们把 max(|w|) 当成一个常量来处理,
    代码中通过 Tensor.data 这样求 max(|w|) 的过程就不会加入到计算图中,
    可以看出, max_abs_w 就是一个一个常量
    """
    max_abs_w = torch.max(torch.abs(tanh_w)).data
    norm_weight = ((tanh_w / max_abs_w) + 1) / 2

    return 2 * quantize_tanh(norm_weight) - 1


def quantize_activations_tanh(activation):
    activation = torch.clamp(activation, 0.0, 1.0)
    return 2 * quantize_tanh(activation) - 1


def quantize_weights_bias_gemm(weight):
    return quantize_gemm(weight)


def quantize_activations_gemm(activation):
    return quantize_gemm(activation)


class QWConv2D(torch.nn.Conv2d):
    def __init__(self, n_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(QWConv2D, self).__init__(n_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)
        # nn.init.xavier_normal_(self.weight, 1)
        # nn.init.constant_(self.weight, 1)

    def forward(self, input):
        """
        关键在于使用函数 F.conv2d, 而不是使用模块 nn.ConV2d
        """
        qweight = quantize_weights_bias_gemm(self.weight)
        if self.bias is not None:
            qbias = quantize_weights_bias_gemm(self.bias)
        else:
            qbias = None
        return F.conv2d(input, qweight, qbias, self.stride,
                        self.padding, self.dilation, self.groups)

class QWAConv2D(torch.nn.Conv2d):
    def __init__(self, n_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(QWAConv2D, self).__init__(n_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)
        # nn.init.xavier_normal_(self.weight, 1)
        # nn.init.constant_(self.weight, 1)

    def forward(self, input):
        qweight = quantize_weights_bias_gemm(self.weight)
        if self.bias is not None:
            qbias = quantize_weights_bias_gemm(self.bias)
        else:
            qbias = None
        qinput = quantize_activations_gemm(input)
        return F.conv2d(qinput, qweight, qbias, self.stride,
                        self.padding, self.dilation, self.groups)

class QWLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, num_bits=8, num_bits_weight=None,
                 num_bits_grad=None, biprecision=False):
        super(QWLinear, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        qweight = quantize_weights_bias_gemm(self.weight)

        if self.bias is not None:
            qbias = quantize_weights_bias_gemm(self.bias)
        else:
            qbias = None

        return F.linear(input, qweight, qbias)

class QWALinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(QWALinear, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        qinput = quantize_activations_gemm(input)
        qweight = quantize_weights_bias_gemm(self.weight)

        if self.bias is not None:
            qbias = quantize_weights_bias_gemm(self.bias)
        else:
            qbias = None

        return F.linear(qinput, qweight, qbias)

"""
论文中 scalar layer 层设计 (多个 GPU )
"""

class Scalar(nn.Module):

    def __init__(self):
        super(Scalar, self).__init__()  # 这一行很重要
        # 第1种错误
        # self.scalar = torch.tensor([0.01], requires_grad=True)
        # RuntimeError: Expected object of type torch.FloatTensor
        # but found type torch.cuda.FloatTensor for argument

        # 第2种错误
        # self.scalar = torch.tensor([0.01], requires_grad=True).cuda()
        # RuntimeError: arguments are located on different GPUs

        # 第3种错误
        # self.scalar = nn.Parameter(torch.tensor(0.01, requires_grad=True))
        # RuntimeError: slice() cannot be applied to a 0-dim tensor,
        #  而加了方括号正确为 1-dim tensor

        # 第4中错误
        #  scalar = nn.Parameter(torch.tensor([0.01], requires_grad=True))
        #  self.register_buffer("scalar", scalar)
        #  scalar没有梯度更新(全是None), register_buffer 用于存储非训练参数, 如bn的平均值存储

        # 第1种方法, 可以使用
        # self.scalar = nn.Parameter(torch.tensor([0.01], requires_grad=True))

        # 第2种方法, 可以使用
        # scalar = nn.Parameter(torch.tensor([0.01], requires_grad=True))
        # self.register_parameter("scalar", scalar)

        # 根据训练经验, 设为 2.5
        self.scalar = nn.Parameter(torch.tensor([1.0], requires_grad=True, dtype=torch.float))

    def forward(self, i):
        return self.scalar * i


if __name__ == "__main__":
    qconv = QWConv2D(1, 1, 3)
    qconv.zero_grad()
    x = torch.ones(1, 1, 3, 3, requires_grad=True).float()
    y = qconv(x)
    y.backward()
    print("QConv2D 权重梯度", qconv.weight.grad)

    # 直接求梯度
    a = torch.ones(3, 3, requires_grad=True).float()
    w = nn.init.constant_(torch.empty(3, 3, requires_grad=True), 1)
    qw = quantize_weights_bias_gemm(w)

    z = (qw * a).sum()
    z.backward()
    print("求权重梯度", w.grad)

    # 验证量化梯度
    qa = quantize_weights_bias_gemm(a).sum()
    qa.backward()
    print("直接求量化权重梯度", a.grad)