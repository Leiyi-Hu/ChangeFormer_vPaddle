import paddle.nn.functional as F

import math
import paddle as pd

from paddle import nn

from math import sqrt

import random


class ConvBlock(pd.nn.Layer):
    def __init__(
            self,
            input_size,
            output_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            activation='prelu',
            norm=None):
        super(ConvBlock, self).__init__()
        self.conv = pd.nn.Conv2D(
            input_size,
            output_size,
            kernel_size,
            stride,
            padding,
            bias_attr=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = pd.nn.BatchNorm2D(output_size)
        elif self.norm == 'instance':
            self.bn = pd.nn.InstanceNorm2D(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = pd.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = pd.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = pd.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = pd.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = pd.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation != 'no':
            return self.act(out)
        else:
            return out


class DeconvBlock(pd.nn.Layer):
    def __init__(
            self,
            input_size,
            output_size,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=True,
            activation='prelu',
            norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = pd.nn.Conv2DTranspose(
            input_size,
            output_size,
            kernel_size,
            stride,
            padding,
            bias_attr=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = pd.nn.BatchNorm2D(output_size)
        elif self.norm == 'instance':
            self.bn = pd.nn.InstanceNorm2D(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = pd.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = pd.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = pd.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = pd.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = pd.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ConvLayer(nn.Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding):
        super(ConvLayer, self).__init__()
        #         reflection_padding = kernel_size // 2
        #         self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding)

    def forward(self, x):
        #         out = self.reflection_pad(x)
        out = self.conv2d(x)
        return out


class UpsampleConvLayer(pd.nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        self.conv2d = nn.Conv2DTranspose(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=1)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResidualBlock(pd.nn.Layer):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv2 = ConvLayer(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = pd.add(out, residual)
        return out


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.shape[1] * weight[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        param = module.create_parameter(
            shape=weight.shape,
            dtype=weight.dtype,
            attr=pd.ParamAttr(
                initializer=nn.initializer.Assign(
                    weight.data)))
        module.add_parameter(name + "_orig", param)
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module
