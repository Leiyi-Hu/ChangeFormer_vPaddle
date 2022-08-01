import math
import paddle as pd
from paddle import nn


def icnr(
    x,
    scale=2,
    init=nn.initializer.KaimingNormal(
        nonlinearity="leaky_relu")):
    """
    Checkerboard artifact free sub-pixel convolution
    https://arxiv.org/abs/1707.02937
    """
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale**2))

    tmp_zeros = pd.zeros([ni2, nf, h, w])
    init(tmp_zeros)
    k = tmp_zeros.transpose(0, 1)

    k = k.reshape(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.reshape([nf, ni, h, w]).transpose(0, 1)
    x.data.copy_(k)


class PixelShuffle(nn.Layer):
    """
    Real-Time Single Image and Video Super-Resolution
    https://arxiv.org/abs/1609.05158
    """

    def __init__(self, n_channels, scale):
        super(PixelShuffle, self).__init__()
        self.conv = nn.Conv2D(n_channels, n_channels *
                              (scale**2), kernel_size=1)
        icnr(self.conv.weight)
        self.shuf = nn.PixelShuffle(scale)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return x


def upsample(in_channels, out_channels, upscale, kernel_size=3):
    # A series of x 2 upsamling until we get to the upscale we want
    layers = []
    conv1x1 = nn.Conv2D(
        in_channels,
        out_channels,
        kernel_size=1,
        bias_attr=False)
    init_op = nn.initializer.KaimingNormal()
    init_op(conv1x1.weight.data)
    # nn.init.kaiming_normal_(conv1x1.weight.data, nonlinearity='relu')
    layers.append(conv1x1)
    for i in range(int(math.log(upscale, 2))):
        layers.append(PixelShuffle(out_channels, scale=2))
    return nn.Sequential(*layers)


class PS_UP(nn.Layer):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super(PS_UP, self).__init__()
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x):
        x = self.upsample(x)
        return x
