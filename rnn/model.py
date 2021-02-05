import os

import torch
from torch import nn
import torch.nn.functional as F

# InceptionTime model
# inspired by Inception-v4 architecture
# https://arxiv.org/abs/1909.04939
# code: https://github.com/hfawaz/InceptionTime


# def split_sequence(sequence, n_steps):


class SamePaddingConv1d(nn.Conv1d):
    """Workaround for same padding functionality with Tensorflow
    https://github.com/pytorch/pytorch/issues/3867
    """

    def forward(self, input):
        padding = (())

        return F.conv1d(input, self.weight, self.bias, self.stride, self.dilation, self.groups)


class MyModel(nn.Module):

    def __init__(self, num_inception_blocks: int, in_channels: int, out_channels: int,
                 kernel_sizes: int, output_dim: int = 1) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.output_dim = output_dim

        # Inception Time blocks as first layers
        self.model = nn.Sequential(
            InceptionBlock(in_channels, out_channels)
        )

        self.linear = nn.Sequential(
            nn.SELU()
        )


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_resudual: bool, stride: int = 1,
                 bottleneck_channels: int = 32) -> None:
        super().__init__()

        kernel_sizes = [10, 20, 40]

        """ Bottleneck Layer: Transform input of M dimensions to m << M dimensions
        m filters each having a length of 1 (=kernel_size) and stride of 1
        """
        # TODO: check if bias=True or bias=False (default: bias=True)
        # TODO: find good value for out_channels = number of output channels for bottleneck
        self.bottleneck_layer = SamePaddingConv1d(in_channels=in_channels, out_channels=bottleneck_channels,
                                                  kernel_size=1, stride=1, bias=False)

        """ Convolution Layers applied on output of Bottleneck Layer consisting of kernels with the sizes of 10, 20 and
        40 and strides of 1
        """
        self.convolution_layers = nn.Sequential(*[
            SamePaddingConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_sizes[i],
                              stride=stride, bias=False)
            for i in range(len(kernel_sizes))
        ])

        """ MaxPooling Layer on input
        """
        self.max_pool_layer = nn.BatchNorm1d

        self.batch_norm = nn.BatchNorm1d(num_features=in_channels[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        initial_x = x
        x = self.bottleneck_layer(x)
        x = self.conv_layers(x)
        x = x + self.residual(initial_x)
        return x
