import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from typing import List

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


class InceptionTimeModel(nn.Module):
    """
    num_inception_blocks:
        Number of inception block to use.
    in_features:
        Number of input features within training data
    out_channels:
        Number of output channels (hidden) of a block
    bottleneck_channels:
        Number of channels for the bottleneck layer.
    kernel_sizes:
        Size of kernels to use within each inception block
    output_dim:
        Number of output features = target
    """

    def __init__(self, num_inception_blocks: int, in_channels: int, out_channels: int,
                 bottleneck_channels: int, kernel_sizes: int, num_dense_blocks: int = 4,
                 dense_in_channels: int = 64, output_dim: int = 1) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck_channels = bottleneck_channels
        self.kernel_sizes = kernel_sizes
        self.dense_in_channels: dense_in_channels
        self.output_dim = output_dim

        # generate in- and out-channel dimensions
        inception_channels = [in_channels] + self._inception_channels(out_channels, num_inception_blocks)
        dense_channels = self._dense_channels(dense_in_channels, num_dense_blocks)

        # Inception Time blocks as first layers
        self.inception_blocks = nn.Sequential(*[
            InceptionBlock(in_channels=inception_channels[i], out_channels=inception_channels[i+1], use_resudual=False)
            for i in range(num_inception_blocks)
        ])

        # linear layer transforming to desired dense blocks input dimensionality
        self.linear_to_dense = nn.Linear(in_features=inception_channels[-1], out_features=dense_channels[0])

        # dense net for port specific training and to funnel inputs
        self.dense_blocks = nn.Sequential(*[
            DenseBlock(in_channels=dense_channels[i], out_channels=dense_channels[i+1])
            for i in range(num_dense_blocks)
        ])

        # last layer folding to target value
        self.target_layer = nn.Linear(dense_channels[-1], self.output_dim)

    @staticmethod
    def _inception_channels(out_channels: int, num_of_blocks: int) -> List[int]:
        return [out_channels] * num_of_blocks

    @staticmethod
    def _dense_channels(in_channels: int, num_of_blocks: int) -> List[int]:
        chs = np.arange(num_of_blocks + 1)
        return in_channels // (2 ** chs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inception_blocks(x).mean(dim=-1)   # apply global average pooling on each inception block
        x = torch.cat(x, dim=-1)    # concatenate outputs of each inception block based on dimension of last block
        x = self.linear_to_dense(x)
        x = self.dense_blocks(x)
        return self.target_layer(x)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_resudual: bool, stride: int = 1,
                 bottleneck_channels: int = 32) -> None:
        super().__init__()

        kernel_sizes = [10, 20, 40]

        # create in- and out-channel dimensions for each Inception Time component
        channels = [in_channels] + [out_channels] * 3

        """ Bottleneck Layer: Transform input of M dimensions to m << M dimensions
        m filters each having a length of 1 (=kernel_size) and stride of 1
        """
        # TODO: check if bias=True or bias=False (default: bias=True)
        # TODO: find good value for out_channels = number of output channels for bottleneck
        self.bottleneck_layer = SamePaddingConv1d(in_channels=in_channels, out_channels=bottleneck_channels,
                                                  kernel_size=1, stride=1, bias=False)

        """ Convolution Layers applied on output of Bottleneck Layer
        kernel sizes of 10, 20 and 40 with strides of 1
        """
        self.convolution_layers = nn.Sequential(*[
            SamePaddingConv1d(in_channels=channels[i], out_channels=channels[i + 1],
                              kernel_size=kernel_sizes[i], stride=stride, bias=False)
            for i in range(len(kernel_sizes))
        ])

        """ MaxPooling Layer on input (bottleneck)
        """
        # accelerate Deep NN training by reducing internal covariance shift
        # https://arxiv.org/abs/1502.03167
        self.max_pool_layer = nn.BatchNorm1d(num_features=channels[-1])
        # self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp_x = x
        x = self.bottleneck_layer(x)    # bottleneck applied on input
        x = self.convolution_layers(x)  # convolution layers of lengths {10, 20, 40] applied on bottleneck output
        x = x + self.max_pool_layer(inp_x)      # parallel max pooling on input
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.relu = nn.ReLU()  # nn.Selu()
        self.batchNorm1d = nn.BatchNorm1d(64)
        self.linear = nn.Linear(in_features=in_channels, out_features=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        x = self.batchNorm1d(x)
        return self.linear(x)
