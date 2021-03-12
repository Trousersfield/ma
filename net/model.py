import os

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
    def __init__(self, num_inception_blocks: int, in_channels: int, out_channels: int,
                 kernel_sizes: int, output_dim: int = 1) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.output_dim = output_dim

        # create in- and out-channel dimensions for each block
        inception_channels = [in_channels] + self._block_channels(in_channels, num_inception_blocks)

        # Inception Time blocks as first layers
        self.inception_blocks = nn.Sequential(*[
            InceptionBlock(inception_channels[i], inception_channels[i+1], use_resudual=False)
            for i in range(num_inception_blocks)
        ])

        # net to funnel inputs to desired output
        self.funnel_net = nn.Sequential(
            nn.Linear(in_features=inception_channels[-1], out_features=64),
            nn.ReLU(),  # nn.Selu()
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, self.output_dim)
        )

    @staticmethod
    # value: Union[int, bool, List[int], List[bool]]
    # Irgendeiner der Werte in Union[...] muss als Input gegeben werden
    def _block_channels(channels: int, num_of_blocks: int) -> List[int]:
        result = [channels] * num_of_blocks
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply global average pooling on each inception block
        x = self.inception_blocks(x).mean(dim=-1)
        # concatenate outputs of each inception block based on dimension of last block
        x = torch.cat(x, dim=-1)
        return self.funnel_net(x)


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
            SamePaddingConv1d(in_channels=channels[i], out_channels=channels[i], kernel_size=kernel_sizes[i],
                              stride=stride, bias=False)
            for i in range(len(kernel_sizes))
        ])

        """ MaxPooling Layer on input (bottleneck)
        """
        # accelerate Deep NN training by reducing internal covariate shift
        # https://arxiv.org/abs/1502.03167
        self.max_pool_layer = nn.BatchNorm1d(num_features=channels[-1])
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bottleneck_layer(x)
        x = self.conv_layers(x)
        x = self.max_pool_layer(x)
        return self.relu(x)
