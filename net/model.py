import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from typing import List


class Conv1dSamePadding(nn.Conv1d):
    """
    Same padding functionality with Tensorflow, inspired from https://github.com/pytorch/pytorch/issues/3867
    same padding = even padding to left/right & up/down of tensor, so that output has same dimension as the input
    """
    def forward(self, input):
        conv1d_same_padding(input, self.weight, self.bias, self.stride, self.dilation, self.groups)


# custom conv1d, see issue mentioned above
def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    kernel = weight.size(2)
    dilation = dilation[0]
    stride = stride[0]
    size = input.size(2)
    # padding = ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2
    padding = ((((size - 1) * stride) - size + (dilation * (kernel - 1))) + 1) // 2

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding, dilation=dilation, groups=groups)


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
                 num_bottleneck_channels: int, use_residual: bool = True,
                 num_dense_blocks: int = 4, dense_in_channels: int = 64,
                 output_dim: int = 1) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_bottleneck_channels = num_bottleneck_channels
        # self.kernel_sizes = kernel_sizes # These are fixed
        self.use_residual = use_residual
        self.dense_in_channels: dense_in_channels
        self.output_dim = output_dim

        # generate in- and out-channel dimensions
        inception_channels = [in_channels] + self._inception_channels(out_channels, num_inception_blocks)
        bottleneck_channels = [num_bottleneck_channels] * num_inception_blocks
        dense_channels = self._dense_channels(dense_in_channels, num_dense_blocks)

        use_residuals = self._use_residuals(num_inception_blocks)

        # Inception Time blocks
        self.inception_blocks = nn.Sequential(*[
            InceptionBlock(in_channels=inception_channels[i], out_channels=inception_channels[i + 1],
                           use_residual=use_residuals[i], bottleneck_channels=bottleneck_channels[i])
            for i in range(num_inception_blocks)
        ])

        # linear layer transforming to desired dense blocks input dimensionality
        self.linear_to_dense = nn.Linear(in_features=inception_channels[-1], out_features=dense_channels[0])

        # dense net for port specific training and to funnel inputs
        self.dense_blocks = nn.Sequential(*[
            DenseBlock(in_channels=dense_channels[i], out_channels=dense_channels[i + 1])
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

    @staticmethod
    def _use_residuals(num_of_blocks: int) -> List[bool]:
        return [True if i % 3 == 2 else False for i in range(num_of_blocks)]    # each 3rd block uses residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inception_blocks(x).mean(dim=-1)   # mean = global average pooling at the end of inception blocks
        # x = torch.cat(x, dim=-1)    # concatenate outputs of each inception block based on dimension of last block
        x = self.linear_to_dense(x)
        x = self.dense_blocks(x)
        return self.target_layer(x)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_residual: bool,
                 stride: int = 1, bottleneck_channels: int = 0) -> None:
        super().__init__()

        self.use_residual = use_residual
        self.use_bottleneck = bottleneck_channels > 0
        kernel_sizes = [10, 20, 40]

        # in- and out-channels for convolution layers
        channels = [bottleneck_channels if self.use_bottleneck else in_channels] + [out_channels] * 3

        if self.use_bottleneck:
            self.bottleneck_layer = Conv1dSamePadding(in_channels=in_channels, out_channels=bottleneck_channels,
                                                      kernel_size=1, stride=stride, bias=False)

        self.convolution_layers = nn.Sequential(*[
            Conv1dSamePadding(in_channels=channels[i], out_channels=channels[i + 1],
                              kernel_size=kernel_sizes[i], stride=stride, bias=False)
            for i in range(len(kernel_sizes))
        ])

        # accelerate Deep NN training by reducing internal covariance shift
        # https://arxiv.org/abs/1502.03167
        self.batchNorm1d = nn.BatchNorm1d(num_features=channels[-1])
        self.relu = nn.ReLU()

        if self.use_residual:
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp_x = x
        if self.use_bottleneck:
            x = self.bottleneck_layer(x)    # bottleneck applied on input
        x = self.convolution_layers(x)      # convolution layers with kernels' sizes of {10, 20, 40]
        if self.use_residual:
            x = x + self.residual(inp_x)    # residual on original input
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
