import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F

from typing import List


class Conv1dSamePadding(nn.Conv1d):
    """
    Same padding functionality with Tensorflow, inspired from https://github.com/pytorch/pytorch/issues/3867
    same padding = even padding to left/right & up/down of tensor, so that output has same dimension as the input
    - keep number of channels
    - keep temporal depth
    """
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride, self.dilation, self.groups)


# custom conv1d, see issue mentioned above
def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    kernel = weight.size(2)
    dilation = dilation[0]
    stride = stride[0]
    # print(f"kernel: {kernel} dilation: {dilation} stride: {stride}")
    size = input.size(2)    # number of rows (= channels)
    # padding = ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2
    padding = (((size - 1) * stride) - size + (dilation * (kernel - 1)) + 1)  # // 2
    # print(f"padding left and right: {padding}")
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])
        # print(f"input padded beforehand: {input.size()}")

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2, dilation=dilation, groups=groups)


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
                 bottleneck_channels: int, use_residual: bool = True,
                 num_dense_blocks: int = 4, dense_in_channels: int = 32,
                 output_dim: int = 1) -> None:
        super().__init__()

        self.num_inception_blocks = num_inception_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck_channels = bottleneck_channels
        # self.kernel_sizes = kernel_sizes # These are fixed
        self.use_residual = use_residual
        self.num_dense_blocks = num_dense_blocks
        self.dense_in_channels = dense_in_channels
        self.output_dim = output_dim

        # generate in- and out-channel dimensions
        inception_channels = [in_channels] + self._inception_channels(out_channels, num_inception_blocks)
        # print(f"inception_channels: {inception_channels}")
        expanded_bottleneck_channels = [bottleneck_channels] * num_inception_blocks
        # print(f"bottleneck_channels: {bottleneck_channels}")
        dense_channels = self._dense_channels(dense_in_channels, num_dense_blocks)
        # print(f"dense_channels: {dense_channels}")

        use_residuals = self._use_residuals(num_inception_blocks)

        # Inception Time blocks
        self.inception_blocks = nn.Sequential(*[
            InceptionBlock(in_channels=inception_channels[i], out_channels=inception_channels[i + 1],
                           use_residual=use_residuals[i], bottleneck_channels=expanded_bottleneck_channels[i])
            for i in range(num_inception_blocks)
        ])

        # linear layer transforming to desired dense blocks input dimensionality
        # self.linear_to_dense = nn.Linear(in_features=inception_channels[-1], out_features=dense_channels[0])

        # dense net for port specific training and to funnel inputs
        self.dense_blocks = nn.Sequential(*[
            DenseBlock(in_channels=dense_channels[i], out_channels=dense_channels[i + 1])
            for i in range(num_dense_blocks)
        ])

        # last layer folding to target value
        # print(f"dense_channels[-1]: {dense_channels[-1]}")
        self.target_layer = nn.Linear(dense_channels[-1], self.output_dim)

    @staticmethod
    def _inception_channels(out_channels: int, num_of_blocks: int) -> List[int]:
        return [out_channels] * num_of_blocks

    @staticmethod
    def _dense_channels(in_channels: int, num_of_blocks: int) -> List[int]:
        chs = np.arange(num_of_blocks + 1)
        print(f"num_of_blocks: {num_of_blocks}")
        print(f"chs: {chs}")
        return in_channels // (2 ** chs)

    @staticmethod
    def _use_residuals(num_of_blocks: int) -> List[bool]:
        return [True if i % 3 == 2 else False for i in range(num_of_blocks)]    # each 3rd block uses residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # required format: (N: batch-size, C: channels, L: window-width)
        # print(f"permuted tensor: {x.size()}")
        x = self.inception_blocks(x).mean(dim=-1)   # mean = global average pooling at the end of inception blocks
        # print(f"tensor after inception: {x.size()}")
        x = self.dense_blocks(x)
        # print(f"tensor after dense_blocks: {x.size()}")
        return self.target_layer(x)

    def save(self, path: str) -> None:
        print(f"Saving model at {path}")
        torch.save({
            "state_dict": self.state_dict(),
            "num_inception_blocks": self.num_inception_blocks,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "bottleneck_channels": self.bottleneck_channels,
            # self.kernel_sizes # These are fixed
            "use_residual": self.use_residual,
            "num_dense_blocks": self.num_dense_blocks,
            "dense_in_channels": self.dense_in_channels,
            "output_dim": self.output_dim
        }, path)

    @staticmethod
    def load(model_path: str, device) -> 'InceptionTimeModel':
        # see https://pytorch.org/tutorials/beginner/saving_loading_models.html
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, device)
        model = InceptionTimeModel(num_inception_blocks=checkpoint["num_inception_blocks"],
                                   in_channels=checkpoint["in_channels"],
                                   out_channels=checkpoint["out_channels"],
                                   bottleneck_channels=checkpoint["bottleneck_channels"],
                                   use_residual=checkpoint["use_residual"],
                                   num_dense_blocks=checkpoint["num_dense_blocks"],
                                   dense_in_channels=checkpoint["dense_in_channels"],
                                   output_dim=checkpoint["output_dim"])
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device)
        return model


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_residual: bool,
                 stride: int = 1, bottleneck_channels: int = 0) -> None:
        super().__init__()

        self.use_residual = use_residual
        self.use_bottleneck = bottleneck_channels > 0
        kernel_sizes = [10, 20, 40]

        # in- and out-channels for convolution layers
        # channels = [bottleneck_channels if self.use_bottleneck else in_channels] + [out_channels] * 3
        channels = [bottleneck_channels if self.use_bottleneck else in_channels] + [out_channels // 4] * 3
        if out_channels % 4 != 0:   # add desired output channels, preferably to longer term extractor
            for i in range(out_channels % 4):
                channels[-(i + 1)] += 1
        # print(f"inception conv channels: {channels}")

        if self.use_bottleneck:
            self.bottleneck_layer = Conv1dSamePadding(in_channels=in_channels, out_channels=bottleneck_channels,
                                                      kernel_size=1, stride=stride, bias=False)

        # self.convolution_layers = nn.Sequential(*[
        #     Conv1dSamePadding(in_channels=channels[i], out_channels=channels[i + 1],
        #                       kernel_size=kernel_sizes[i], stride=stride, bias=False)
        #     for i in range(len(kernel_sizes))
        # ])

        self.conv1D_10 = Conv1dSamePadding(in_channels=channels[0], out_channels=channels[1],
                                           kernel_size=kernel_sizes[0], stride=stride, bias=False)
        self.conv1D_20 = Conv1dSamePadding(in_channels=channels[0], out_channels=channels[2],
                                           kernel_size=kernel_sizes[1], stride=stride, bias=False)
        self.conv1D_40 = Conv1dSamePadding(in_channels=channels[0], out_channels=channels[3],
                                           kernel_size=kernel_sizes[2], stride=stride, bias=False)
        self.invariant_layer = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            Conv1dSamePadding(in_channels=channels[0], out_channels=channels[-1], kernel_size=1, stride=1, bias=False)
        )

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
        # print(f"inception tensor size: {x.size()}")
        inp_x = x
        if self.use_bottleneck:
            x = self.bottleneck_layer(x)    # bottleneck applied on input
        # x = self.convolution_layers(x)      # convolution layers with kernels' sizes of {10, 20, 40]
        x_10 = self.conv1D_10(x)
        x_20 = self.conv1D_20(x)
        x_40 = self.conv1D_40(x)
        # x_mean = torch.mean(inp_x)
        x_inv = self.invariant_layer(x)
        x = torch.cat((x_10, x_20, x_40, x_inv), dim=1)     # concat channel dimension
        # print(f"inception tensor size after concatenating 10, 20, 40: {x.size()}")
        if self.use_residual:
            x = x + self.residual(inp_x)    # residual on original input
            # print(f"inception tensor after residual {x.size()}")
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        # self.activation = nn.ReLU()
        self.activation = nn.SELU()
        # self.batchNorm1d = nn.BatchNorm1d(in_channels)
        self.linear = nn.Linear(in_features=in_channels, out_features=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"Dense block tensor: {x.size()}")
        x = self.activation(x)
        # x = self.batchNorm1d(x)
        return self.linear(x)
