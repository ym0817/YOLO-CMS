#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
from torch import nn
import numpy as np

# reference: https://github.com/Megvii-BaseDetection/YOLOX
# modify by MingZhang

class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(in_channels, mid_channels, ksize=1, stride=1, act="lrelu")
        self.layer2 = BaseConv(mid_channels, in_channels, ksize=3, stride=1, act="lrelu")

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out
'''
#原focus
class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1,
        )
        return self.conv(x)
'''
class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.focus_conv = nn.Conv2d(3, 12, 2, 2, bias = False)
        self._init_focus_conv()
        self.conv = BaseConv(in_channels *4, out_channels, ksize, stride, act = act)

    def _init_focus_conv(self):
        conv_1_weight = np.zeros((3, 2, 2))
        conv_1_weight[0, 0, 0] = 1
        conv_1_weight = np.expand_dims(conv_1_weight, 0)

        conv_2_weight = np.zeros((3, 2, 2))
        conv_2_weight[1, 0, 0] = 1
        conv_2_weight = np.expand_dims(conv_2_weight, 0)

        conv_3_weight = np.zeros((3, 2, 2))
        conv_3_weight[2, 0, 0] = 1
        conv_3_weight = np.expand_dims(conv_3_weight, 0)


        conv_4_weight = np.zeros((3, 2, 2))
        conv_4_weight[0, 1, 0] = 1
        conv_4_weight = np.expand_dims(conv_4_weight, 0)

        conv_5_weight = np.zeros((3, 2, 2))
        conv_5_weight[1, 1, 0] = 1
        conv_5_weight = np.expand_dims(conv_5_weight, 0)

        conv_6_weight = np.zeros((3, 2, 2))
        conv_6_weight[2, 1, 0] = 1
        conv_6_weight = np.expand_dims(conv_6_weight, 0)


        conv_7_weight = np.zeros((3, 2, 2))
        conv_7_weight[0, 0, 1] = 1
        conv_7_weight = np.expand_dims(conv_7_weight, 0)

        conv_8_weight = np.zeros((3, 2, 2))
        conv_8_weight[1, 0, 1] = 1
        conv_8_weight = np.expand_dims(conv_8_weight, 0)

        conv_9_weight = np.zeros((3, 2, 2))
        conv_9_weight[2, 0, 1] = 1
        conv_9_weight = np.expand_dims(conv_9_weight, 0)

        conv_10_weight = np.zeros((3, 2, 2))
        conv_10_weight[0, 1, 1] = 1
        conv_10_weight = np.expand_dims(conv_10_weight, 0)

        conv_11_weight = np.zeros((3, 2, 2))
        conv_11_weight[1, 1, 1] = 1
        conv_11_weight = np.expand_dims(conv_11_weight, 0)

        conv_12_weight = np.zeros((3, 2, 2))
        conv_12_weight[2, 1, 1] = 1
        conv_12_weight = np.expand_dims(conv_12_weight, 0)

        focus_conv_weight = np.concatenate((conv_1_weight, conv_2_weight, conv_3_weight, conv_4_weight, 
                                            conv_5_weight, conv_6_weight, conv_7_weight, conv_8_weight, 
                                            conv_9_weight, conv_10_weight, conv_11_weight, conv_12_weight), axis = 0)

        assert focus_conv_weight.shape == (12, 3, 2, 2)
        focus_conv_weight = torch.Tensor(focus_conv_weight)
        self.focus_conv.weight.data = focus_conv_weight
        self.focus_conv.weight.requires_grad = False

    def forward(self, x):
        x = self.focus_conv(x)
        x = self.conv(x)
        return x 

class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels, in_channels, ksize=ksize,
            stride=stride, groups=in_channels, act=act
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1,
            stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
            self, in_channels, out_channels, shortcut=True,
            expansion=0.5, depthwise=False, act="silu"
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
            self, in_channels, out_channels, n=1,
            shortcut=True, expansion=0.5, depthwise=False, act="silu"
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act)
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)

class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()#nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Darknet(nn.Module):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(self, depth, in_channels=3, stem_out_channels=32, out_indices=(3, 4, 5)):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output chanels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        out_features = out_indices
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
        )
        in_channels = stem_out_channels * 2  # 64

        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential(*self.make_group_layer(in_channels, num_blocks[0], stride=2))
        in_channels *= 2  # 128
        self.dark3 = nn.Sequential(*self.make_group_layer(in_channels, num_blocks[1], stride=2))
        in_channels *= 2  # 256
        self.dark4 = nn.Sequential(*self.make_group_layer(in_channels, num_blocks[2], stride=2))
        in_channels *= 2  # 512

        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)]
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu"
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x):
        outputs = []
        for idx, layer in enumerate([self.stem, self.dark2, self.dark3, self.dark4, self.dark5]):
            x = layer(x)

            if idx + 1 in self.out_features:
                outputs.append(x)
        return outputs


class CSPDarknet(nn.Module):

    def __init__(self, dep_mul=1., wid_mul=1., out_indices=(3, 4, 5), depthwise=False, act="silu"):
        super().__init__()
        self.out_features = out_indices
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2, base_channels * 2,
                n=base_depth, depthwise=depthwise, act=act,
                expansion=1.0, #concat输入通道数32倍数，for relu_yolox porting
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4, base_channels * 4,
                n=base_depth * 3, depthwise=depthwise, act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8, base_channels * 8,
                n=base_depth * 3, depthwise=depthwise, act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16, base_channels * 16, n=base_depth,
                shortcut=False, depthwise=depthwise, act=act,
            ),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def forward(self, x):
        outputs = []
        for idx, layer in enumerate([self.stem, self.dark2, self.dark3, self.dark4, self.dark5]):
            x = layer(x)

            if idx + 1 in self.out_features:
                outputs.append(x)
        return outputs


if __name__ == "__main__":
    from thop import profile

    # self.depth = 0.33
    # self.width = 0.50
    depth = 0.33
    width = 0.375
    m = CSPDarknet(dep_mul=depth, wid_mul=width, out_indices=(3, 4, 5))
    m.init_weights()
    m.eval()

    inputs = torch.rand(1, 3, 640, 640)
    # total_ops, total_params = profile(m, (inputs,))
    # print("total_ops {}G, total_params {}M".format(total_ops/1e9, total_params/1e6))
    level_outputs = m(inputs)
    for level_out in level_outputs:
        print(tuple(level_out.shape))
