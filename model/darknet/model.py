import os
import os.path as osp
import yaml

import torch
import torch.nn as nn


__all__ = [ "Darknet53" ]


class CNNBlock(nn.Module):
    """Basic CNN block"""
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1, inplace=True)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    """Residual with repeated blocks

    Arguments:
        channels (int): number of channels in each block
        use_residual (bool): whether perform residual connection or not
        num_repeats (int): number of repeated blocks
    """
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels//2, kernel_size=1),
                    CNNBlock(channels//2, channels, kernel_size=3, padding=1)
                    )
                ]
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x) if self.use_residual else layer(x)
        return x


class Darknet53(nn.Module):
    ARCH_PATH = osp.join(osp.dirname(osp.abspath(__file__)), 'arch.yml')
    def __init__(self, in_channels, num_classes, arch_path=None):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # Generate model architecture from config file
        self.cur_channels = in_channels
        arch_path = self.ARCH_PATH if arch_path is None else arch_path
        self.feature = self._parse_yaml(arch_path)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        # self.

    def forward(self, x):
        # Forward backbone feature extractor
        for layer in self.feature:
            x = layer(x)
        # Predict class
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _parse_yaml(self, path):
        layers = nn.ModuleList()
        in_channels = self.cur_channels # Keep track of current channels

        with open(path, 'r') as f:
            arch = yaml.full_load(f)

        for layer in arch['layers']:
            option = layer[0]
            # Basic CNN Block
            if option == 'C':
                out_channels, kernel_size, stride = layer[1:]
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0
                        )
                    )
                in_channels = out_channels
            # Residual Block
            elif option == 'B':
                num_repeats = layer[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))
            else:
                raise ValueError("Don't know how to parse '{}' type layer".format(option))

        self.cur_channels = in_channels
        return layers


if __name__ == "__main__":
    num_classes = 1000
    IMAGE_SIZE = 224
    model = Darknet53(in_channels=3, num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    print("Output Shape:", out.shape)
