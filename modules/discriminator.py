import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function
import torch.nn.functional as F

from math import sqrt

from modules.style_conv import EqualConv2d, FusedDownsample, EqualLinear


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            padding,
            kernel_size2=None,
            padding2=None,
            downsample=False,
            fused=False,
    ):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        self.conv1 = nn.Sequential(
            EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
            nn.LeakyReLU(0.2),
        )

        if downsample:
            if fused:
                self.conv2 = nn.Sequential(
                    FusedDownsample(out_channel, out_channel, kernel2, padding=pad2),
                    nn.LeakyReLU(0.2),
                )

            else:
                self.conv2 = nn.Sequential(
                    EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                    nn.AvgPool2d(2),
                    nn.LeakyReLU(0.2),
                )

        else:
            self.conv2 = nn.Sequential(
                EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                nn.LeakyReLU(0.2),
            )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(ConvBlock(in_channel=512,
                                            out_channel=512,
                                            kernel_size=3,
                                            padding=1,
                                            downsample=False,
                                            fused=False
                                            ), nn.BatchNorm2d(512))

        self.fromRGB = nn.Sequential(EqualConv2d(3, 512, 1))

        self.downConv = nn.Sequential(
            EqualConv2d(3, 3, 1),
            nn.AvgPool2d(2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(3))
        self.conv1 = nn.Sequential(ConvBlock(in_channel=512,
                                             out_channel=1,
                                             kernel_size=4,
                                             padding=1,
                                             downsample=True,
                                             fused=False
                                             ))

    def forward(self, x):
        ### Initial Block
        out = self.downConv(x)
        out = self.downConv(out)
        out = self.downConv(out)
        out = self.downConv(out)
        out = self.downConv(out)
        out = self.downConv(out)
        out = self.fromRGB(out)
        out = self.conv(out)
        out = self.conv1(out)
        out = torch.sigmoid(out)
        return out
