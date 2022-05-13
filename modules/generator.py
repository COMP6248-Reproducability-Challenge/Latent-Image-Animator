import torch
from torch import nn
from torch.nn import functional as F

from modules.style_conv import flow_warp, EqualConv2d, StyledConvBlock
from modules.utils import get_device


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.style_conv = nn.Sequential(StyledConvBlock(in_channel=512,
                                                        out_channel=512,
                                                        kernel_size=3,
                                                        style_dim=512,
                                                        upsample=False,
                                                        fused=False,
                                                        initial=True))
        self.style_conv1 = nn.Sequential(StyledConvBlock(in_channel=512,
                                                         out_channel=512,
                                                         kernel_size=3,
                                                         style_dim=512,
                                                         upsample=True,
                                                         fused=False))
        self.style_conv2 = nn.Sequential(StyledConvBlock(in_channel=512,
                                                         out_channel=512,
                                                         kernel_size=3,
                                                         style_dim=512,
                                                         upsample=True,
                                                         fused=False))
        self.style_conv3 = nn.Sequential(StyledConvBlock(in_channel=512,
                                                         out_channel=512,
                                                         kernel_size=3,
                                                         style_dim=512,
                                                         upsample=True,
                                                         fused=False))
        self.style_conv4 = nn.Sequential(StyledConvBlock(in_channel=512,
                                                         out_channel=256,
                                                         kernel_size=3,
                                                         style_dim=512,
                                                         upsample=True,
                                                         fused=False))
        self.style_conv5 = nn.Sequential(StyledConvBlock(in_channel=256,
                                                         out_channel=128,
                                                         kernel_size=3,
                                                         style_dim=512,
                                                         upsample=True,
                                                         fused=False))
        self.style_conv6 = nn.Sequential(StyledConvBlock(in_channel=128,
                                                         out_channel=64,
                                                         kernel_size=3,
                                                         style_dim=512,
                                                         upsample=True,
                                                         fused=False
                                                         ))
        self.style_conv7 = nn.Sequential(StyledConvBlock(in_channel=128,
                                                         out_channel=64,
                                                         kernel_size=3,
                                                         style_dim=512,
                                                         upsample=True,
                                                         fused=False))
        self.toRGB = nn.Sequential(EqualConv2d(512, 3, 1))
        self.toRGB1 = nn.Sequential(EqualConv2d(512, 3, 1))
        self.toRGB2 = nn.Sequential(EqualConv2d(512, 3, 1))
        self.toRGB3 = nn.Sequential(EqualConv2d(512, 3, 1))
        self.toRGB4 = nn.Sequential(EqualConv2d(256, 3, 1))
        self.toRGB5 = nn.Sequential(EqualConv2d(128, 3, 1))
        self.toRGB6 = nn.Sequential(EqualConv2d(64, 3, 1))
        self.toRGB7 = nn.Sequential(EqualConv2d(64, 3, 1))

        self.upConv = nn.Upsample(scale_factor=2, mode='nearest')
        self.warpconv1 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=3, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(3), nn.ReLU()).to(get_device())
        self.warpconv2 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=3, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(3), nn.ReLU()).to(get_device())
        self.warpconv3 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=3, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(3), nn.ReLU()).to(get_device())
        self.warpconv4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=3, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(3), nn.ReLU()).to(get_device())
        self.warpconv5 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(3), nn.ReLU()).to(get_device())
        self.warpconv6 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(3), nn.ReLU()).to(get_device())
        self.warpconv7 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(3), nn.ReLU()).to(get_device())

    def forward(self, appearance_features, latent_code):
        ### Initial Block


        out = self.style_conv((appearance_features[0], latent_code))
        rgb = self.toRGB(out)
        upConv = self.upConv(rgb)

        ### First
        out = self.style_conv1((out, latent_code))
        warpout = self.warpconv1(out)
        warpout = flow_warp(appearance_features[0], warpout)
        rgb1 = self.toRGB1(warpout)
        sum1 = rgb1 + upConv
        upConv1 = self.upConv(sum1)

        # ###### Second Block
        out = self.style_conv2((out, latent_code))
        warpout = self.warpconv2(out)
        warpout = flow_warp(appearance_features[1], warpout)
        rgb2 = self.toRGB2(warpout)
        sum2 = rgb2 + upConv1
        upConv2 = self.upConv(sum2)

        # ##### Third Block
        out = self.style_conv3((out, latent_code))
        warpout = self.warpconv3(out)
        warpout = flow_warp(appearance_features[2], warpout)
        rgb3 = self.toRGB3(warpout)
        sum3 = rgb3 + upConv2
        upConv3 = self.upConv(sum3)

        ##### Fourth Block
        out = self.style_conv4((out, latent_code))
        warpout = self.warpconv4(out)
        warpout = flow_warp(appearance_features[3], warpout)
        rgb4 = self.toRGB4(warpout)
        sum4 = rgb4 + upConv3
        upConv4 = self.upConv(sum4)

        # ##### Fifth Block
        out = self.style_conv5((out, latent_code))
        warpout = self.warpconv5(out)
        warpout = flow_warp(appearance_features[4], warpout)
        rgb5 = self.toRGB5(warpout)
        sum5 = rgb5 + upConv4
        upConv5 = self.upConv(sum5)

        # ##### Sixth Block
        out = self.style_conv6((out, latent_code))
        warpout = self.warpconv6(out)

        warpout = flow_warp(appearance_features[5], warpout)
        rgb6 = self.toRGB6(warpout)
        sum6 = rgb6 + upConv5

        return sum6
