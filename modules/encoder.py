import torch
import torch.nn.functional as F
from torch import nn
from modules.utils import ResBlock
from modules.utils import MultiLayerPerceptron
from torch import optim
from torch.utils.data import DataLoader


class Encoder(nn.Module):
    def __init__(self, in_channels, isSource):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.layer1 = nn.Sequential(ResBlock(64, 128))
        self.layer2 = nn.Sequential(ResBlock(128, 256))
        self.layer3 = nn.Sequential(ResBlock(256, 512))
        self.layer4 = nn.Sequential(ResBlock(512, 512))
        self.layer5 = nn.Sequential(ResBlock(512, 512))
        self.layer6 = nn.Sequential(ResBlock(512, 512))
        self.layer7 = nn.Sequential(ResBlock(512, 512, pool_stride=4))
        self.conv2 = nn.Conv2d(512, 512, kernel_size=4, padding=2, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(512)
        self.magnitude_mlp = MultiLayerPerceptron(input_size=512, hidden_size=512, magnitudes=20)
        self.isSource = isSource

    def forward(self, x):

        x_enc6 = self.conv1(x)

        x = self.bn1(x_enc6)
        x = F.relu(x)
        x_enc5 = self.layer1(x)
        x_enc4 = self.layer2(x_enc5)
        x_enc3 = self.layer3(x_enc4)
        x_enc2 = self.layer4(x_enc3)
        x_enc1 = self.layer5(x_enc2)
        x = self.layer6(x_enc1)
        x = self.layer7(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        if self.isSource:
            return [x_enc1, x_enc2, x_enc3, x_enc4, x_enc5, x_enc6], x
        else:
            magnitudes = self.magnitude_mlp(x)
            return magnitudes[0]
