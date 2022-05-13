from torch import nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
import torch
import math
import os




class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, pool_stride=2):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pool1 = nn.MaxPool2d(pool_stride)
        self.pool2 = nn.MaxPool2d(pool_stride)  ##used to downsample in resblock
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.pool1(out)
        residual = self.pool2(residual)

        out += residual
        out = F.relu(out)
        return out


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, magnitudes):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc3 = nn.Linear(input_size, hidden_size)
        self.fc4 = nn.Linear(input_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, magnitudes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        out = F.relu(out)
        out = self.fc5(out)
        return out


class ImagePyramid(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss.
    """

    def __init__(self, scales, num_channels):
        super(ImagePyramid, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """

    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out


def flow_warp(x, warped_conv, padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (n, c, h, w)
        flow (Tensor): size (n, 2, h, w), values range from -1 to 1 (relevant to image width or height)
        padding_mode (str): 'zeros' or 'border'

    Returns:
        Tensor: warped image or feature map
    """
    phi = torch.tanh(torch.tensor(warped_conv[0][0:2]))

    m = torch.sigmoid(torch.tensor(warped_conv[0][2]))
    # print(x.size()[-2:], flow.size()[-2:])
    assert x.size()[-2:] == phi.size()[-2:]
    n = 1
    _, _, h, w = x.size()
    x_ = torch.arange(w).view(1, -1).expand(h, -1)
    y_ = torch.arange(h).view(-1, 1).expand(-1, w)
    grid = torch.stack([x_, y_], dim=0).float()
    grid = grid.unsqueeze(0).expand(n, -1, -1, -1)
    grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
    grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
    grid += 2 * phi
    grid = grid.permute(0, 2, 3, 1)
    warped_image = F.grid_sample(x, grid)
    raw_masked_image = warped_image * m
    return raw_masked_image


def get_device():
    device = "cpu"
    # for working in GPU
    if torch.cuda.is_available():
        device = "cuda:0"
        deviceid = 1
        os.environ['CUDA_VISIBLE_DEVICES'] = "%d" % deviceid
        total, used = os.popen(
            '"nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
        ).read().split('\n')[deviceid].split(',')
        total = int(total)
        used = int(used)
        print(deviceid, 'Total GPU mem:', total, 'used:', used)

    return device


def load_model(source_Encoder, driver_Encoder, discriminator, generator,lmd, PATH):
    # Load the existing model
    if os.path.exists(PATH) and os.path.getsize(PATH) != 0:
        checkpoint = torch.load(PATH)
        source_Encoder = checkpoint['sourceencoder']
        driver_Encoder = checkpoint['driver_encoder']
        discriminator = checkpoint['discriminator']
        generator = checkpoint['generator']
        lmd = checkpoint['lmd']
    return source_Encoder, driver_Encoder, discriminator, generator, lmd


def save_model(source_Encoder, driver_Encoder, discriminator, generator,lmd, PATH):
    torch.save({
        'sourceencoder': source_Encoder,
        'driver_encoder': driver_Encoder,
        'discriminator': discriminator,
        'generator': generator,
        'lmd': lmd
    }, PATH)
