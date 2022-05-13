

from torch import nn
import torch
import torch.nn.functional as F
from torchvision import models
import numpy as np

from modules.utils import AntiAliasInterpolation2d


class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss.
    """

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = (x - self.mean) / self.std
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramid(torch.nn.Module):
    """
    Create image pyramid for computing pyramid perceptual loss.
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



class LossModel(torch.nn.Module):
    """
    Merge all updates into single model for better multi-gpu usage
    """

    def __init__(self):
        super(LossModel, self).__init__()
        train_params = { "scales": [1, 0.5, 0.25, 0.125], "loss_weights": {"perceptual": [10, 10, 10, 10], "reconstruction": 10,"adversarial_loss": 10}}
        self.scales = train_params['scales']
        self.recon_loss = nn.L1Loss(reduction='mean')
        self.pyramid = ImagePyramid(self.scales, 3)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()
        self.loss_weights = train_params['loss_weights']
        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()

    def forward(self, reconstructed_image, target_image, disc_prediction):
        loss_values = {}

        pyramide_real = self.pyramid(target_image)
        pyramide_generated = self.pyramid(reconstructed_image)

        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
                loss_values['perceptual'] = value_total

        if self.loss_weights['reconstruction'] != 0:
            loss_values['reconstruction'] = self.recon_loss(reconstructed_image,target_image)

        if self.loss_weights['adversarial_loss'] != 0:
            loss_values['adversarial_loss'] = -np.log(disc_prediction.detach().numpy())

        return loss_values
