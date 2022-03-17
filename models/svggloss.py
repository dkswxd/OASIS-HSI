import torch
import torch.nn as nn
import torchvision
import torch.utils.checkpoint as cp


class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
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
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = X[:,5:36:5,:,:]
        _b, _c, _h, _w = X.shape
        X = X.reshape(_b * _c, 1, _h, _w).repeat(1, 3, 1, 1)
        h_relu1 = cp.checkpoint(self.slice1, X)
        h_relu2 = cp.checkpoint(self.slice2, h_relu1)
        h_relu3 = cp.checkpoint(self.slice3, h_relu2)
        h_relu4 = cp.checkpoint(self.slice4, h_relu3)
        h_relu5 = cp.checkpoint(self.slice5, h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
