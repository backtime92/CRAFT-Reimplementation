from collections import namedtuple
import torch
from torchvision import models
from torchvision.models.vgg import model_urls
import torchutil
from torchutil import *
import os

weights_folder = os.path.join(os.path.dirname(__file__) + '/basenet/vgg16_bn-6c64b313.pth')

class vgg16_bn(torch.nn.Module):
    def __init__(self, pretrained=True, freeze=False):

        super(vgg16_bn, self).__init__()
        ##download link of vgg16
        model_urls['vgg16_bn'] = model_urls['vgg16_bn'].replace('https://', 'http://')
        vgg_pretrained_model= models.vgg16_bn(pretrained=False)
        if pretrained:
            vgg_pretrained_model.load_state_dict(
                copyStateDict(torch.load(weights_folder)))
        vgg_pretrained_features = vgg_pretrained_model.features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(12):  # conv2_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 19):  # conv3_3
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 29):  # conv4_3
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(29, 39):  # conv5_3
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        ## replace layers from no. range (39,44)
        self.slice5 = torch.nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1)
        )

        ## custom weights defined in torchutils for every stack
        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())

        init_weights(self.slice5.modules())  ## custom weights whether pretrained or not

        if freeze:
            for param in self.slice1.parameters():  # freeze first layer params
                param.requires_grad = False


    ## forward function
    def forward(self, X):
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h

        ## get every stack output
        vgg_outputs = namedtuple("VggOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out
