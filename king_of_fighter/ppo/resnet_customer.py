from torchvision import models
from torch import nn
import torch
import torch.nn.functional as F

class resnet18_customer(nn.Module):
    def __init__(self, out_dim=512, input_size=[128, 160]):
        super().__init__()
        self.input_size = input_size
        self.features_dim = out_dim
        n_patches = (self.input_size[0]//32)*(self.input_size[1]//32)
        self.resnet = models.resnet18()
        self.linear = nn.Linear(512*n_patches, out_dim)

    def forward(self, x):
        if x.shape[-1] != self.input_size[-1] or x.shape[-2] != self.input_size[-2]:
            x = F.interpolate(x, self.input_size)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = torch.flatten(x, 1)
        x = self.linear(x.relu())
        return x