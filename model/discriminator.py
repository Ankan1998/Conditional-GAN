import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, chn_dim, hid_dim, kernel):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(chn_dim, hid_dim, kernel)
        self.bn = nn.BatchNorm2d(hid_dim)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, img):
        img = self.conv(img)
        img = self.bn(img)
        img = self.leakyrelu(img)
        return img



