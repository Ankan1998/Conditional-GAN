import torch
import torch.nn as nn

class UpConvBlock(nn.Module):

    def __init__(self,chn_dim, hid_dim,kernel,stride,final_layer=False):
        super(UpConvBlock, self).__init__()

        self.final_layer = final_layer
        if not self.final_layer:
            self.upconv = nn.ConvTranspose2d(chn_dim, hid_dim, kernel, stride)
            self.bn = nn.BatchNorm2d(hid_dim)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.upconv = nn.ConvTranspose2d(chn_dim, hid_dim, kernel, stride)
            self.tanh = nn.Tanh()

    def forward(self, img):
        if not self.final_layer:
            img = self.upconv(img)
            img = self.bn(img)
            img = self.relu(img)
            return img
        else:
            img = self.conv(img)
            return self.tanh(img)


