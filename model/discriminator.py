import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, chn_dim, hid_dim, kernel,stride,final_layer=False):
        super(ConvBlock, self).__init__()

        self.final_layer = final_layer
        if not self.final_layer:
            self.conv = nn.Conv2d(chn_dim, hid_dim, kernel,stride)
            self.bn = nn.BatchNorm2d(hid_dim)
            self.leakyrelu = nn.LeakyReLU(0.2)
        else:
            self.conv = nn.Conv2d(chn_dim, hid_dim, kernel, stride)


    def forward(self, img):
        if not self.final_layer:
            img = self.conv(img)
            img = self.bn(img)
            img = self.leakyrelu(img)
            return img
        else:
            return self.conv(img)


class Discriminator(nn.Module):

    def __init__(self, img_dim, channel_dim, hid_dim, label_dim):
        super(Discriminator, self).__init__()
        self.img_dim = img_dim
        self.label_linear = nn.Linear(label_dim, self.img_dim * self.img_dim)

        self.conv1 = ConvBlock(channel_dim + 1, hid_dim, 3,2)
        self.conv2 = ConvBlock(hid_dim, hid_dim * 2, 3,1)
        self.conv3 = ConvBlock(hid_dim * 2, 1, 3,1, final_layer=True)
        self.dropout = nn.Dropout(0.25)

    def forward(self, img, label):
        label = self.label_linear(label)
        label_reshaped = torch.reshape(label, (label.shape[0],1, self.img_dim, self.img_dim))
        concat_inp = torch.cat([img, label_reshaped], 1)
        inp = self.conv1(concat_inp)
        inp = self.conv2(inp)
        inp = self.dropout(self.conv3(inp))
        print(inp.shape)
        flattend = inp.view(inp.shape[0],-1)

        return flattend

if __name__=="__main__":

    disc = Discriminator(28,3,32,10)
    print(disc)
    img = torch.rand(4,3,28,28)
    label = torch.rand(4,1,1,10)
    vec = disc(img,label)
    print(vec.shape)