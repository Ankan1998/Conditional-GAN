import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, chn_dim, hid_dim, kernel=4,stride=2,final_layer=False):
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

    def __init__(self, channel_dim, label_dim,hid_dim=64):
        super(Discriminator, self).__init__()
        self.conv1 = ConvBlock(channel_dim + label_dim, hid_dim)
        self.conv2 = ConvBlock(hid_dim, hid_dim * 2)
        self.conv3 = ConvBlock(hid_dim * 2, 1,final_layer=True)
        self.dropout = nn.Dropout(0.25)

    # def construct_label(self,label):
    #     batch_list = []
    #     for i in range(len(label)):
    #         empty_tensor_list = []
    #         for j in label[i]:
    #             empty_tensor_list.append(torch.full((28, 28), j))
    #         tensor_label = torch.stack(empty_tensor_list, 0)
    #         batch_list.append(tensor_label)
    #     return torch.stack(batch_list, 0)

    # Optimised version
    def construct_label(self,label_tensor):
        return label_tensor[..., None, None].repeat((1, 1, 28, 28))

    def forward(self, img, label):

        label_reshaped = self.construct_label(label)
        concat_inp = torch.cat([img, label_reshaped], 1)
        inp = self.conv1(concat_inp)
        inp = self.conv2(inp)
        inp = self.dropout(self.conv3(inp))

        flattend = inp.view(inp.shape[0],-1)

        return flattend

if __name__=="__main__":

    disc = Discriminator(1,10,32)
    print(disc)
    img = torch.rand(4,1,28,28)
    label = torch.rand(4,10)
    vec = disc(img,label)
    print(vec.shape)