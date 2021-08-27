import torch
import torch.nn as nn

class UpConvBlock(nn.Module):

    def __init__(self,chn_dim, hid_dim=64,kernel=3,stride=2,final_layer=False):
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
            img = self.upconv(img)
            return self.tanh(img)


class Generator(nn.Module):

    def __init__(self, noise_dim, label_dim, hid_dim):
        super(Generator, self).__init__()
        concat_dim = noise_dim + label_dim
        self.upconv1 = UpConvBlock(concat_dim,hid_dim*4)
        self.upconv2 = UpConvBlock(hid_dim*4,hid_dim*2,4,1)
        self.upconv3 = UpConvBlock(hid_dim * 2, hid_dim)
        self.upconv4 = UpConvBlock(hid_dim, 1, 4, final_layer=True)

    def forward(self,noise,label):
        conc_tensor = torch.cat([noise, label], 1)
        conc_tensor = conc_tensor.view(label.shape[0],conc_tensor.shape[1],1,1)
        gen_inp_1 = self.upconv1(conc_tensor)
        gen_inp_2 = self.upconv2(gen_inp_1)
        gen_inp_3 = self.upconv3(gen_inp_2)
        gen_out = self.upconv4(gen_inp_3)


        return gen_out



if __name__=="__main__":
    upcon = Generator(100,10,64)
    noise = torch.rand(4,100)
    label = torch.rand(4,10)
    print(upcon)
    res = upcon(noise,label)
    print(res.shape)


