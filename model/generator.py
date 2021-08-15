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
            img = self.upconv(img)
            return self.tanh(img)


class Generator(nn.Module):

    def __init__(self,gen_img_dim, noise_dim, label_dim, hid_dim):
        super(Generator, self).__init__()
        self.gen_img_dim = gen_img_dim
        concat_dim = noise_dim + label_dim
        self.label_linear = nn.Linear(concat_dim,self.gen_img_dim*self.gen_img_dim)
        self.upconv1 = UpConvBlock(1,hid_dim*2,3,2)
        self.upconv2 = UpConvBlock(hid_dim*2,hid_dim,3,1)
        self.upconv3 = UpConvBlock(hid_dim, 3, 3, 1,final_layer=True)

    def forward(self,noise,label):
        conc_tensor = torch.cat([noise, label], 1)
        out = self.label_linear(conc_tensor)
        out_reshaped = torch.reshape(out, (label.shape[0],1, 9, 9))
        gen_inp = self.upconv1(out_reshaped)
        gen_inp = self.upconv2(gen_inp)
        gen_out = self.upconv3(gen_inp)

        return gen_out



if __name__=="__main__":
    upcon = Generator(28,100,10,32)
    noise = torch.rand(4,100)
    label = torch.rand(4,10)
    print(upcon)
    res = upcon(noise,label)
    print(res.shape)


