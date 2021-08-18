import torch
import torch.nn as nn
import torch.nn.functional as F


def disc_loss(gen, disc, criterion, real, label, num_images, z_dim, device):
    fake_noise = torch.randn(num_images, z_dim, device=device)
    fake_label = F.one_hot(torch.randint(0, 10, (num_images,)), num_classes=label.shape[1])
    fake = gen(fake_noise, fake_label)
    disc_fake_pred = disc(fake.detach())
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    disc_real_pred = disc(real, label)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    disc_loss = (disc_real_loss + disc_fake_loss) / 2
    return disc_loss


def gen_loss(gen, disc, criterion, num_images, z_dim, label, device, ):
    fake_noise = torch.randn(num_images, z_dim, device=device)
    fake_label = F.one_hot(torch.randint(0, 10, (num_images,)), num_classes=label.shape[1])
    fake = gen(fake_noise, fake_label)
    disc_fake_pred = disc(fake)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
    return gen_loss
