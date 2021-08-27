import torch
import torch.nn as nn
from model.generator import Generator
from model.discriminator import Discriminator
from data_prep import data_loader
import torch.optim as optim
from training import train_cgan

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def training_cgan(
    data_dir,
    csv_file,
    criterion = nn.BCEWithLogitsLoss(),
    n_epochs = 20,
    z_dim = 64,
    display_step = 5,
    batch_size = 128,
    lr = 0.0002,
    viz = True,
    device=DEVICE):

    gen = Generator(z_dim,10).to(device)
    disc = Discriminator(1,10).to(device)
    optimizer_gen = optim.Adam(gen.parameters(), lr=lr)
    optimizer_disc = optim.Adam(disc.parameters(), lr=lr)

    dataloader = data_loader(data_dir, csv_file,batch_size)

    train_cgan(
        gen,
        disc,
        optimizer_gen,
        optimizer_disc,
        dataloader,
        criterion,
        display_step,
        z_dim,
        n_epochs,
        viz,
        device)


if __name__ == '__main__':
    data_dir = r'C:\Users\Ankan\Downloads\fashion'
    csv_file = r'C:\Users\Ankan\Downloads\fashion\index.csv'
    training_cgan(data_dir,csv_file)
