import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class CGANDataSet(Dataset):

    def __init__(self, data_dir, csv_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        df = pd.read_csv(csv_file)
        one_enc = pd.get_dummies(df.iloc[:, 0])
        df = pd.concat([df, one_enc], axis=1)
        df = df.iloc[:, 1:]
        self.dataframe = df
        keys = df.columns.tolist()[1:]
        val = list(range(len(keys)))
        self.dictionary = dict(zip(keys, val))

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.dataframe.iloc[idx, 0])
        img = Image.open(img_path)
        label = torch.tensor(self.dataframe.iloc[idx, 1:].tolist())

        if self.transform is not None:
            img = self.transform(img)

        return img, label

def data_loader(data_dir, csv_file,batch_size):
    transform = transforms.Compose(
        [

            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    return DataLoader(
        CGANDataSet(data_dir, csv_file, transform=transform),
        batch_size = batch_size,
        shuffle = True
    )


if __name__ == "__main__":
    data_dir = r'C:\Users\Ankan\Downloads\fashion'
    csv_file = r'C:\Users\Ankan\Downloads\fashion\index.csv'
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    cgandataset = CGANDataSet(data_dir,csv_file,transform = transform)
    # print(cgandataset.__len__)
    # img, label = cgandataset.__getitem__(0)
    # print(label.shape)
    # print('*'*100)
    # print(cgandataset.dictionary)
    # print(img.shape)
    # plt.imshow(img)
    # plt.show()
    for img, label in data_loader(data_dir,csv_file,4):
        print(img.shape)
        print(label.shape)

        break

