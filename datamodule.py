import os 
import torch 
from torch.utils.data import random_split
import torch_geometric.loader as loader 
import torch_geometric.datasets as datasets
import torch_geometric.transforms as transforms 
import pytorch_lightning as pl 

def get_3d_feats(data):
    data.x = torch.cat([data.x, data.pos], dim=1)
    return data

class MNISTSuperpixelsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./MNISTSuperpixels', train_split=50000, batch_size=4096):
        super().__init__()
        self.data_dir = data_dir
        self.train_split = train_split
        self.batch_size = batch_size

    def setup(self, stage=None):
        train = datasets.MNISTSuperpixels(root=self.data_dir, transform=get_3d_feats, train=True)
        test = datasets.MNISTSuperpixels(root=self.data_dir, transform=get_3d_feats, train=False)
        self.mnist_superpixels_train, self.mnist_superpixels_val = random_split(
            train, [self.train_split , len(train) - self.train_split]
        )
        self.mnist_superpixels_test = test

    def train_dataloader(self):
        return loader.DataLoader(self.mnist_superpixels_train, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())
    
    def val_dataloader(self):
        return loader.DataLoader(self.mnist_superpixels_val, batch_size=self.batch_size, num_workers=os.cpu_count())

    def test_dataloader(self):
        return loader.DataLoader(self.mnist_superpixels_test, batch_size=self.batch_size, num_workers=os.cpu_count())
