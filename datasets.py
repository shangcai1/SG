import glob
import random
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(root + '/*.*'))
    def __getitem__(self, index):
        name = int(self.files[index].split('/')[-1].split('.')[0])
        img = Image.open(self.files[index]).convert('RGB')
        return self.transform(img)

    def __len__(self):
        return len(self.files)#,len(self.files1)

class ImageDataset_test(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(root + '/*.*'))

    def __getitem__(self, index):
        name = int(self.files[index].split('/')[-1].split('.')[0])
        img = Image.open(self.files[index]).convert('RGB')

        return self.transform(img),name

    def __len__(self):
        return len(self.files)

# Configure dataloaders

def Get_dataloader(path,batch):
    #Image.BICUBIC
    transforms_ = [ transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

    train_dataloader = DataLoader(
        ImageDataset(path, transforms_=transforms_),
        batch_size=batch, shuffle=True, num_workers=2, drop_last=True)
    return train_dataloader

def Get_dataloader_test(path,batch):
    transforms_ = [
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

    train_dataloader = DataLoader(
        ImageDataset_test(path, transforms_=transforms_),
        batch_size=batch, shuffle=False, num_workers=2, drop_last=True)

    return train_dataloader


