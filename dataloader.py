import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
from skimage import io
from torchvision import transforms
from PIL import Image
import pandas as pd

class MnistClutteredDataset(Dataset):

    def __init__(self, data_path, type, transform=None):

        self.root_dir = data_path +'/'+ type + '/path.txt'
        self.transform = transform
        self.path = pd.read_csv(self.root_dir, sep=' ', header=None)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.path.iloc[idx,0]

        image = Image.open(img_path)

        label = int(self.path.iloc[idx,1])

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.path)
