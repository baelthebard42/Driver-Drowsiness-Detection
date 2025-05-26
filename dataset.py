import os, json
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch

class DrowsinessDataset(Dataset):

    def __init__(self, annotation_dir, transform=None):
        super().__init__()
        self.annotation_dir = annotation_dir
        self.transform = transform

        with open(self.annotation_dir, 'r') as file:
            self.data = list(json.load(file).values())
    
    def __getitem__(self, index):
        image_path = self.data[index]['image_path']
        image = np.array(Image.open(image_path).convert("RGB"))
        targets = torch.tensor(list(self.data[index]["targets"].values()), dtype=torch.float32)

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]
        
        return image, targets