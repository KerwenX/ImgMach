from torch.utils.data import DataLoader,Dataset
import torch
import numpy as np
from PIL import Image
import os

class GameDataset(Dataset):
    def __init__(self, IMAGE_DIR, num_images=-1, transform=None,target_transform=None):
        super(GameDataset, self).__init__()
        self.file_dir = IMAGE_DIR
        if os.path.isdir(self.file_dir) is False:
            os.mkdir(self.file_dir)
        self.file_list = []
        for file in os.listdir(self.file_dir):
            if os.path.splitext(file)[1].endswith(('jpg','png')):
                self.file_list.append(os.path.join(self.file_dir,file))
        self.transforms = transform
        self.num_images = num_images
        self.file_list = self.file_list[:self.num_images]
        self.target_transform = target_transform


    def __getitem__(self, item):
        img_src = Image.open(self.file_list[item])
        if self.transforms:
            img_src = self.transforms(img_src)
        return img_src

    def __len__(self):
        return len(self.file_list)
