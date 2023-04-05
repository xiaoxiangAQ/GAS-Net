"""
Utilities for loading training data
"""
import os
import random
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data

from data_utils import get_transform,make_one_hot
from torchvision import transforms

class Dataset_self(data.Dataset):
    def __init__(self, data_dir):
        super(Dataset_self, self).__init__()

        self.root_path = data_dir
        self.filenames = [x for x in sorted(os.listdir(data_dir))]
        
        # self.transform = get_transform(convert=True, normalize= True)
        self.transform = get_transform(convert=True, normalize= False)
        self.label_transform = get_transform()

    def __getitem__(self, index):

        filepath = os.path.join(self.root_path, self.filenames[index])
        filepath2 = filepath.replace('images','t2').replace('t1','t2')
        filepath_label = filepath.replace('images','label').replace('t1','label')
        
        ## without resize
        img = self.transform(Image.open(filepath).convert('RGB'))
        img2 = self.transform(Image.open(filepath2).convert('RGB'))
        label_img = Image.open(filepath_label).convert('1')

        label = self.label_transform(label_img)
        label = make_one_hot(label.unsqueeze(0).long(), 2).squeeze(0)

        return img, img2, label

    def __len__(self):
        return len(self.filenames)



if  __name__ == "__main__":
    data_A_dir = '/media/zrq/Data/lim-cd' 
    img_data = Dataset(data_A_dir)
    data_loader_train = torch.utils.data.DataLoader(dataset=img_data,
                                                batch_size = 4,
                                                shuffle = True)
    for x in data_loader_train:
        print(x.shape)
        exit()