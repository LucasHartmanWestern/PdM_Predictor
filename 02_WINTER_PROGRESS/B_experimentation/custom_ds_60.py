import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import get_dataset_path, fix_path


class Custom_DS_60(Dataset):
    def __init__(self, ds_folder_name, partition_name, binary_targets=False, resize_shape=None):
        assert partition_name in ['train', 'val', 'test'], "ERROR: Invalid partition name for custom dataset"
        day_interval = 6 if partition_name == 'val' else 1

        # initialize private variables
        self.resize_shape = resize_shape  # must be smaller than 1920x1080
        self.binary_targets = binary_targets
        self.x, self.y, self.day, self.hour = [], [], [], []

        # get dataset path and list name
        main_dir = os.path.join(partition_name, get_dataset_path(ds_folder_name))
        list_name = 'binary_list.txt' if binary_targets else 'multiclass_list.txt'

        # read list file and assign private variables
        for line in open(os.path.join(main_dir, list_name), "r"):
            x, y = fix_path(line).split(",")
            if int(x.split("_")[1]) % day_interval == 0:
                self.x.append(os.path.join(main_dir, x.strip()))
                self.y.append(os.path.join(main_dir, y.strip()))
                self.day.append(int(x.split("_")[1]))
                self.hour.append(str(x.split("_")[2]))
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        day = self.day[idx]
        hour = self.hour[idx]
        sample = cv2.imread(self.x[idx], cv2.IMREAD_COLOR)
        target = cv2.imread(self.y[idx], cv2.IMREAD_GRAYSCALE)

        # resize sample and target if specified
        # if self.resize_shape is not None and self.resize_shape != target.shape[:2]:
        #     sample = cv2.resize(sample, self.resize_shape, interpolation=cv2.INTER_AREA)
        #     target = cv2.resize(target, self.resize_shape, interpolation=cv2.INTER_AREA)

        # transpose sample from (H, W, C) to (C, H, W) and normalize values from [0, 255] to [0, 1]
        # sample = np.transpose(sample, axes=(2, 0, 1))
        # sample = sample.astype(np.float32)
        # sample *= (1 / 255.0)

        # convert target to LongTensor
        # target = torch.from_numpy(target).type(torch.LongTensor)

        return day, hour, sample, target
    

    