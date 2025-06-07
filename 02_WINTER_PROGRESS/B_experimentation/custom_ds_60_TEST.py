import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import get_dataset_path, fix_path


class Custom_DS_60_TEST(Dataset):
    def __init__(self, ds_folder_name: str, binary_targets: bool):
        # initialize private variables
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.binary_targets = binary_targets
        self.x, self.y, self.day, self.hour = [], [], [], []

        # get dataset path and list name
        main_dir_1 = os.path.join(get_dataset_path(ds_folder_name), 'train')
        main_dir_2 = os.path.join(get_dataset_path(ds_folder_name), 'val')
        main_dir_3 = os.path.join(get_dataset_path(ds_folder_name), 'test')
        list_name = 'binary_list.txt' if binary_targets else 'multiclass_list.txt'

        # read list file and assign private variables
        for dir_count, main_dir in enumerate([main_dir_1, main_dir_2, main_dir_3]):
            for line in open(os.path.join(main_dir, list_name), "r"):
                x, y = fix_path(line).split(",")
                self.x.append(os.path.join(main_dir, x.strip()))
                self.y.append(os.path.join(main_dir, y.strip()))
                day = int(x.split("_")[1])
                self.day.append(day + (dir_count * 20))
                self.hour.append(str(x.split("_")[2].split(".")[0]))
                if day == 20:
                    break
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        day = self.day[idx]
        hour = self.hour[idx]
        sample = cv2.imread(self.x[idx], cv2.IMREAD_COLOR)
        target = cv2.imread(self.y[idx], cv2.IMREAD_GRAYSCALE)

        # transpose sample from (H, W, C) to (C, H, W) and normalize values from [0, 255] to [0, 1]
        sample = np.transpose(sample, axes=(2, 0, 1))
        sample = sample.astype(np.float32)
        sample *= (1 / 255.0)

        # convert sample and target to tensor and move to device
        sample = torch.from_numpy(sample).type(torch.FloatTensor).to(device=self.device)
        target = torch.from_numpy(target).type(torch.LongTensor).to(device=self.device)

        return day, hour, sample, target
    

    