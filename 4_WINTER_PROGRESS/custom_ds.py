import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import get_dataset_path, fix_path


class FullSize_DS(Dataset):
    def __init__(self, ds_folder_list, binary_targets=False, resize_shape=None):
        self.resize_shape = resize_shape  # must be smaller than 1920x1080
        self.binary_targets = binary_targets
        self.x, self.y, self.day = [], [], []

        list_name = 'binary_list.txt' if binary_targets else 'list.txt'

        # merge train datasets into one
        for ds_folder_name in ds_folder_list:
            main_dir = os.path.join("full_size", get_dataset_path(ds_folder_name))

            for line in open(os.path.join(main_dir, "full_size", list_name), "r"):
                x, y = fix_path(line).split(",")
                self.x.append(os.path.join(main_dir, "full_size", x.strip()))
                self.y.append(os.path.join(main_dir, "full_size", y.strip()))
                self.day.append(int(x.split("_")[1]))

        # print("-- Full Size Dataset Created --") # debugging
        # print(f"\tConcatenated {len(ds_folder_list)} datasets,")
        # print(f"\tContaining a total of {len(self.x)} samples.")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = cv2.imread(self.x[idx], cv2.IMREAD_COLOR)
        target = cv2.imread(self.y[idx], cv2.IMREAD_GRAYSCALE)

        if self.resize_shape is not None and self.resize_shape != target.shape[:2]:
            sample = cv2.resize(sample, self.resize_shape, interpolation=cv2.INTER_AREA)
            target = cv2.resize(target, self.resize_shape, interpolation=cv2.INTER_AREA)

        # transpose RGB sample to be (C, H, W) instead of (H, W, C)
        sample = np.transpose(sample, axes=(2, 0, 1))  

        # normalize values from [0, 255] to [0, 1]
        sample = sample.astype(np.float32)
        sample *= (1 / 255.0)

        # convert target to LongTensor
        target = torch.from_numpy(target).type(torch.LongTensor)

        return self.day[idx], sample, target
    

# Only used for getting preliminary results
class FullSize_DS_Val(Dataset):
    def __init__(self, ds_folder, binary_targets=False, resize_shape=None):
        self.resize_shape = resize_shape  # must be smaller than 1920x1080
        self.binary_targets = binary_targets
        self.x, self.y, self.day = [], [], []
        list_name = 'binary_list.txt' if binary_targets else 'list.txt'
        main_dir = os.path.join("full_size", get_dataset_path(ds_folder))
        day_interval = 8
        for line in open(os.path.join(main_dir, "full_size", list_name), "r"):
            x, y = fix_path(line).split(",") 
            if int(x.split("_")[1]) % day_interval == 0:
                self.x.append(os.path.join(main_dir, "full_size", x.strip()))
                self.y.append(os.path.join(main_dir, "full_size", y.strip()))
                self.day.append(int(x.split("_")[1]))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = cv2.imread(self.x[idx], cv2.IMREAD_COLOR)
        target = cv2.imread(self.y[idx], cv2.IMREAD_GRAYSCALE)

        if self.resize_shape is not None and self.resize_shape != target.shape[:2]:
            sample = cv2.resize(sample, self.resize_shape, interpolation=cv2.INTER_AREA)
            target = cv2.resize(target, self.resize_shape, interpolation=cv2.INTER_AREA)

        sample = np.transpose(sample, axes=(2, 0, 1))  
        sample = sample.astype(np.float32)
        sample *= (1 / 255.0)
        target = torch.from_numpy(target).type(torch.LongTensor)
        return self.day[idx], sample, target


class ROI_DS(Dataset):
    def __init__(self, ds_folder_list, binary_targets=False):
        self.binary_targets = binary_targets
        self.x, self.y, self.day = [], [], []

        list_name = 'binary_list.txt' if binary_targets else 'list.txt'

        # merge train datasets into one
        for ds_folder_name in ds_folder_list:
            main_dir = os.path.join("rois", get_dataset_path(ds_folder_name))

            for line in open(os.path.join(main_dir, "rois", list_name), "r"):
                x, y = fix_path(line).split(",")
                self.x.append(os.path.join(main_dir, "rois", x.strip()))
                self.y.append(os.path.join(main_dir, "rois", y.strip()))
                self.day.append(int(x.split("_")[1]))

        # print("-- ROI Dataset Created --") # debugging
        # print(f"\tConcatenated {len(ds_folder_list)} datasets,")
        # print(f"\tContaining a total of {len(self.x)} samples.")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = cv2.imread(self.x[idx], cv2.IMREAD_COLOR)
        target = cv2.imread(self.y[idx], cv2.IMREAD_GRAYSCALE)

        # transpose RGB sample to be (C, H, W) instead of (H, W, C)
        sample = np.transpose(sample, axes=(2, 0, 1))  

        # normalize values from [0, 255] to [0, 1]
        sample = sample.astype(np.float32)
        sample *= (1 / 255.0)

        # convert target to LongTensor
        target = torch.from_numpy(target).type(torch.LongTensor)

        return self.day[idx], sample, target


# Only used for getting preliminary results
class ROI_DS_Val(Dataset):
    def __init__(self, ds_folder, binary_targets=False):
        self.binary_targets = binary_targets
        self.x, self.y, self.day = [], [], []
        list_name = 'binary_list.txt' if binary_targets else 'list.txt'
        main_dir = os.path.join("rois", get_dataset_path(ds_folder))
        day_interval = 8
        for line in open(os.path.join(main_dir, "rois", list_name), "r"):
            x, y = fix_path(line).split(",") 
            if int(x.split("_")[1]) % day_interval == 0:
                self.x.append(os.path.join(main_dir, "rois", x.strip()))
                self.y.append(os.path.join(main_dir, "rois", y.strip()))
                self.day.append(int(x.split("_")[1]))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = cv2.imread(self.x[idx], cv2.IMREAD_COLOR)
        target = cv2.imread(self.y[idx], cv2.IMREAD_GRAYSCALE)
        # transpose RGB sample to be (C, H, W) instead of (H, W, C)
        sample = np.transpose(sample, axes=(2, 0, 1))  
        # normalize values from [0, 255] to [0, 1]
        sample = sample.astype(np.float32)
        sample *= (1 / 255.0)
        target = torch.from_numpy(target).type(torch.LongTensor)
        return self.day[idx], sample, target
    