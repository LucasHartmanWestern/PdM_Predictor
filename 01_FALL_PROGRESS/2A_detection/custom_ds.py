import os
import sys

import cv2
import numpy as np
from torch.utils.data import Dataset

from utils.constants import *
from utils.misc_util import fix_path


class CustomDS(Dataset):
    def __init__(self, x_set, y_set, ds_folder_name, resize_shape=None):
        self.resize_shape = resize_shape  # must be smaller than 1920x1080
        if sys.platform == 'darwin':
            root_dir = os.path.join(data_path_mac, str(ds_folder_name))
        elif sys.platform == 'win32':
            root_dir = os.path.join(data_path_win32, str(ds_folder_name))
        else:
            root_dir = os.path.join(data_path_linux, str(ds_folder_name))
        self.x, self.y = [], []
        for idx in range(len(x_set)):
            self.x.append(os.path.join(root_dir, fix_path(x_set[idx])))
            self.y.append(os.path.join(root_dir, fix_path(y_set[idx])))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = cv2.imread(self.x[idx], cv2.IMREAD_COLOR)
        target = cv2.imread(self.y[idx], cv2.IMREAD_GRAYSCALE)

        if self.resize_shape is not None and self.resize_shape != target.shape[:2]:
            image = cv2.resize(image, self.resize_shape, interpolation=cv2.INTER_AREA)
            target = cv2.resize(target, self.resize_shape, interpolation=cv2.INTER_AREA)

        image = np.transpose(image, axes=(2, 0, 1))  # transpose RGB image to be (C, H, W) instead of (H, W, C)
        target = np.expand_dims(target, axis=0)  # add dimension for where channels normally are

        # normalize values from [0, 255] to [0, 1]
        image = image.astype(np.float32)
        target = target.astype(np.float32)
        image *= (1 / 255.0)
        target *= (1 / 255.0)

        # ensure that the target is still binary after previous operations
        target[target < 0.5] = 0
        target[target >= 0.5] = 1

        return image, target


class SMSTestDS(Dataset):
    def __init__(self):
        ds_folder_name = 'sm_SMS_ds'
        if sys.platform == 'darwin':
            root_dir = os.path.join(data_path_mac, str(ds_folder_name))
        elif sys.platform == 'win32':
            root_dir = os.path.join(data_path_win32, str(ds_folder_name))
        else:
            root_dir = os.path.join(data_path_linux, str(ds_folder_name))

        list_path = os.path.join(root_dir, 'test', 'list.txt')
        assert os.path.isfile(list_path), "ERROR: no dataset list exists at '{}'".format(list_path)

        self.x, self.y, self.day = [], [], []
        for line in open(list_path, "r"):
            x, y, d = line.split(' ')
            self.x.append(os.path.join(root_dir, fix_path(x.strip())))
            self.y.append(os.path.join(root_dir, fix_path(y.strip())))
            self.day.append(int(d.strip()))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = cv2.imread(self.x[idx], cv2.IMREAD_COLOR)
        target = cv2.imread(self.y[idx], cv2.IMREAD_GRAYSCALE)

        image = np.transpose(image, axes=(2, 0, 1))  # transpose RGB image to be (C, H, W) instead of (H, W, C)
        target = np.expand_dims(target, axis=0)  # add dimension for where channels normally are

        # normalize values from [0, 255] to [0, 1]
        image = image.astype(np.float32)
        target = target.astype(np.float32)
        image *= (1 / 255.0)
        target *= (1 / 255.0)

        # ensure that the target is still binary after previous operations
        target[target < 0.5] = 0
        target[target >= 0.5] = 1

        return image, target, self.day[idx]
