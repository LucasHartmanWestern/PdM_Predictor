import csv
import logging
import os
import random
import sys

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from modified_unet import Modified_UNet
from modified_cgnet import Modified_CGNet


# ---------- Visualization Methods ---------- #

# def imshow_and_wait(img):
#     cv2.imshow('img', img)
#     keys = cv2.waitKey(0) & 0xFF
#     if keys == ord('q'):
#         cv2.destroyAllWindows()
#         quit()

# def visualize_seg_mask(img):
#     img_copy = img.copy()
#     img_copy = img_copy.astype(np.float32)
#     img_copy *= (255.0/(np.unique(img_copy).shape[0]-1))
#     img_copy = img_copy.astype(np.uint8)
#     imshow_and_wait(img_copy)

# ---------- Logging Methods ---------- #

def setup_basic_logger(save_path, log_name):
    # assert log_name in ['training', 'testing'], 'ERROR: incorrect log name input'
    logging.basicConfig(
        filename=os.path.join(save_path, f'{log_name}.log'),
        filemode='w',
        format="%(message)s",
        level=logging.INFO)

def log(msg):
    logging.info(msg)

def log_and_print(msg):
    logging.info(msg)
    print(msg)

def print_hyperparams(hyperparams):
    log_and_print('\nHYPERPARAMETERS:')
    for key, value in hyperparams.items():
        log_and_print('\t{}: {}'.format(key, str(value)))
    log_and_print('\n')

# ---------- Seed Methods ---------- #

def get_random_seed():
    """
    Usually the best random sample you could get in any programming language is generated through the operating system.
    In Python, you can use the os module.

    source: https://stackoverflow.com/questions/57416925/best-practices-for-generating-a-random-seeds-to-seed-pytorch/57416967#57416967
    """
    # Return a string of size random bytes suitable for cryptographic use.
    return int.from_bytes(os.urandom(4), byteorder="big")


def make_deterministic(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_seed_from_dataset(ds_folder_name):
    dataset_path = get_dataset_path(ds_folder_name)
    seed_path = os.path.join(dataset_path, 'dataset_seed.txt')
    with open(seed_path, 'r') as f:
        seed = int(f.read().strip())
    return seed

# ---------- Dataset Path Methods ---------- #

def get_dataset_path(ds_folder_name):
    if sys.platform == 'darwin':
        base_path = '/Users/nick_1/PycharmProjects/UWO Masters/data_60'
    elif sys.platform == 'win32':
        base_path = 'C:\\Users\\NickS\\UWO_Masters\\Datasets'
    else:
        base_path = '/mnt/storage_1/bell_5g_datasets'
    full_path = os.path.join(base_path, ds_folder_name)
    assert os.path.isdir(full_path), f"ERROR: dataset folder does not exist at '{full_path}'"
    return full_path

def fix_path(path):
    return path.replace('/', '\\') if sys.platform == 'win32' else path


# ---------- Model Methods ---------- #

def get_model(model_name, num_classes, use_rois, resize_shape=None):
    assert model_name.lower() in ['unet', 'cgnet'], 'ERROR: incorrect model name input'
    if model_name.lower() == 'unet':
        return Modified_UNet(num_classes, use_rois, resize_shape)
    elif model_name.lower() == 'cgnet':
        return Modified_CGNet(num_classes, use_rois, resize_shape)