import csv
import logging
import os
import random
import sys

import numpy as np
import torch

# ---------- Logging Methods ---------- #

def setup_basic_logger(save_path):
    logging.basicConfig(
        filename=os.path.join(save_path, 'training.log'),
        filemode='w',
        format="%(message)s",
        level=logging.INFO)

def log(msg):
    logging.info(msg)

def log_and_print(msg):
    logging.info(msg)
    print(msg)

def print_hyperparams(hyperparams):
    log_and_print('HYPERPARAMETERS:')
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

# ---------- Dataset Path Methods ---------- #

def get_dataset_path(ds_folder_name):
    if sys.platform == 'darwin':
        base_path = '/Users/nick_1/PycharmProjects/UWO Masters/data'
    elif sys.platform == 'win32':
        base_path = 'C:\\Users\\NickS\\UWO_Masters\\Datasets'
    else:
        base_path = '/mnt/storage_1/bell_5g_datasets'
    full_path = os.path.join(base_path, ds_folder_name)
    assert os.path.isdir(full_path), f"ERROR: dataset folder does not exist at '{full_path}'"
    return os.path.join(base_path, ds_folder_name)

def fix_path(path):
    return path.replace('/', '\\') if sys.platform == 'win32' else path

