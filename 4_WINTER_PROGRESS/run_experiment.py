import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchsummary import summary

from custom_ds import ROI_DS, ROI_DS_Val, FullSize_DS, FullSize_DS_Val
from architectures.unet import UNet
from architectures.cgnet import Context_Guided_Network as CGNet
from training import train
from utils import *


def run_ROI_experiment(model, results_folder, n_epochs=20, seed=None):
    # hyperparameters
    batch_sz = 16  # batch size
    patience = 5
    num_classes = 7
    input_shape = (360, 360)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

     # set up deterministic seed
    seed = get_random_seed() if seed is not None else seed
    make_deterministic(seed)

    # set up paths and directories
    save_path = os.path.join(".", "4_WINTER_PROGRESS", "preliminary_RESULTS", results_folder)
    os.makedirs(save_path, exist_ok=True)

    # set up logger
    setup_basic_logger(save_path)

    # print training hyperparameters
    print_hyperparams({
        "Seed": seed,
        "Epochs": n_epochs,
        "Batch Size": batch_sz,
        "Patience": patience,
        "Device": device,
        "Input Shape": f"({input_shape[0]}, {input_shape[1]}, 3)",
        "Output Shape": f"({input_shape[0]}, {input_shape[1]}, {num_classes})",
        "Number of Classes": 7,
        "Optimizer": "AdamW",
        "Loss Function": "Cross Entropy",
        "Training Dataset Name": "feb28_ds1",
        "Validation Dataset Name": "feb28_ds2",
        "Using ROI's?": True,
        "Save Location": save_path
    })

    # set up dataset(s)
    train_ds = ROI_DS(["feb28_ds1"])
    val_ds = ROI_DS_Val("feb28_ds2")
    train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_sz, shuffle=False)

    # compile model
    model.to(device=device)

    # init model optimization parameters
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters())

    # run torch summary report
    # summary(model, input_size=(3, input_shape[0], input_shape[1]))

    # train model
    train(model, loss_fn, optimizer, train_loader, val_loader, n_epochs, device, save_path, patience)


if __name__ == '__main__':
    # hyperparameters
    custom_seed = 9876543210
    epochs = 2

    # run experiments
    run_ROI_experiment(UNet(num_classes=7), "UNet_w_ROIs", epochs, custom_seed)
    run_ROI_experiment(CGNet(classes=7), "CGNet_w_ROIs", epochs, custom_seed)
