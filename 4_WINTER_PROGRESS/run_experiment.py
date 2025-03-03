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


def run_ROI_experiment(model_name, results_folder, seed=None, binary_labels=False):
    # hyperparameters
    batch_sz = 15  # 15 roi's make up 1 full image
    n_epochs = 15
    patience = 5
    num_classes = 2 if binary_labels else 7
    input_shape = (360, 360)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

     # set up deterministic seed
    seed = get_random_seed() if seed is None else seed
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
        "Number of Classes": num_classes,
        "Optimizer": "AdamW",
        "Loss Function": "Cross Entropy",
        "Training Dataset Name": "feb28_ds1",
        "Validation Dataset Name": "feb28_ds2",
        "Using ROI's?": True,
        "Binary Labels?": binary_labels,
        "Save Location": save_path
    })

    # set up dataset(s)
    train_ds = ROI_DS(["feb28_ds1"], binary_labels)
    val_ds = ROI_DS_Val("feb28_ds2", binary_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_sz, shuffle=False)

    # compile model
    model = UNet(num_classes=num_classes) if model_name.lower() == "unet" else CGNet(num_classes=num_classes)
    model.to(device=device)

    # init model optimization parameters
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters())

    # run torch summary report
    # summary(model, input_size=(3, input_shape[0], input_shape[1]))

    # train model
    train(model, loss_fn, optimizer, train_loader, val_loader, n_epochs, device, save_path, num_classes, patience)


if __name__ == '__main__':
    # hyperparameters
    custom_seed = 1234567890

    # -- run experiments -- #

    # multiclass label experiments
    run_ROI_experiment("unet", "UNet_w_ROIs_multiclass", custom_seed, False)
    run_ROI_experiment("cgnet", "CGNet_w_ROIs_multiclass", custom_seed, False)

    # binary label experiments
    run_ROI_experiment("unet", "UNet_w_ROIs_binary", custom_seed, True)
    run_ROI_experiment("cgnet", "CGNet_w_ROIs_binary", custom_seed, True)
