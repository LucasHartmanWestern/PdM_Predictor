import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchsummary import summary

from custom_ds import ROI_DS, ROI_DS_Val, FullSize_DS, FullSize_DS_Val
from architectures.unet import UNet
from architectures.cgnet import Context_Guided_Network as CGNet
from training import train
from testing import test
from utils import *


def run_ROI_experiment(model_name, results_folder, seed=None, binary_labels=False):
    # hyperparameters
    batch_sz = 15  # 15 roi's make up 1 full image
    n_epochs = 15
    patience = 10
    num_classes = 2 if binary_labels else 7
    input_shape = (360, 360)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

     # set up deterministic seed
    seed = get_random_seed() if seed is None else seed
    make_deterministic(seed)

    # set up paths and directories
    save_path = os.path.join(".", "preliminary_RESULTS", results_folder)
    os.makedirs(save_path, exist_ok=True)

    # set up logger
    setup_basic_logger(save_path, 'training')

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


def test_ROI_experiment(model_name, results_folder, seed=None, binary_labels=False):
    # hyperparameters
    batch_sz = 15  # 15 roi's make up 1 full image
    num_classes = 2 if binary_labels else 7
    input_shape = (360, 360)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

     # set up deterministic seed
    seed = get_random_seed() if seed is None else seed
    make_deterministic(seed)

    # set up paths and directories
    save_path = os.path.join(".", "preliminary_RESULTS", results_folder)
    os.makedirs(save_path, exist_ok=True)

    # set up logger
    setup_basic_logger(save_path, 'testing')

    # print training hyperparameters
    print_hyperparams({
        "Seed": seed,
        "Batch Size": batch_sz,
        "Device": device,
        "Input Shape": f"({input_shape[0]}, {input_shape[1]}, 3)",
        "Output Shape": f"({input_shape[0]}, {input_shape[1]}, {num_classes})",
        "Number of Classes": num_classes,
        "Testing Dataset Name": "feb28_ds3",
        "Using ROI's?": True,
        "Binary Labels?": binary_labels,
        "Save Location": save_path
    })

    # set up dataset(s)
    test_ds = ROI_DS(["feb28_ds3"], binary_labels)
    test_loader = DataLoader(test_ds, batch_size=batch_sz, shuffle=False)

    # compile model
    model = UNet(num_classes=num_classes) if model_name.lower() == "unet" else CGNet(num_classes=num_classes)
    model.load_state_dict(torch.load(os.path.join(save_path, 'best_weights.pth'), weights_only=True))
    model.to(device=device)

    # test model
    test(model, test_loader, device, save_path, num_classes)


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp', type=str, help='experiment type', required=True)
    parser.add_argument('-model', type=str, help='model name', required=True)
    parser.add_argument('-folder', type=str, help='save folder', required=True)
    parser.add_argument('-binary', type=str, help='binary labels (y/n)', required=True)
    args = parser.parse_args()
    
    # hyperparameters
    custom_seed = 1234567890
    exp = args.exp.lower()
    model_name = args.model.lower()
    folder_name = args.folder
    is_binary = args.binary.lower() == 'y'

    # assert input arguments
    assert exp in ['train', 'test'], 'ERROR: incorrect experiment type input'
    assert model_name in ['unet', 'cgnet'], 'ERROR: incorrect model name input'
    assert folder_name in ['UNet_w_ROIs_multiclass', 'CGNet_w_ROIs_multiclass', 'UNet_w_ROIs_binary', 'CGNet_w_ROIs_binary'], 'ERROR: incorrect save folder input'

    # -- run experiment -- #
    if exp == 'train':
        run_ROI_experiment(model_name, folder_name, custom_seed, is_binary)
    else:
        test_ROI_experiment(model_name, folder_name, custom_seed, is_binary)
    