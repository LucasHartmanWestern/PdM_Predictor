import argparse
import os
from datetime import datetime

import cv2
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import multiclass_f1_score as f1_score, multiclass_jaccard_index as jaccard_index
from torchsummary import summary
from tqdm import tqdm

from custom_ds import ROI_DS, ROI_DS_Val, FullSize_DS, FullSize_DS_Val
from architectures.unet import UNet
from architectures.nicknet import NickNet
from architectures.cgnet import Context_Guided_Network as CGNet
from utils import *


def train(model, loss_fn, optimizer, train_loader, val_loader, n_epochs, device, patience):
    global save_path

    early_stop_counter = 0
    best_epoch = 0
    best_avg_score = 0.0

    losses_train, losses_val = [], []
    f1_train, f1_val = [], []
    jaccard_train, jaccard_val = [], []

    # --- iterate through all epochs --- #
    log_and_print("{} starting training...".format(datetime.now()))
    for epoch in range(n_epochs):

        # --- training step --- #
        model.train()
        epoch_loss, epoch_f1, epoch_jac = 0.0, 0.0, 0.0
        for _, samples, targets in tqdm(train_loader, desc="epoch {} train progress".format(epoch + 1)):
            samples = samples.to(device=device)
            targets = targets.to(device=device)
            outputs = model(samples)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_f1 += f1_score(outputs, targets, num_classes=7).item()
            epoch_jac += jaccard_index(outputs, targets, num_classes=7).item()
            del samples, targets, outputs

        losses_train.append(epoch_loss / len(train_loader))
        f1_train.append(epoch_f1 / len(train_loader))
        jaccard_train.append(epoch_jac / len(train_loader))

        # --- validation step --- #
        model.eval()
        epoch_loss, epoch_f1, epoch_jac = 0.0, 0.0, 0.0
        with torch.no_grad():
            for _, samples, targets in tqdm(val_loader, desc="epoch {} val progress".format(epoch + 1)):
                samples = samples.to(device=device)
                targets = targets.to(device=device)
                outputs = model(samples)
                epoch_loss += loss_fn(outputs, targets).item()
                epoch_f1 += f1_score(outputs, targets, num_classes=7).item()
                epoch_jac += jaccard_index(outputs, targets, num_classes=7).item()
                del samples, targets, outputs

        losses_val.append(epoch_loss / len(val_loader))
        f1_val.append(epoch_f1 / len(val_loader))
        jaccard_val.append(epoch_jac / len(val_loader))

        # --- save weights for best epoch --- #
        avg_metric_score = (f1_val[epoch] + jaccard_val[epoch]) / 2
        if avg_metric_score >= best_avg_score:
            best_epoch = epoch + 1
            best_avg_score = avg_metric_score
            torch.save(model.state_dict(), os.path.join(save_path, "best_weights.pth"))
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # --- print epoch results --- #
        log_and_print("{} epoch {}/{} metrics:".format(datetime.now(), epoch + 1, n_epochs))
        log_and_print("\t[train] loss: {:.9f}, f1_score: {:.9f}, jaccard_idx: {:.9f}".format(
            losses_train[epoch], f1_train[epoch], jaccard_train[epoch]))
        log_and_print("\t[valid] loss: {:.9f}, f1_score: {:.9f}, jaccard_idx: {:.9f}".format(
            losses_val[epoch], f1_val[epoch], jaccard_val[epoch]))

        # --- check for potential early stopping --- #
        if early_stop_counter == patience:
            log_and_print("{} early stopping at epoch {}".format(datetime.now(), epoch + 1))
            break

    # --- print and plot metrics --- #
    log_and_print("{} training complete.".format(datetime.now()))
    log_and_print("best avg val score: {:.9f}, occurred on epoch {}".format(best_avg_score, best_epoch))
    log_and_print("{} generating plots...".format(datetime.now()))
    metrics_history = [
        ("loss", losses_train, losses_val),
        ("f1_score", f1_train, f1_val),
        ("jaccard_index", jaccard_train, jaccard_val),
    ]
    save_metrics_CSV(metrics_history, save_path)
    log_and_print("{} script finished.".format(datetime.now()))


if __name__ == '__main__':
    # hyperparameters
    n_epochs = 5  # num of epochs
    batch_sz = 1  # batch size
    patience = 5
    num_classes = 7
    # input_shape = (360, 360)
    input_shape = (1080, 1920)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

     # set up deterministic seed
    seed = get_random_seed()
    make_deterministic(seed)

    # set up paths and directories
    save_path = os.path.join("./4_WINTER_PROGRESS/preliminary_RESULTS")
    os.makedirs(save_path, exist_ok=True)

    # set up logger
    setup_basic_logger(save_path)

    # print training hyperparameters
    print_hyperparams({
        "Seed": seed,
        "Device": device,
        "Epochs": n_epochs,
        "Batch Size": batch_sz,
        "Patience": patience,
        "Input Shape": f"({input_shape[0]}, {input_shape[1]}, 3)",
        "Output Shape": f"({input_shape[0]}, {input_shape[1]}, {num_classes})",
        "Number of Classes": 7,
        "Optimizer": "AdamW",
        "Loss Function": "Cross Entropy",
        "Training Dataset Name": "feb28_ds1",
        "Validation Dataset Name": "feb28_ds2",
        "Using ROI's?": True
    })

    # set up dataset(s)
    train_ds = FullSize_DS(["feb28_ds1"])
    val_ds = FullSize_DS_Val("feb28_ds2")
    # train_ds = ROI_DS(["feb28_ds1"])
    # val_ds = ROI_DS_Val("feb28_ds2")
    train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_sz, shuffle=False)

    # compile model
    # model = NickNet(num_classes=num_classes, input_channels=3)
    # model = CGNet(num_classes=num_classes, M=3, N=21)
    model = UNet(num_classes=num_classes, input_channels=3)
    model.to(device=device)

    # init model optimization parameters
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters())

    # run torch summary report
    summary(model, input_size=(3, input_shape[0], input_shape[1]))

    # train model
    train(model, loss_fn, optimizer, train_loader, val_loader, n_epochs, device, patience)
