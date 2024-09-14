import argparse
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import binary_f1_score, binary_jaccard_index
from torchsummary import summary
from tqdm import tqdm

from custom_ds import CustomDS
from utils.data_import_util import get_xy_data
from architectures.nested_unet import UNet_new, NestedUNet
from architectures.plain_unet import UNet
from architectures.cgnet import Context_Guided_Network
from architectures.custom_unet import FastNestedUNet
from architectures.mobilenetv3 import MobileNetV3_Large, MobileNetV3_Small
from utils.log_util import log_and_print, setup_basic_logger, print_hyperparams
from utils.misc_util import print_metric_plots
from utils.seed_util import get_random_seed, make_deterministic


def train(model, loss_fn, optimizer, train_loader, val_loader, n_epochs, device):
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
        for images, targets in tqdm(train_loader, desc="epoch {} train progress".format(epoch + 1)):
            images = images.to(device=device)
            targets = targets.to(device=device)
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_f1 += binary_f1_score(outputs, targets).item()
            epoch_jac += binary_jaccard_index(outputs, targets).item()
            del images, targets, outputs

        losses_train.append(epoch_loss / len(train_loader))
        f1_train.append(epoch_f1 / len(train_loader))
        jaccard_train.append(epoch_jac / len(train_loader))

        # --- validation step --- #
        model.eval()
        epoch_loss, epoch_f1, epoch_jac = 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="epoch {} val progress".format(epoch + 1)):
                images = images.to(device=device)
                targets = targets.to(device=device)
                outputs = model(images)
                epoch_loss += loss_fn(outputs, targets).item()
                epoch_f1 += binary_f1_score(outputs, targets).item()
                epoch_jac += binary_jaccard_index(outputs, targets).item()
                del images, targets, outputs

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
        if early_stop_counter == 10:
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
    print_metric_plots(metrics_history, save_path)
    log_and_print("{} script finished.".format(datetime.now()))


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-mn', type=str, help='model name', required=True)
    parser.add_argument('-rs', type=str, help='random seed (y/n)', required=True)
    args = parser.parse_args()
    assert args.mn in [
        'new_unet', 'nested_unet', 'cgnet', 'fast_nested_unet', 'mnet3_small', 'mnet3_large', 'plain_unet'
    ], 'ERROR: incorrect mn input'
    assert args.rs in ['y', 'n'], 'ERROR: incorrect rs input'

    # hyperparameters
    n_epochs = 100  # num of epochs
    batch_sz = 4  # batch size
    val_split = 0.2  # split for validation dataset
    input_shape = (512, 512)  # same size used in U-Net paper for training
    dataset_name = 'sm_CGS_ds'
    loss_fn_name = 'binary_cross_entropy'
    optimizer_name = 'default_adam_w'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set up paths and directories
    save_path = os.path.join('.', 'experiments', args.mn, 'training')
    os.makedirs(save_path, exist_ok=True)

    # set up logger and deterministic seed
    setup_basic_logger(os.path.join(save_path, 'training.log'))  # initialize logger
    seed = 987654321 if args.rs == 'n' else get_random_seed()
    make_deterministic(seed)  # set deterministic seed

    # print training hyperparameters
    print_hyperparams(
        model_name=args.mn, num_epochs=n_epochs, batch_size=batch_sz,
        seed=seed, validation_split=val_split, input_shape=input_shape, dataset_name=dataset_name,
        loss_fn_name=loss_fn_name, optimizer_name=optimizer_name, device=device
    )

    # set up dataset(s)
    x_train, y_train, x_val, y_val = get_xy_data(dataset_name, partition='train', split=val_split, seed=seed)
    train_ds = CustomDS(x_train, y_train, dataset_name)
    val_ds = CustomDS(x_val, y_val, dataset_name)
    train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_sz, shuffle=False)

    # compile model
    if args.mn == 'new_unet':
        model = UNet_new()
    elif args.mn == 'nested_unet':
        model = NestedUNet()
    elif args.mn == 'cgnet':
        model = Context_Guided_Network()
    elif args.mn == 'fast_nested_unet':
        model = FastNestedUNet()
    elif args.mn == 'mnet3_small':
        model = MobileNetV3_Small()
    elif args.mn == 'mnet3_large':
        model = MobileNetV3_Large()
    else:
        model = UNet()
    model.to(device=device)

    # init model optimization parameters
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(params=model.parameters())

    # run torch summary report
    summary(model, input_size=(3, input_shape[0], input_shape[1]))

    # train model
    train(model, loss_fn, optimizer, train_loader, val_loader, n_epochs, device)
