import argparse
import csv
import os
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryPrecisionRecallCurve
from torchmetrics.functional.classification import binary_f1_score, binary_jaccard_index
from tqdm import tqdm

from utils.data_import_util import get_xy_data
from utils.log_util import setup_basic_logger, log_and_print, print_hyperparams
from custom_ds import CustomDS
from unet_model import UNet


def plot_metric(metric, metric_name):
    global model_version, save_path
    plt.clf()
    metric.plot(score=True)
    plt.savefig(os.path.join(save_path, 'model_{}_test_{}.png'.format(model_version, metric_name)))

    torch.save(metric.state_dict(), os.path.join(save_path, '{}.pth'.format(metric_name)))
    metric.reset()


def print_hist(metric_vals, metric_name):
    global model_version, save_path
    plt.clf()
    plt.figure(figsize=(8, 6))
    values, bins, bars = plt.hist(metric_vals, edgecolor='white')
    plt.xlabel(metric_name)
    plt.ylabel('number of predictions')
    plt.bar_label(bars)
    plt.suptitle('Prediction Histogram for {}'.format(metric_name), fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(save_path, 'model_{}_test_{}.png'.format(model_version, metric_name)))

    csv_path = os.path.join(save_path, '{}.csv'.format(metric_name))
    open(csv_path, 'w+').close()  # overwrite/ make new blank file
    with open(csv_path, 'a', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([metric_name])
        for val in metric_vals:
            writer.writerow([val])


def test(model, test_loader, device):
    global model_version, save_path

    f1_scores, jac_idxs = [], []
    bprc = BinaryPrecisionRecallCurve(thresholds=1000).to(device)
    bprc.persistent(True)
    model.eval()
    log_and_print("{} starting testing...".format(datetime.now()))

    # --- validation step --- #
    with torch.no_grad():
        for image, target in tqdm(test_loader, desc="test progress"):
            image = image.to(device=device)
            target = target.to(device=device)
            output = model(image)
            f1_scores.append(binary_f1_score(output, target).item())
            jac_idxs.append(binary_jaccard_index(output, target).item())
            bprc.update(output, target.long())
            del image, target, output

    # --- print epoch results --- #
    log_and_print("{} testing metrics:".format(datetime.now()))
    log_and_print("\tf1_score:\t{:.9f} (best) | {:.9f} (worst) | {:.9f} (avg)".format(
        np.max(f1_scores), np.min(f1_scores), np.mean(f1_scores)))
    log_and_print("\tjaccard_idx:\t{:.9f} (best) | {:.9f} (worst) | {:.9f} (avg)".format(
        np.max(jac_idxs), np.min(jac_idxs), np.mean(jac_idxs)))

    # --- save metric outputs --- #
    log_and_print("{} generating prediction plots and figures...".format(datetime.now()))
    plot_metric(bprc, 'bprc')
    print_hist(f1_scores, 'f1_score')
    # print_hist(jac_idxs, 'jaccard_index')
    log_and_print("{} testing complete.".format(datetime.now()))


if __name__ == '__main__':
    # hyperparameters
    model_version = 1
    input_shape = (512, 512)
    dataset_name = 'sm_CGS_ds'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set up paths and directories
    save_path = os.path.join('.', 'model_{}'.format(model_version))

    # set up logger and deterministic seed
    setup_basic_logger(os.path.join(save_path, 'testing.log'))

    # print training hyperparameters
    print_hyperparams(
        model_ver=model_version, input_shape=input_shape, dataset_name=dataset_name, device=device
    )

    # set up dataset(s)
    x_test, y_test, _, _ = get_xy_data(dataset_name, partition='test')
    test_ds = CustomDS(x_test, y_test, dataset_name, input_shape)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # compile model
    model = UNet()
    weights_file = os.path.join(save_path, 'best_weights.pth')
    model.load_state_dict(torch.load(weights_file, map_location=device, weights_only=True))
    model.to(device=device)

    # test model
    test(model, test_loader, device)
