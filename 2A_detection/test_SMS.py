import argparse
import csv
import os
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryPrecisionRecallCurve
from torchmetrics.functional.classification import binary_f1_score
from tqdm import tqdm

from utils.log_util import setup_basic_logger, log_and_print, print_hyperparams
from custom_ds import SMSTestDS
from architectures.nested_unet import UNet_new, NestedUNet
from architectures.plain_unet import UNet
from architectures.cgnet import Context_Guided_Network
from architectures.custom_arch import QuickNet


def make_scenario_csvs(data):
    for sc in range(len(data)):
        csv_path = os.path.join(save_path, 'sc{}_data.csv'.format(sc + 1))
        open(csv_path, 'w+').close()  # overwrite/ make new blank file
        with open(csv_path, 'a', encoding='UTF8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['day', 'avg_f1_score', 'percent_img_fouling'])
            writer.writerows(data[sc])


def get_fouling_percentage(tgt_image):
    pixel_count = np.count_nonzero(tgt_image > 0)
    return 100 * (pixel_count / (512 * 512))


def plot_metric(metric, metric_name):
    global save_path
    plt.clf()
    metric.plot(score=True)
    plt.savefig(os.path.join(save_path, 'test_{}.png'.format(metric_name)))
    torch.save(metric.state_dict(), os.path.join(save_path, '{}.pth'.format(metric_name)))
    metric.reset()


def print_hist(metric_vals, metric_name):
    global save_path
    plt.clf()
    plt.figure(figsize=(8, 6))
    values, bins, bars = plt.hist(metric_vals, edgecolor='white')
    plt.xlabel(metric_name)
    plt.ylabel('number of predictions')
    plt.bar_label(bars)
    plt.suptitle('Prediction Histogram for {}'.format(metric_name), fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(save_path, 'test_{}.png'.format(metric_name)))

    # csv_path = os.path.join(save_path, '{}.csv'.format(metric_name))
    # open(csv_path, 'w+').close()  # overwrite/ make new blank file
    # with open(csv_path, 'a', encoding='UTF8', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow([metric_name])
    #     for val in metric_vals:
    #         writer.writerow([val])


def test(model, test_loader, device):
    global save_path

    prev_day = 1
    curr_scenario = 0
    scenarios_data = [[], [], []]
    day_f1_scores = []

    f1_scores = []
    bprc = BinaryPrecisionRecallCurve(thresholds=1000).to(device)
    bprc.persistent(True)
    model.eval()
    log_and_print("{} starting testing...".format(datetime.now()))

    # --- performing testing --- #
    with torch.no_grad():
        for image, target, day in tqdm(test_loader, desc="test progress"):
            image = image.to(device=device)
            target = target.to(device=device)
            output = model(image)

            bprc.update(output, target.long())
            f1 = binary_f1_score(output, target).item()
            f1_scores.append(f1)

            if day != prev_day:
                scenarios_data[curr_scenario].append([
                    prev_day.item(), round(np.mean(day_f1_scores), 5), prev_day_fouling])
                day_f1_scores = []

                if day < prev_day:  # started new scenario
                    curr_scenario += 1

            day_f1_scores.append(f1)
            prev_day_fouling = round(get_fouling_percentage(target.cpu().numpy()), 5)
            prev_day = day

            del image, target, output

    scenarios_data[curr_scenario].append([
        prev_day.item(), round(np.mean(day_f1_scores), 5), prev_day_fouling])
    make_scenario_csvs(scenarios_data)

    # --- print epoch results --- #
    log_and_print("{} testing metrics:".format(datetime.now()))
    log_and_print("\tf1_score:\t{:.9f} (best) | {:.9f} (worst) | {:.9f} (avg)".format(
        np.max(f1_scores), np.min(f1_scores), np.mean(f1_scores)))

    # --- save metric outputs --- #
    log_and_print("{} generating prediction plots and figures...".format(datetime.now()))
    plot_metric(bprc, 'bprc')
    print_hist(f1_scores, 'f1_score')
    log_and_print("{} testing complete.".format(datetime.now()))


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-mn', type=str, help='model name', required=True)
    args = parser.parse_args()

    # hyperparameters
    input_shape = (512, 512)
    dataset_name = 'sm_SMS_ds'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set up paths and directories
    base_path = os.path.join('.', 'experiments', args.mn)
    assert os.path.exists(base_path), 'ERROR: directory does not exist: {}'.format(base_path)
    save_path = os.path.join(base_path, 'testing')
    os.makedirs(save_path, exist_ok=True)

    # set up logger and deterministic seed
    setup_basic_logger(os.path.join(save_path, 'testing.log'))

    # print training hyperparameters
    print_hyperparams(
        model_name=args.mn, input_shape=input_shape, dataset_name=dataset_name, device=device
    )

    # set up dataset(s)
    test_ds = SMSTestDS()
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # compile model
    if args.mn == 'new_unet':
        model = UNet_new()
    elif args.mn == 'nested_unet':
        model = NestedUNet()
    elif args.mn == 'cgnet':
        model = Context_Guided_Network()
    elif args.mn == 'quicknet':
        model = QuickNet()
    else:
        model = UNet()
    weights_file = os.path.join(base_path, 'training', 'best_weights.pth')
    model.load_state_dict(torch.load(weights_file, map_location=device, weights_only=True))
    model.to(device=device)

    # test model
    test(model, test_loader, device)
