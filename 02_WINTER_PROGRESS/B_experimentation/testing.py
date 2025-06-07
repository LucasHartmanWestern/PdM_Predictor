import argparse
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import multiclass_f1_score as f1_score, multiclass_jaccard_index as jaccard_index, binary_confusion_matrix as confusion_matrix
from tqdm import tqdm

from utils import *
from image_processing import preprocess_target, postprocess_seg_mask
from custom_ds_60 import Custom_DS_60
from custom_ds_60_TEST import Custom_DS_60_TEST

# ---------- Helper Methods ---------- #

def save_metrics_CSV(metrics_dict, save_path, n_classes, scenario):
    # Create headers
    headers = ['Day', 'Hour']
    for metric_name in metrics_dict.keys():
        if metric_name != 'Day' and metric_name != 'Hour':  # Skip the day and hour key
            for class_idx in range(n_classes):
                headers.append(f'Class {class_idx} {metric_name}')
    
    # Create the CSV file
    filepath = os.path.join(save_path, f'{scenario}_testing_metrics.csv')
    open(filepath, 'w+').close()
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        # Write data rows
        for i in range(len(metrics_dict['Day'])):
            row = [metrics_dict['Day'][i], metrics_dict['Hour'][i]]
            for metric_name in metrics_dict.keys():
                if metric_name != 'Day' and metric_name != 'Hour':  # Skip the day and hour key
                    for class_idx in range(n_classes):
                        row.append(metrics_dict[metric_name][class_idx][i])
            writer.writerow(row)


def create_metric_plots(metrics_dict, save_path, n_classes, scenario):
    save_path = os.path.join(save_path, f"{scenario}_testing_plots")
    os.makedirs(save_path, exist_ok=True)

    # per-class f1 score bar plot
    plt.clf()
    mean_f1_scores = []
    for class_idx in range(n_classes):
        class_f1_scores = []
        for i in range(len(metrics_dict["Day"])):
            class_f1_scores.append(metrics_dict["F1 Score"][class_idx][i])
        mean_f1_scores.append(np.mean(class_f1_scores))

    bar_container = plt.bar(range(n_classes), mean_f1_scores)
    plt.bar_label(bar_container, fmt="{:.2f}")
    plt.title("Mean Testing F1 Score by Class")
    plt.xlabel("Class")
    plt.ylabel("F1 Score")
    plt.savefig(os.path.join(save_path, "f1_bar_plot.png"))


# ---------- Testing Method ---------- #

def test(model, test_loader, save_path, n_classes, scenario):
    metrics_history = {
        "Hour": [],
        "Day": [],
        "F1 Score": {},
        "Jaccard Index": {}
    }
    for class_idx in range(n_classes):
        metrics_history["F1 Score"][class_idx] = []
        metrics_history["Jaccard Index"][class_idx] = []

    model.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    # --- iterate through all test samples --- #
    start_time = datetime.now()
    log_and_print("{} starting testing...\n".format(datetime.now()))
    model.eval()
    with torch.no_grad():
        for day_num, hour_num, sample, target in tqdm(test_loader, desc="testing progress"):
            metrics_history["Day"].append(day_num.item())
            metrics_history["Hour"].append(hour_num[0])
            target = preprocess_target(target, model.use_rois)
            output = model(sample)
            target = postprocess_seg_mask(target, n_classes, model.use_rois)
            output = postprocess_seg_mask(output, n_classes, model.use_rois)
            f1_scores = f1_score(output, target, num_classes=n_classes, average="none", zero_division=1.0).tolist()
            jac_scores = jaccard_index(output, target, num_classes=n_classes, average="none", zero_division=1.0).tolist()
            assert len(f1_scores) == len(jac_scores) == n_classes, "ERROR: f1_scores and jac_scores must have length n_classes"
            for class_idx in range(n_classes):
                metrics_history["F1 Score"][class_idx].append(f1_scores[class_idx])
                metrics_history["Jaccard Index"][class_idx].append(jac_scores[class_idx])

            del day_num, hour_num, sample, target, output

    # --- print results --- #
    log_and_print("\n{} mean testing metrics:".format(datetime.now()))
    for class_idx in range(n_classes):
        mean_f1 = np.mean(metrics_history["F1 Score"][class_idx])
        std_f1 = np.std(metrics_history["F1 Score"][class_idx])
        mean_jac = np.mean(metrics_history["Jaccard Index"][class_idx])
        std_jac = np.std(metrics_history["Jaccard Index"][class_idx])
        log_and_print("\t[class_{}] f1_score: {:.9f} +/- {:.5f}, jaccard_idx: {:.9f} +/- {:.5f}".format(
            class_idx, mean_f1, std_f1, mean_jac, std_jac))

    # --- save metrics --- #
    total_testing_time = datetime.now() - start_time
    log_and_print("\n{} testing complete.".format(datetime.now()))
    log_and_print("total testing time: {}".format(total_testing_time))
    log_and_print("{} saving metrics and generating plots...".format(datetime.now()))
    save_metrics_CSV(metrics_history, save_path, n_classes, scenario)
    create_metric_plots(metrics_history, save_path, n_classes, scenario)
    log_and_print("{} testing script finished.\n".format(datetime.now()))

# ---------- Main Method ---------- #

if __name__ == "__main__":
    # get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, required=True, help="model name (str)")
    parser.add_argument("-rois", type=str, required=True, help="use rois (y/n)")
    # parser.add_argument("-binary", type=str, required=True, help="use binary targets (y/n)")
    parser.add_argument("-dataset", type=str, required=True, help="dataset folder name (str)")
    parser.add_argument("-trial", type=int, required=True, help="trial number (int)")
    args = parser.parse_args()

    # THIS IS FOR MULTICLASS TARGETS ONLY!!!

    # get hyperparameters
    model_name = args.model.lower()
    use_rois = args.rois.lower() == "y"
    # binary_targets = args.binary.lower() == "y"
    binary_targets = False
    dataset_name = args.dataset.lower()
    trial = args.trial
    classes = 2 if binary_targets else 7

    # make extra testing folder for SMS and CGS
    scenario = ''
    if dataset_name == 'mar28_ds4': # for SMS Testing
        scenario = 'SMS'
    elif dataset_name == 'mar28_ds5': # for CGS Testing
        scenario = 'CGS'

    # set up save path
    results_folder_name = f"{model_name}_{'rois' if use_rois else 'full'}_{'binary' if binary_targets else 'multiclass'}"
    save_location = os.path.join(".", "RESULTS", f"trial_{trial}", results_folder_name)

     # set up logger
    setup_basic_logger(save_location, f'testing_{scenario}')
    log_and_print(f"\n--- Testing {results_folder_name} ---\n")

    # set up data loaders
    if dataset_name == 'mar28_ds4': # for SMS Testing
        test_ds = Custom_DS_60_TEST(dataset_name, binary_targets)
    elif dataset_name == 'mar28_ds5': # for CGS Testing
        test_ds = Custom_DS_60(dataset_name, 'test', binary_targets)
    else:
        test_ds = Custom_DS_60(dataset_name, 'test', binary_targets)
    test_ds_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # set up model and load weights
    model = get_model(model_name, num_classes=classes, use_rois=use_rois)
    model.load_state_dict(torch.load(os.path.join(save_location, "best_weights.pth"), weights_only=True))

    # test model
    test(model, test_ds_loader, save_location, classes, scenario)