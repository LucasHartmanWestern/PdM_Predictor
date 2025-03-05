import os
from datetime import datetime

import torch
from torchmetrics.functional.classification import multiclass_f1_score as f1_score, multiclass_jaccard_index as jaccard_index
from tqdm import tqdm

from utils import *

# ---------- Testing Helper Methods ---------- #

def save_metrics_CSV(metrics_dict, save_path, n_classes):
    # Create headers
    headers = ['Day']
    for metric_name in metrics_dict.keys():
        if metric_name != 'Day':  # Skip the day key
            for class_idx in range(n_classes):
                headers.append(f'Class {class_idx} {metric_name}')
    
    # Create the CSV file
    filepath = os.path.join(save_path, 'testing_metrics.csv')
    open(filepath, 'w+').close()
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        # Write data rows
        for i in range(len(metrics_dict['Day'])):
            row = [metrics_dict['Day'][i]]
            for metric_name in metrics_dict.keys():
                if metric_name != 'Day':  # Skip the day key
                    for class_idx in range(n_classes):
                        row.append(metrics_dict[metric_name][f'Class {class_idx}'][i])
            writer.writerow(row)


def create_metric_plots(metrics_dict, save_path, n_classes):
    save_path = os.path.join(save_path, "testing_plots")
    os.makedirs(save_path, exist_ok=True)

    # per-day metric plots
    for key, val in metrics_dict.items():
        if key != "Day":
            plt.clf()
            legend_list = []
            for class_idx in range(n_classes):
                legend_list.append(f"Class {class_idx}")
                plt.plot(val[f"Class {class_idx}"])
            plt.title(f"Testing {key} by Day")
            plt.ylabel(key)
            plt.xlabel("Day")
            plt.legend(legend_list)
            plt.savefig(os.path.join(save_path, f"{key.lower().split(' ')[0]}_plot.png"))

    # per-class f1 score bar plot
    plt.clf()
    mean_f1_scores = []
    for class_idx in range(n_classes):
        class_f1_scores = []
        for i in range(len(metrics_dict["Day"])):
            class_f1_scores.append(metrics_dict["F1 Score"][f"Class {class_idx}"][i])
        mean_f1_scores.append(np.mean(class_f1_scores))

    plt.bar(range(n_classes), mean_f1_scores)
    plt.title("Mean Testing F1 Score by Class")
    plt.xlabel("Class")
    plt.ylabel("F1 Score")
    plt.savefig(os.path.join(save_path, "f1_bar_plot.png"))


# ---------- Testing Main Methods ---------- #

def test(model, test_loader, device, save_path, n_classes):
    metrics_history = {
        "Day": [],
        "F1 Score": {},
        "Jaccard Index": {}
    }
    for class_idx in range(n_classes):
        metrics_history["F1 Score"][f"Class {class_idx}"] = []
        metrics_history["Jaccard Index"][f"Class {class_idx}"] = []

    # --- iterate through all test samples --- #
    log_and_print("{} starting testing...".format(datetime.now()))
    model.eval()
    with torch.no_grad():
        for day_nums, samples, targets in tqdm(test_loader, desc="testing progress"):
            day_nums = day_nums.tolist()
            print(day_nums)
            assert all(day_nums[0] in day_nums), "ERROR: day_nums are not the same for all samples in batch"
            metrics_history["Day"].append(day_nums[0])

            samples = samples.to(device=device)
            targets = targets.to(device=device)
            outputs = model(samples)

            f1_scores = f1_score(outputs, targets, num_classes=n_classes, average="none").tolist()
            jac_scores = jaccard_index(outputs, targets, num_classes=n_classes, average="none").tolist()
            assert len(f1_scores) == len(jac_scores) == n_classes, "ERROR: f1_scores and jac_scores must have length n_classes"
            for class_idx in range(n_classes):
                metrics_history["F1 Score"][f"Class {class_idx}"].append(f1_scores[class_idx])
                metrics_history["Jaccard Index"][f"Class {class_idx}"].append(jac_scores[class_idx])

            del day_nums, samples, targets, outputs

    # --- print results --- #
    log_and_print("{} mean testing metrics:".format(datetime.now()))
    for class_idx in range(n_classes):
        class_key = f"Class {class_idx}"
        log_and_print("\t[class_{}] f1_score: {:.9f}, jaccard_idx: {:.9f}".format(
            class_idx, np.mean(metrics_history["F1 Score"][class_key]), np.mean(metrics_history["Jaccard Index"][class_key])))

    # --- save metrics --- #
    log_and_print("{} testing complete.".format(datetime.now()))
    log_and_print("{} saving metrics and generating plots...".format(datetime.now()))
    save_metrics_CSV(metrics_history, save_path, n_classes)
    create_metric_plots(metrics_history, save_path)
    log_and_print("{} testing script finished.".format(datetime.now()))
