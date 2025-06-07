import argparse
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import binary_f1_score as f1_score, binary_jaccard_index as jaccard_index, binary_confusion_matrix as confusion_matrix
from tqdm import tqdm

from utils import *
from image_processing import preprocess_target, postprocess_seg_mask
from custom_ds_60 import Custom_DS_60
from custom_ds_60_TEST import Custom_DS_60_TEST

# ---------- Helper Methods ---------- #

def save_metrics_CSV(metrics_dict, save_path, scenario):
    # Create headers
    headers = ['Day', 'Hour', 'F1 Score', 'Jaccard Index', 'TP', 'FP', 'FN', 'TN']
    
    # Create the CSV file
    filepath = os.path.join(save_path, f'{scenario}_testing_metrics.csv')
    open(filepath, 'w+').close()
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        # Write data rows
        for i in range(len(metrics_dict['Day'])):
            row = []
            for metric_name in metrics_dict.keys():
                row.append(metrics_dict[metric_name][i])
            writer.writerow(row)


# ---------- Testing Method ---------- #

def test(model, test_loader, save_path, n_classes, scenario):
    metrics_history = {
        "Hour": [],
        "Day": [],
        "F1 Score": [],
        "Jaccard Index": [],
        "TP": [], "FP": [], "FN": [], "TN": []
    }

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

            metrics_history["F1 Score"].append(f1_score(output, target).item())
            metrics_history["Jaccard Index"].append(jaccard_index(output, target).item())

            conf_mat = confusion_matrix(output, target).tolist()
            metrics_history["TP"].append(conf_mat[0][0])
            metrics_history["FP"].append(conf_mat[0][1])
            metrics_history["FN"].append(conf_mat[1][0])
            metrics_history["TN"].append(conf_mat[1][1])

            del day_num, hour_num, sample, target, output

    # --- print results --- #
    log_and_print("\n{} mean testing metrics:".format(datetime.now()))
    log_and_print("\tf1_score: {:.9f} +/- {:.5f}, jaccard_idx: {:.9f} +/- {:.5f}".format(
        np.mean(metrics_history["F1 Score"]), np.std(metrics_history["F1 Score"]), 
        np.mean(metrics_history["Jaccard Index"]), np.std(metrics_history["Jaccard Index"])))

    # --- save metrics --- #
    total_testing_time = datetime.now() - start_time
    log_and_print("\n{} testing complete.".format(datetime.now()))
    log_and_print("total testing time: {}".format(total_testing_time))
    log_and_print("{} saving metrics and generating plots...".format(datetime.now()))
    save_metrics_CSV(metrics_history, save_path, scenario)
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

    # THIS IS FOR BINARY TARGETS ONLY!!!

    # get hyperparameters
    model_name = args.model.lower()
    use_rois = args.rois.lower() == "y"
    # binary_targets = args.binary.lower() == "y"
    binary_targets = True
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