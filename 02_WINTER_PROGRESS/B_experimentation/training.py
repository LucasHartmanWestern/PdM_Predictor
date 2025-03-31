import argparse
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import multiclass_f1_score as f1_score, multiclass_jaccard_index as jaccard_index
from tqdm import tqdm

from utils import *
from custom_ds_60 import Custom_DS_60
from image_processing import preprocess_target, postprocess_seg_mask

# ---------- Helper Methods ---------- #

def save_metrics_CSV(metrics_dict, save_path):
    # Create headers
    headers = ['Epoch']
    for metric_name in metrics_dict.keys():
        if metric_name != 'Epoch':  # Skip the epochs key
            headers.extend([f'Train {metric_name}', f'Val {metric_name}'])
    
    # Create the CSV file
    filepath = os.path.join(save_path, 'training_metrics.csv')
    open(filepath, 'w+').close()
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        # Write data rows
        for i in range(len(metrics_dict['Epoch'])):
            row = [metrics_dict['Epoch'][i]]
            for metric_name in metrics_dict.keys():
                if metric_name != 'Epoch':  # Skip the epochs key
                    row.extend([
                        metrics_dict[metric_name]['Train'][i],
                        metrics_dict[metric_name]['Val'][i]
                    ])
            writer.writerow(row)
            

def create_metric_plots(metrics_dict, save_path):
    save_path = os.path.join(save_path, "training_plots")
    os.makedirs(save_path, exist_ok=True)

    for key, values in metrics_dict.items():
        if key != "Epoch":
            plt.clf()
            plt.title(f"Training {key}")
            plt.ylabel(key)
            plt.xlabel("Epoch")
            plt.plot(values["Train"], label="Train")
            plt.plot(values["Val"], label="Val")

            if key == "Loss":
                best_score_train = min(values["Train"])
                best_score_val = min(values["Val"])
            else:
                best_score_train = max(values["Train"])
                best_score_val = max(values["Val"])

            best_epoch_train = values["Train"].index(best_score_train)
            best_epoch_val = values["Val"].index(best_score_val)

            # plt.axvline(x=best_epoch_train, color='blue', linestyle='--', label='Best Epoch Train')
            # plt.axvline(x=best_epoch_val, color='orange', linestyle='--', label='Best Epoch Val')

            # plt.plot(best_epoch_train, best_score_train, '*', label='Best Epoch Train')
            # plt.plot(best_epoch_val, best_score_val, '*', label='Best Epoch Val')

            plt.plot(best_epoch_train, best_score_train, '*')
            plt.plot(best_epoch_val, best_score_val, '*')

            plt.legend()
            plt.savefig(os.path.join(save_path, f"{key.lower().split(' ')[0]}_plot.png"))

# ---------- Training Method ---------- #

def train(model, n_classes, train_loader, val_loader, save_path, n_epochs, n_patience=None):
    best_loss, best_f1_score, best_jac_score = 0.0, 0.0, 0.0
    best_loss_epoch, best_f1_epoch, best_jac_epoch, epochs_without_improvement = 0, 0, 0, 0
    losses_train, losses_val = [], []
    f1_train, f1_val = [], []
    jaccard_train, jaccard_val = [], []

    # hard-coded parameters
    model.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())

    # --- iterate through all epochs --- #
    log_and_print("{} starting training...\n".format(datetime.now()))
    for epoch in range(n_epochs):

        # --- training step --- #
        model.train()
        epoch_loss, epoch_f1, epoch_jac = 0.0, 0.0, 0.0
        for _, _, sample, target in tqdm(train_loader, desc="epoch {} train progress".format(epoch + 1)):
            target = preprocess_target(target, model.use_rois)
            output = model(sample)
            loss = loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            target = postprocess_seg_mask(target, n_classes, model.use_rois)
            output = postprocess_seg_mask(output, n_classes, model.use_rois)
            epoch_f1 += f1_score(output, target, num_classes=n_classes, zero_division=1).item()
            epoch_jac += jaccard_index(output, target, num_classes=n_classes, zero_division=1).item()
            del sample, target, output
        
        losses_train.append(epoch_loss / len(train_loader))
        f1_train.append(epoch_f1 / len(train_loader))
        jaccard_train.append(epoch_jac / len(train_loader))

        # --- validation step --- #
        model.eval()
        epoch_loss, epoch_f1, epoch_jac = 0.0, 0.0, 0.0
        with torch.no_grad():
            for _, _, sample, target in tqdm(val_loader, desc="epoch {} val progress".format(epoch + 1)):
                target = preprocess_target(target, model.use_rois)
                output = model(sample)
                epoch_loss += loss_fn(output, target).item()
                target = postprocess_seg_mask(target, n_classes, model.use_rois)
                output = postprocess_seg_mask(output, n_classes, model.use_rois)
                epoch_f1 += f1_score(output, target, num_classes=n_classes, zero_division=1).item()
                epoch_jac += jaccard_index(output, target, num_classes=n_classes, zero_division=1).item()
                del sample, target, output
        
        losses_val.append(epoch_loss / len(val_loader))
        f1_val.append(epoch_f1 / len(val_loader))
        jaccard_val.append(epoch_jac / len(val_loader))

        # --- print epoch results --- #
        log_and_print("{} epoch {}/{} metrics:".format(datetime.now(), epoch + 1, n_epochs))
        log_and_print("\t[train] loss: {:.9f}, f1_score: {:.9f}, jaccard_idx: {:.9f}".format(
            losses_train[epoch], f1_train[epoch], jaccard_train[epoch]))
        log_and_print("\t[valid] loss: {:.9f}, f1_score: {:.9f}, jaccard_idx: {:.9f}".format(
            losses_val[epoch], f1_val[epoch], jaccard_val[epoch]))
        
        # --- update best metrics scores and save weights --- #
        if epoch == 0 or losses_val[epoch] < best_loss:
            best_loss = losses_val[epoch]
            best_loss_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_path, "best_weights.pth"))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if f1_val[epoch] > best_f1_score:
            best_f1_score = f1_val[epoch]
            best_f1_epoch = epoch + 1

        if jaccard_val[epoch] > best_jac_score:
            best_jac_score = jaccard_val[epoch]
            best_jac_epoch = epoch + 1

        # --- check for early stopping --- #
        if n_patience is not None and epochs_without_improvement == n_patience:
            log_and_print("{} early stopping at epoch {}".format(datetime.now(), epoch + 1))
            break

    # --- print best metrics --- #
    log_and_print("\n{} training complete.\n".format(datetime.now()))
    log_and_print("lowest loss: {:.9f}, occurred on epoch {}".format(best_loss, best_loss_epoch))
    log_and_print("highest f1 score: {:.9f}, occurred on epoch {}".format(best_f1_score, best_f1_epoch))
    log_and_print("highest jaccard idx score: {:.9f}, occurred on epoch {}".format(best_jac_score, best_jac_epoch))

    # --- save metrics --- #
    log_and_print("\n{} saving metrics and generating plots...".format(datetime.now()))
    metrics_history = {
        "Epoch": list(range(1, epoch + 2)),
        "Loss": {
            "Train": losses_train, 
            "Val": losses_val
        },
        "F1 Score": {
            "Train": f1_train, 
            "Val": f1_val
        },
        "Jaccard Index": {
            "Train": jaccard_train, 
            "Val": jaccard_val
        }
    }
    save_metrics_CSV(metrics_history, save_path)
    create_metric_plots(metrics_history, save_path)
    log_and_print("{} training script finished.\n".format(datetime.now()))

# ---------- Main Method ---------- #

if __name__ == "__main__":
    # get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, required=True, help="model name (str)")
    parser.add_argument("-rois", type=str, required=True, help="use rois (y/n)")
    parser.add_argument("-binary", type=str, required=True, help="use binary targets (y/n)")
    parser.add_argument("-dataset", type=str, required=True, help="dataset folder name (str)")
    parser.add_argument("-epochs", type=int, required=True, help="number of epochs (int)")
    parser.add_argument("-patience", type=int, default=None, help="early stopping patience (int)")
    args = parser.parse_args()

    # get hyperparameters
    model_name = args.model.lower()
    use_rois = args.rois.lower() == "y"
    binary_targets = args.binary.lower() == "y"
    dataset_name = args.dataset.lower()
    epochs = args.epochs
    patience = args.patience
    classes = 2 if binary_targets else 7

    # set up save path
    results_folder_name = f"{model_name}_{'rois' if use_rois else 'full'}_{'binary' if binary_targets else 'multiclass'}"
    save_location = os.path.join(".", "RESULTS", results_folder_name)
    os.makedirs(save_location, exist_ok=True)

    # set up seed for reproducibility
    seed = get_seed_from_dataset(dataset_name)
    make_deterministic(seed)

     # set up logger
    setup_basic_logger(save_location, 'training')

    # print training hyperparameters
    print_hyperparams({
        "Seed": seed,
        "Model": model_name,
        "Epochs": epochs,
        "Patience": patience,
        "Binary Targets": binary_targets,
        "Number of Classes": classes,
        "Using ROIs": use_rois,
        "Dataset Name": dataset_name,
        "Save Path": save_location,
        "Batch Size": "1 (hard-coded)",
        "Optimizer": "AdamW (hard-coded)",
        "Loss Function": "Cross Entropy Loss (hard-coded)",
    })

    # set up data loaders
    train_ds = Custom_DS_60(dataset_name, 'train', binary_targets)
    val_ds = Custom_DS_60(dataset_name, 'val', binary_targets)
    train_ds_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_ds_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # set up model and train
    model = get_model(model_name, num_classes=classes, use_rois=use_rois)
    train(model, classes, train_ds_loader, val_ds_loader, save_location, epochs, patience)

   