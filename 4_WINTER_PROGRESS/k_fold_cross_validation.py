import os
from datetime import datetime

import torch
from torchmetrics.functional.classification import multiclass_f1_score as f1_score, multiclass_jaccard_index as jaccard_index
from tqdm import tqdm

from utils import *




def train_fold(model, train_loader, n_epochs, save_path, n_classes, fold_num):
    losses, f1_scores, jaccard_scores = [], [], []

    # --- set up device and model --- #
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device=device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters())

    # --- iterate through all epochs --- #
    log_and_print("{} starting fold {} training...".format(datetime.now(), fold_num))
    for epoch in range(n_epochs):
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
            epoch_f1 += f1_score(outputs, targets, num_classes=n_classes).item()
            epoch_jac += jaccard_index(outputs, targets, num_classes=n_classes).item()
            del samples, targets, outputs
        
        losses.append(epoch_loss / len(train_loader))
        f1_scores.append(epoch_f1 / len(train_loader))
        jaccard_scores.append(epoch_jac / len(train_loader))

        # --- print epoch results --- #
        log_and_print("{} training epoch {}/{} metrics:".format(datetime.now(), epoch + 1, n_epochs))
        log_and_print("\tloss: {:.9f}, f1_score: {:.9f}, jaccard_idx: {:.9f}".format(
            losses[epoch], f1_scores[epoch], jaccard_scores[epoch]))

    # --- save weights and metrics --- #
    log_and_print("{} saving weights and metrics...".format(datetime.now()))
    torch.save(model.state_dict(), os.path.join(save_path, "fold_{}_weights.pth".format(fold_num)))
    metrics_history = {
        "Epoch": list(range(1, epoch + 1)),
        "Losses": losses, 
        "F1 Scores": f1_scores, 
        "Jaccard Index Scores": jaccard_scores, 
    }
    save_metrics_CSV(metrics_history, save_path) # NOT DONE
    log_and_print("{} fold {} training complete.\n".format(datetime.now(), fold_num))


# NOT DONE FIXING THIS
def test_fold(model, test_loader, save_path, n_classes, fold_num):
    metrics_history = {
        "Day": [],
        "F1 Score": {},
        "Jaccard Index": {}
    }
    for class_idx in range(n_classes):
        metrics_history["F1 Score"][f"Class {class_idx}"] = []
        metrics_history["Jaccard Index"][f"Class {class_idx}"] = []

    # --- set up device and model --- #
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device=device)

    # --- iterate through all test samples --- #
    log_and_print("{} starting fold {} testing...".format(datetime.now(), fold_num))
    model.eval()
    with torch.no_grad():
        for day_nums, samples, targets in tqdm(test_loader, desc="testing progress"):
            day_nums = day_nums.tolist()
            assert all(day_num == day_nums[0] for day_num in day_nums), "ERROR: day_nums are not the same for all samples in batch"
            metrics_history["Day"].append(day_nums[0])

            samples = samples.to(device=device)
            targets = targets.to(device=device)
            outputs = model(samples)

            f1_scores = f1_score(outputs, targets, num_classes=n_classes, average="none", zero_division=1.0).tolist()
            jac_scores = jaccard_index(outputs, targets, num_classes=n_classes, average="none", zero_division=1.0).tolist()
            assert len(f1_scores) == len(jac_scores) == n_classes, "ERROR: f1_scores and jac_scores must have length n_classes"
            for class_idx in range(n_classes):
                metrics_history["F1 Score"][f"Class {class_idx}"].append(f1_scores[class_idx])
                metrics_history["Jaccard Index"][f"Class {class_idx}"].append(jac_scores[class_idx])

            del day_nums, samples, targets, outputs

    # --- print results --- #
    log_and_print("{} testing metrics:".format(datetime.now()))
    for class_idx in range(n_classes):
        class_key = f"Class {class_idx}"
        log_and_print("\t[class_{}] f1_score: {:.9f}, jaccard_idx: {:.9f}".format(
            class_idx, np.mean(metrics_history["F1 Score"][class_key]), np.mean(metrics_history["Jaccard Index"][class_key])))

    # --- save metrics --- #
    log_and_print("{} saving metrics...".format(datetime.now()))
    metrics_history = {
        "Day": [],
        "F1 Score": {},
        "Jaccard Index": {}
    }
    save_metrics_CSV(metrics_history, save_path, n_classes)
    create_metric_plots(metrics_history, save_path, n_classes)
    log_and_print("{} fold {} testing complete.\n".format(datetime.now(), fold_num))