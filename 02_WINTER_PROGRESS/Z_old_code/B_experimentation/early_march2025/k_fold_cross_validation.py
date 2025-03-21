import os
from datetime import datetime

import torch
from torchmetrics.functional.classification import multiclass_f1_score as f1, dice
from tqdm import tqdm

from utils import *


def train_fold(model, train_loader, n_epochs, save_path, n_classes, fold_num):
    losses, f1_scores, dice_scores = [], [], []

    # --- set up device and model --- #
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device=device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters())

    # --- iterate through all epochs --- #
    log_and_print("{} starting fold {} training...".format(datetime.now(), fold_num))
    for epoch in range(n_epochs):
        model.train()
        epoch_loss, epoch_f1, epoch_dice = 0.0, 0.0, 0.0
        for _, _, samples, targets in tqdm(train_loader, desc="epoch {} train progress".format(epoch + 1)):
            samples = samples.to(device=device)
            targets = targets.to(device=device)
            outputs = model(samples)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_f1 += f1(outputs, targets, num_classes=n_classes).item()
            epoch_dice += dice(outputs, targets, num_classes=n_classes).item()
            del samples, targets, outputs
        
        losses.append(epoch_loss / len(train_loader))
        f1_scores.append(epoch_f1 / len(train_loader))
        dice_scores.append(epoch_dice / len(train_loader))

        # --- print epoch results --- #
        log_and_print("{} training epoch {}/{} metrics:".format(datetime.now(), epoch + 1, n_epochs))
        log_and_print("\tloss: {:.9f}, f1_score: {:.9f}, jaccard_idx: {:.9f}".format(
            losses[epoch], f1_scores[epoch], dice_scores[epoch]))

    # --- save weights and metrics --- #
    log_and_print("{} saving weights and metrics...".format(datetime.now()))
    torch.save(model.state_dict(), os.path.join(save_path, "fold_{}_weights.pth".format(fold_num)))
    metrics_history = {
        "Fold": fold_num,
        "Epoch": list(range(1, epoch + 1)),
        "Loss": losses, 
        "F1 Score": f1_scores, 
        "Dice Score": dice_scores, 
    }
    save_metrics_CSV(metrics_history, save_path) # NOT DONE
    log_and_print("{} fold {} training complete.\n".format(datetime.now(), fold_num))


def test_fold(model, test_loader, save_path, n_classes, fold_num):
    days, hours, f1_scores, dice_scores = [], [], [], []
    for class_idx in range(n_classes):
        f1_scores.append([])
        dice_scores.append([])

    # --- set up device and model --- #
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device=device)

    # --- iterate through all test samples --- #
    log_and_print("{} starting fold {} testing...".format(datetime.now(), fold_num))
    model.eval()
    with torch.no_grad():
        for day_nums, hour_nums, samples, targets in tqdm(test_loader, desc="testing progress"):
            day_nums = day_nums.tolist()
            assert all(day_num == day_nums[0] for day_num in day_nums), "ERROR: day_nums are not the same for all samples in batch"
            days.append(day_nums[0])
            hour_nums = hour_nums.tolist()
            assert all(hour_num == hour_nums[0] for hour_num in hour_nums), "ERROR: hour_nums are not the same for all samples in batch"
            hours.append(hour_nums[0])

            samples = samples.to(device=device)
            targets = targets.to(device=device)
            outputs = model(samples)
            batch_f1_scores = f1(outputs, targets, num_classes=n_classes, average="none", zero_division=1.0).tolist()
            batch_dice_scores = dice(outputs, targets, num_classes=n_classes, average="none", zero_division=1.0).tolist()
            assert len(batch_f1_scores) == len(batch_dice_scores) == n_classes, "ERROR: f1_scores and dice_scores must have length n_classes"
            for class_idx in range(n_classes):
                f1_scores[class_idx].append(batch_f1_scores[class_idx])
                dice_scores[class_idx].append(batch_dice_scores[class_idx])

            del day_nums, hour_nums, samples, targets, outputs

    # --- print results --- #
    log_and_print("{} testing metrics:".format(datetime.now()))
    for class_idx in range(n_classes):
        log_and_print("\t[class_{}] f1_score: {:.9f}, dice_score: {:.9f}".format(
            class_idx, np.mean(f1_scores[class_idx]), np.mean(dice_scores[class_idx])))

    # --- save metrics --- #
    log_and_print("{} saving metrics...".format(datetime.now()))
    metrics_history = {
        "Fold": fold_num,
        "Day": days,
        "Hour": hours,
        # Perhaps track precision and recall too? For a PR curve?
    }
    for class_idx in range(n_classes):
        metrics_history[f"F1 Score [Class {class_idx}]"] = f1_scores[class_idx]
    for class_idx in range(n_classes):
        metrics_history[f"Dice Score [Class {class_idx}]"] = dice_scores[class_idx]
    
    save_metrics_CSV(metrics_history, save_path, n_classes) # NOT DONE
    create_metric_plots(metrics_history, save_path, n_classes) # NOT DONE
    log_and_print("{} fold {} testing complete.\n".format(datetime.now(), fold_num))