import os
from datetime import datetime

import torch
from torchmetrics.functional.classification import multiclass_f1_score as f1_score, multiclass_jaccard_index as jaccard_index
from tqdm import tqdm

from utils import *


def training_step(model, loss_fn, optimizer, train_loader, device, epoch):
    model.train()
    epoch_loss, epoch_f1, epoch_jac = 0.0, 0.0, 0.0
    for _, samples, targets in tqdm(train_loader, desc="epoch {} train progress".format(epoch)):
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
    return epoch_loss / len(train_loader), epoch_f1 / len(train_loader), epoch_jac / len(train_loader)


def validation_step(model, loss_fn, val_loader, device, epoch):
    model.eval()
    epoch_loss, epoch_f1, epoch_jac = 0.0, 0.0, 0.0
    with torch.no_grad():
        for _, samples, targets in tqdm(val_loader, desc="epoch {} val progress".format(epoch)):
            samples = samples.to(device=device)
            targets = targets.to(device=device)
            outputs = model(samples)
            epoch_loss += loss_fn(outputs, targets).item()
            epoch_f1 += f1_score(outputs, targets, num_classes=7).item()
            epoch_jac += jaccard_index(outputs, targets, num_classes=7).item()
            del samples, targets, outputs
    return epoch_loss / len(val_loader), epoch_f1 / len(val_loader), epoch_jac / len(val_loader)


def train(model, loss_fn, optimizer, train_loader, val_loader, n_epochs, device, save_path, patience=None):
    epochs_without_improvement = 0
    best_epoch = 0
    best_f1_score = 0.0

    losses_train, losses_val = [], []
    f1_train, f1_val = [], []
    jaccard_train, jaccard_val = [], []

    # --- iterate through all epochs --- #
    log_and_print("{} starting training...".format(datetime.now()))
    for epoch in range(n_epochs):

        # --- training step --- #
        epoch_loss, epoch_f1, epoch_jac = training_step(model, loss_fn, optimizer, train_loader, device, epoch + 1)
        losses_train.append(epoch_loss)
        f1_train.append(epoch_f1)
        jaccard_train.append(epoch_jac)

        # --- validation step --- #
        epoch_loss, epoch_f1, epoch_jac = validation_step(model, loss_fn, val_loader, device, epoch + 1)
        losses_val.append(epoch_loss)
        f1_val.append(epoch_f1)
        jaccard_val.append(epoch_jac)

        # --- print epoch results --- #
        log_and_print("{} epoch {}/{} metrics:".format(datetime.now(), epoch + 1, n_epochs))
        log_and_print("\t[train] loss: {:.9f}, f1_score: {:.9f}, jaccard_idx: {:.9f}".format(
            losses_train[epoch], f1_train[epoch], jaccard_train[epoch]))
        log_and_print("\t[valid] loss: {:.9f}, f1_score: {:.9f}, jaccard_idx: {:.9f}".format(
            losses_val[epoch], f1_val[epoch], jaccard_val[epoch]))
        
        # --- check for early stopping --- #
        if patience is not None:
            if f1_val[epoch] >= best_f1_score:
                best_epoch = epoch + 1
                best_f1_score = f1_val[epoch]
                torch.save(model.state_dict(), os.path.join(save_path, "best_weights.pth"))
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement == patience:
                log_and_print("{} early stopping at epoch {}".format(datetime.now(), epoch + 1))
                break

    # --- save weights --- #
    log_and_print("{} training complete.".format(datetime.now()))
    if patience is not None:
        log_and_print("best f1 score: {:.9f}, occurred on epoch {}".format(best_f1_score, best_epoch))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, "final_weights.pth"))

    # --- save metrics --- #
    metrics_history = [
        ("loss", losses_train, losses_val),
        ("f1_score", f1_train, f1_val),
        ("jaccard_index", jaccard_train, jaccard_val),
    ]
    save_metrics_CSV(metrics_history, save_path)
    log_and_print("{} script finished.".format(datetime.now()))

