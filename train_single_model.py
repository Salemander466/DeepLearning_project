
import copy
import random
from data_loader import make_torch_dataloaders
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from Forward_Lateral_Causal_CNN import l1_l2_regularized_loss, set_seed

#This file contains the main training loop for training a single model. It includes functions for training one epoch, evaluating on the validation set, and the main function that orchestrates the training process with early stopping and learning rate scheduling. It also includes a prediction function to get predictions from the trained model.
def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    l1_lambda=1e-6,
    l2_lambda=1e-5,
    grad_clip=1.0,
):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)

        y_pred = model(X_batch)

        loss = l1_l2_regularized_loss(
            model=model,
            y_pred=y_pred,
            y_true=y_batch,
            l1_lambda=l1_lambda,
            l2_lambda=l2_lambda,
        )

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_size = X_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples

#Evaluation function for one epoch on the validation set. It computes the average MSE and MAE across all batches in the validation set.
def evaluate_one_epoch(
    model,
    val_loader,
    device,
):
    model.eval()

    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)

            mse = torch.mean((y_pred - y_batch) ** 2)
            mae = torch.mean(torch.abs(y_pred - y_batch))

            batch_size = X_batch.size(0)
            total_mse += mse.item() * batch_size
            total_mae += mae.item() * batch_size
            total_samples += batch_size

    average_mse = total_mse / total_samples
    average_mae = total_mae / total_samples

    return average_mse, average_mae


#Main training function that trains the model with early stopping and learning rate scheduling. It returns the best model, the training history, and information about the best epoch and validation performance.
def train_single_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    device=None,
    batch_size=64,
    epochs=200,
    learning_rate=1e-3,
    l1_lambda=1e-6,
    l2_lambda=1e-5,
    grad_clip=1.0,
    patience=25,
    min_delta=1e-6,
    checkpoint_path=None,
    seed=42,
):
    set_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    train_loader, val_loader = make_torch_dataloaders(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=batch_size,
        shuffle_train=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
        min_lr=1e-6,
    )

    best_val_mse = float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    history_rows = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            l1_lambda=l1_lambda,
            l2_lambda=l2_lambda,
            grad_clip=grad_clip,
        )

        val_mse, val_mae = evaluate_one_epoch(
            model=model,
            val_loader=val_loader,
            device=device,
        )

        scheduler.step(val_mse)

        current_lr = optimizer.param_groups[0]["lr"]

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss_regularized": train_loss,
                "val_mse": val_mse,
                "val_mae": val_mae,
                "learning_rate": current_lr,
            }
        )

        improved = val_mse < (best_val_mse - min_delta)

        if improved:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0 or improved:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss: {train_loss:.6f} | "
                f"val MSE: {val_mse:.6f} | "
                f"val MAE: {val_mae:.6f} | "
                f"lr: {current_lr:.2e}"
            )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}.")
            break

    model.load_state_dict(best_state)

    history = pd.DataFrame(history_rows)

    best_info = {
        "best_epoch": best_epoch,
        "best_val_mse_scaled": best_val_mse,
        "best_val_rmse_scaled": float(np.sqrt(best_val_mse)),
    }

    return model, history, best_info


#Prediction functions 

def predict_with_model(
    model,
    X,
    device=None,
    batch_size=256,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    predictions = []

    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            predictions.append(y_pred.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0).ravel()

    return predictions

#This is for hte 200 step recursive forecasting, which generates predictions one step at a time, feeding the previous predictions back into the model as input for the next prediction. This is done in the scaled space, and then we can inverse transform the predictions back to the original scale for evaluation.
def train_one_epoch_for_recursive_objective(
    model,
    train_loader,
    optimizer,
    device,
    l1_lambda=1e-6,
    l2_lambda=1e-5,
    grad_clip=1.0,
):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)

        y_pred = model(X_batch)

        loss = l1_l2_regularized_loss(
            model=model,
            y_pred=y_pred,
            y_true=y_batch,
            l1_lambda=l1_lambda,
            l2_lambda=l2_lambda,
        )

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_size = X_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples
