
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union
from xml.parsers.expat import model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    max_error,
    r2_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from torch import device
import torch

from util import keep_1d


#Safety funciton to make sure the true and predicted values are aligned and have the same length, and are 1d arrays.
def align_true_pred(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]: 
    
    yt = keep_1d(y_true, name="y_true")
    yp = keep_1d(y_pred, name="y_pred")
    
    if len(yt) != len(yp):
        raise ValueError(f"Length of y_true ({len(yt)}) and y_pred ({len(yp)}) must be the same.")
    
    return yt, yp



#Compute evaluation metrics without 200 step recursive forecasting, just on the test set. (This is for the one step forecasting evaluation)
def compute_forecast_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> Dict[str, float]:
    
    yt, yp = align_true_pred(y_true, y_pred)
    
    residuals = yt - yp
    mse = float(mean_squared_error(yt, yp))

    return {
        "n_samples": int(len(yt)),
        "mae": float(mean_absolute_error(yt, yp)),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2_score(yt, yp)) if len(yt) > 1 else float("nan"),
        "median_absolute_error": float(median_absolute_error(yt, yp)),
        "max_error": float(max_error(yt, yp)),
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0,
        "residual_min": float(np.min(residuals)),
        "residual_max": float(np.max(residuals)),
    }
    
    
#Convert into dataframe for better visualization and comparison across models.
def metrics_to_dataframe(metrics: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([metrics])


#function that generates the 200-step recursive forecast, but it does it in the scaled/normalized space, not the original 2-255 laser scale
def recursive_forecast_scaled(
    model,
    initial_window_scaled,
    n_steps=200,
    framework="keras",
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    window = np.asarray(initial_window_scaled, dtype=np.float32).copy()

    if window.ndim == 1:
        window = window.reshape(-1, 1)

    preds_scaled = []

    with torch.no_grad():
        for _ in range(n_steps):

            if framework in ["keras", "raw"]:
                X_input_np = window[np.newaxis, :, :]

            elif framework == "pytorch":
                X_input_np = np.transpose(window[np.newaxis, :, :], (0, 2, 1))

            else:
                raise ValueError("framework must be one of: keras, pytorch, raw")

            X_input = torch.tensor(
                X_input_np,
                dtype=torch.float32,
                device=device,
            )

            next_pred_scaled = model(X_input).detach().cpu().numpy().reshape(-1)[0]

            preds_scaled.append(next_pred_scaled)

            next_row = window[-1, :].copy()
            next_row[0] = next_pred_scaled

            window = np.vstack([window[1:], next_row])

    return np.asarray(preds_scaled, dtype=float)


#Proper evaluation function for the 200 step recursive forecasting, which first generates the 200 step forecast in the scaled space, then inverse transforms it back to the original scale, and then computes the metrics on the real scale. This is the main evaluation function for the recursive forecasting performance on the last 200 points of the original dataset.
def evaluate_recursive_200_original_scale(
    model,
    data,
    device=None,
    fw = "keras",
):
    y_pred_scaled = recursive_forecast_scaled(
            model=model,
            initial_window_scaled=data["last_scaled_window"],
            n_steps=data["recursive_steps"],
            framework=fw,
            device=device,
        )

    y_pred_real = data["target_scaler"].inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).ravel()

    y_true_real = np.asarray(data["heldout_200_real"], dtype=float).reshape(-1)

    mae = mean_absolute_error(y_true_real, y_pred_real)
    mse = mean_squared_error(y_true_real, y_pred_real)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_real, y_pred_real)

    return {
        "recursive_200_mae_real": float(mae),
        "recursive_200_mse_real": float(mse),
        "recursive_200_rmse_real": float(rmse),
        "recursive_200_r2_real": float(r2),
        "y_true_real": y_true_real,
        "y_pred_real": y_pred_real,
    }
    
    




