from util import add_feature_to_scaler, chronological_train_val_split, fit_target_scaler, inverse_scale_series, load_laser_array, scale_feature_arry, scale_target_series
from util import keep_2d_with_time, keep_1d, get_scaler
from typing import Any, Dict, Optional, Sequence, Tuple, Union
from typing import Any, Tuple
import numpy as np
from numpy.typing import ArrayLike
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

def make_one_step_windows(
    feature_array_scaled: ArrayLike,
    target_series_scaled: ArrayLike,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray]:
    
    X_arr = keep_2d_with_time(feature_array_scaled, name="feature_array_scaled")
    y_arr = keep_1d(target_series_scaled, name="target_series_scaled")
    
    if len(X_arr) != len(y_arr):
        raise ValueError(f"Length of feature array ({len(X_arr)}) and target series ({len(y_arr)}) must be the same.")
    
    if lookback < 1:
        raise ValueError(f"Lookback must be at least 1. Got {lookback}.")
    
    if len(X_arr) <= lookback:
        raise ValueError(f"Lookback ({lookback}) must be less than the number of samples ({len(X_arr)}).")
    
    X = []
    y = []
    
    for i in range(lookback, len(X_arr)):
        X.append(X_arr[i - lookback:i])
        y.append(y_arr[i])
        
    return np.array(X), np.array(y)

def reshape_for_keras_cov1d(X: ArrayLike) -> np.ndarray:
    x_reshaped = np.asarray(X, dtype=float)
    
    if x_reshaped.ndim != 3:
        raise ValueError(f"Input array must be 3D (samples, timesteps, features). Got {x_reshaped.shape}D.")
    
    return x_reshaped

def reshape_for_pytorch_conv1d(X: ArrayLike) -> np.ndarray:
    x_reshaped = np.asarray(X, dtype=float)
    
    if x_reshaped.ndim != 3:
        raise ValueError(f"Input array must be 3D (samples, timesteps, features). Got {x_reshaped.shape}D.")
    
    return x_reshaped.transpo

def prepare_train_val_data(
    file_path: Union[str, Path] = "Xtrain.mat",
    variable_name: Optional[str] = None,
    column: Optional[int] = None,
    target_column: int = 0,
    lookback: int = 30,
    val_fraction: float = 0.2,
    scaler_type: str = "standard",
    framework: str = "keras",
) -> Dict[str, Any]:
    
    data_real = load_laser_array(
        file_path=file_path,
        variable_name=variable_name,
        column=column,
    )

    if target_column < 0 or target_column >= data_real.shape[1]:
        raise IndexError(
            f"target_column must be between 0 and {data_real.shape[1] - 1}."
        )

    target_real = data_real[:, target_column]

    train_real, val_real = chronological_train_val_split(
        data_array=data_real,
        val_fraction=val_fraction,
    )

    train_target_real = train_real[:, target_column]
    val_target_real = val_real[:, target_column]

    feature_scaler = add_feature_to_scaler(
        train_array=train_real,
        scaler_type=scaler_type,
    )

    target_scaler = fit_target_scaler(
        train_target=train_target_real,
        scaler_type=scaler_type,
    )

    train_scaled = scale_feature_arry(train_real, feature_scaler)
    val_scaled = scale_feature_arry(val_real, feature_scaler)

    train_target_scaled = scale_target_series(train_target_real, target_scaler)
    val_target_scaled = scale_target_series(val_target_real, target_scaler)

    X_train_raw, y_train = make_one_step_windows(
        feature_array_scaled=train_scaled,
        target_series_scaled=train_target_scaled,
        lookback=lookback,
    )

    # Validation windows need the last training window as context.
    val_context_features_scaled = np.concatenate(
        [train_scaled[-lookback:], val_scaled],
        axis=0,
    )

    val_context_target_scaled = np.concatenate(
        [train_target_scaled[-lookback:], val_target_scaled],
        axis=0,
    )

    X_val_raw, y_val = make_one_step_windows(
        feature_array_scaled=val_context_features_scaled,
        target_series_scaled=val_context_target_scaled,
        lookback=lookback,
    )

    if framework == "keras":
        X_train = reshape_for_keras_cov1d(X_train_raw)
        X_val = reshape_for_keras_cov1d(X_val_raw)

    elif framework == "pytorch":
        X_train = reshape_for_pytorch_conv1d(X_train_raw)
        X_val = reshape_for_pytorch_conv1d(X_val_raw)

    elif framework == "raw":
        X_train = X_train_raw
        X_val = X_val_raw

    else:
        raise ValueError("framework must be one of: 'keras', 'pytorch', 'raw'.")

    return {
        "series_real": data_real,
        "target_real": target_real,
        "train_real": train_real,
        "val_real": val_real,
        "train_scaled": train_scaled,
        "val_scaled": val_scaled,
        "train_target_real": train_target_real,
        "val_target_real": val_target_real,
        "train_target_scaled": train_target_scaled,
        "val_target_scaled": val_target_scaled,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_train_raw": X_train_raw,
        "X_val_raw": X_val_raw,
        "y_train_real": inverse_scale_series(y_train, target_scaler),
        "y_val_real": inverse_scale_series(y_val, target_scaler),
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "scaler": target_scaler,
        "lookback": lookback,
        "target_column": target_column,
        "n_features": data_real.shape[1],
        "last_scaled_window": scale_feature_arry(data_real, feature_scaler)[-lookback:],
        "last_real_window": data_real[-lookback:],
    }



def make_torch_dataloaders(
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size=64,
    shuffle_train=True,
):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_laoder = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=False,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    
    return train_laoder, val_dataloader