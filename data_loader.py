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



#This was the onestep window for when we want to test if the model would work before hyperprametere tuning.
def make_one_step_windows(
    feature_array_scaled: ArrayLike,
    target_series_scaled: ArrayLike,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray]:
    
    # Convert to numpy arrays and check dimensions and add a onestep window 
    
    X_arr = keep_2d_with_time(feature_array_scaled, name="feature_array_scaled")
    y_arr = keep_1d(target_series_scaled, name="target_series_scaled")
    
    
    #Safety checks
    if len(X_arr) != len(y_arr):
        raise ValueError(f"Length of feature array ({len(X_arr)}) and target series ({len(y_arr)}) must be the same.")
    
    if lookback < 1:
        raise ValueError(f"Lookback must be at least 1. Got {lookback}.")
    
    if len(X_arr) <= lookback:
        raise ValueError(f"Lookback ({lookback}) must be less than the number of samples ({len(X_arr)}).")
    
    
    # Once satisfied with the input, create the one step windows for features and targets.
    X = []
    y = []
    
    for i in range(lookback, len(X_arr)):
        X.append(X_arr[i - lookback:i])
        y.append(y_arr[i])
        
    return np.array(X), np.array(y)



# Reshape for both keras and pytorch
#Implementation for Keras 
def reshape_for_keras_cov1d(X: ArrayLike) -> np.ndarray:
    x_reshaped = np.asarray(X, dtype=float)
    
    if x_reshaped.ndim != 3:
        raise ValueError(f"Input array must be 3D (samples, timesteps, features). Got {x_reshaped.shape}D.")
    
    return x_reshaped


#Implementation for Pytorch 
def reshape_for_pytorch_conv1d(X: ArrayLike) -> np.ndarray:
    
    #Reshape 
    x_reshaped = np.asarray(X, dtype=float)
    
    #warnign if the last two dimensions are the same, as this may indicate a potential issue with the input shape.
    if x_reshaped.ndim != 3:
        raise ValueError(f"Input array must be 3D (samples, timesteps, features). Got {x_reshaped.shape}D.")
    
    return x_reshaped.transpo
#So that multiple different model types can be used for testing. 



#Prepreocsseing the data from the dataloader
def prepare_train_val_data(
    #Declare path here as baseline
    file_path: Union[str, Path] = "Xtrain.mat",
    variable_name: Optional[str] = None,
    column: Optional[int] = None,
    target_column: int = 0,
    lookback: int = 30,
    val_fraction: float = 0.2,
    scaler_type: str = "standard",
    framework: str = "keras",
) -> Dict[str, Any]:
    
    
    #Get data from .mat file and do some checks on the data, then split into train and validation sets.
    data_real = load_laser_array(
        file_path=file_path,
        variable_name=variable_name,
        column=column,
    )

    
    #Satefy column check
    if target_column < 0 or target_column >= data_real.shape[1]:
        raise IndexError(
            f"target_column must be between 0 and {data_real.shape[1] - 1}."
        )

    target_real = data_real[:, target_column]



    #Train test split for time series data 
    train_real, val_real = chronological_train_val_split(
        data_array=data_real,
        val_fraction=val_fraction,
    )

    train_target_real = train_real[:, target_column]
    val_target_real = val_real[:, target_column]

    
    
    #Add feature to scaler and fit the target scaler on the training data, then scale the features and targets for both train and validation sets.
    feature_scaler = add_feature_to_scaler(
        train_array=train_real,
        scaler_type=scaler_type,
    )


    #Fit scaler to the y 
    target_scaler = fit_target_scaler(
        train_target=train_target_real,
        scaler_type=scaler_type,
    )
    
    
    #Scale the features and targets for both train and validation sets.
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

    
    #Depending on framework reshape the data.
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



#Dataloader for pytorch models 
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



#This functino is for when we want to test on the last 200 points of the orignal data set
def load_full_data_200(file_path="Xtrain.mat", variable_name=None, column=None):
    
    if "load_laser_array" in globals():
        arr = load_laser_array(
            file_path=file_path,
            variable_name=variable_name,
            column=column,
        )
    else:
        from scipy.io import loadmat

        raw = loadmat(file_path)

        numeric_vars = {
            key: value
            for key, value in raw.items()
            if not key.startswith("__") and isinstance(value, np.ndarray)
        }

        if variable_name is None:
            if len(numeric_vars) != 1:
                raise ValueError(
                    "Multiple numeric variables found. Set VARIABLE_NAME. "
                    f"Available variables: {list(numeric_vars.keys())}"
                )
            arr = next(iter(numeric_vars.values()))
        else:
            if variable_name not in numeric_vars:
                raise KeyError(
                    f"Variable {variable_name!r} not found. "
                    f"Available variables: {list(numeric_vars.keys())}"
                )
            arr = numeric_vars[variable_name]

    arr = np.asarray(arr, dtype=float)
    arr = np.squeeze(arr)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    #Make sure the column order is correct
    elif arr.ndim == 2:
        # MATLAB sometimes gives (features, time). Convert to (time, features).
        if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 10:
            arr = arr.T

    else:
        raise ValueError(f"Expected 1D or 2D data. Got shape {arr.shape}.")

    if column is not None:
        arr = arr[:, [int(column)]]

    if not np.all(np.isfinite(arr)):
        raise ValueError("Data contains NaN or infinite values.")

    return arr
    
    

#This function is for when we want to test on the last 200 points of the orignal data set, so we prepare the data accordingly.
def make_one_step_windows_200(feature_array_scaled, target_scaled, lookback):
    X = []
    y = []

    for i in range(lookback, len(feature_array_scaled)):
        X.append(feature_array_scaled[i - lookback:i, :])
        y.append(target_scaled[i])

    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


#THis function is for we want to test the last 200 steps with the same seature extraction
def prepare_recursive_train_data(
    full_data_real,
    lookback=30,
    recursive_steps=200,
    scaler_type="standard",
    framework="keras",
):
    full_data_real = np.asarray(full_data_real, dtype=float)

    if full_data_real.ndim == 1:
        full_data_real = full_data_real.reshape(-1, 1)

    if len(full_data_real) <= lookback + recursive_steps:
        raise ValueError(
            f"Data length {len(full_data_real)} too short for "
            f"lookback={lookback} and recursive_steps={recursive_steps}."
        )

    train_real = full_data_real[:-recursive_steps]
    heldout_real = full_data_real[-recursive_steps:, 0]

    feature_scaler = get_scaler(scaler_type)
    target_scaler = get_scaler(scaler_type)

    feature_scaler.fit(train_real)
    target_scaler.fit(train_real[:, [0]])

    train_scaled = feature_scaler.transform(train_real)
    target_scaled = target_scaler.transform(train_real[:, [0]]).ravel()

    X_train_raw, y_train = make_one_step_windows_200(
        feature_array_scaled=train_scaled,
        target_scaled=target_scaled,
        lookback=lookback,
    )

    if framework == "keras":
        X_train = X_train_raw

    elif framework == "pytorch":
        X_train = np.transpose(X_train_raw, (0, 2, 1))

    elif framework == "raw":
        X_train = X_train_raw

    else:
        raise ValueError("framework must be one of: keras, pytorch, raw")

    last_scaled_window = train_scaled[-lookback:]

    return {
        "train_real": train_real,
        "heldout_200_real": heldout_real,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "scaler": target_scaler,
        "train_scaled": train_scaled,
        "target_scaled": target_scaled,
        "X_train": X_train,
        "y_train": y_train,
        "last_scaled_window": last_scaled_window,
        "lookback": lookback,
        "recursive_steps": recursive_steps,
    }


    
#Dataloader for pytorch models
def make_train_loader(X_train, y_train, batch_size=64, shuffle=True):
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )
    return loader

def make_full_train_windows(feature_array_scaled, target_scaled, lookback):
    X = []
    y = []

    for i in range(lookback, len(feature_array_scaled)):
        X.append(feature_array_scaled[i - lookback:i, :])
        y.append(target_scaled[i])

    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def prepare_full_train_data_for_final_model(
    full_train_real,
    lookback,
    scaler_type="standard",
    framework="keras",
):
    full_train_real = np.asarray(full_train_real, dtype=float)

    if full_train_real.ndim == 1:
        full_train_real = full_train_real.reshape(-1, 1)

    if len(full_train_real) <= lookback:
        raise ValueError(
            f"Training data length {len(full_train_real)} is too short for lookback={lookback}."
        )

    feature_scaler = get_scaler(scaler_type)
    target_scaler = get_scaler(scaler_type)

    # Fit scalers on the full Xtrain only.
    feature_scaler.fit(full_train_real)
    target_scaler.fit(full_train_real[:, [0]])

    train_scaled = feature_scaler.transform(full_train_real)
    target_scaled = target_scaler.transform(full_train_real[:, [0]]).ravel()

    X_train_raw, y_train = make_full_train_windows(
        feature_array_scaled=train_scaled,
        target_scaled=target_scaled,
        lookback=lookback,
    )

    if framework == "keras":
        X_train = X_train_raw

    elif framework == "pytorch":
        X_train = np.transpose(X_train_raw, (0, 2, 1))

    elif framework == "raw":
        X_train = X_train_raw

    else:
        raise ValueError("framework must be one of: keras, pytorch, raw")

    last_scaled_window = train_scaled[-lookback:]

    return {
        "train_real": full_train_real,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "scaler": target_scaler,
        "train_scaled": train_scaled,
        "target_scaled": target_scaled,
        "X_train": X_train,
        "y_train": y_train,
        "last_scaled_window": last_scaled_window,
        "lookback": lookback,
    }


def prepare_xtest_recursive_eval_data(
    full_train_real,
    test_real,
    lookback,
    recursive_steps,
    scaler_type="standard",
):
    full_train_real = np.asarray(full_train_real, dtype=float)
    test_real = np.asarray(test_real, dtype=float)

    if full_train_real.ndim == 1:
        full_train_real = full_train_real.reshape(-1, 1)

    if test_real.ndim == 1:
        test_real = test_real.reshape(-1, 1)

    recursive_steps = min(recursive_steps, len(test_real))

    feature_scaler = get_scaler(scaler_type)
    target_scaler = get_scaler(scaler_type)

    # Fit only on Xtrain. Do not fit on Xtest.
    feature_scaler.fit(full_train_real)
    target_scaler.fit(full_train_real[:, [0]])

    train_scaled = feature_scaler.transform(full_train_real)
    target_scaled = target_scaler.transform(full_train_real[:, [0]]).ravel()

    last_scaled_window = train_scaled[-lookback:]

    return {
        "train_real": full_train_real,
        "heldout_200_real": test_real[:recursive_steps, 0].reshape(-1),
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "scaler": target_scaler,
        "train_scaled": train_scaled,
        "target_scaled": target_scaled,
        "last_scaled_window": last_scaled_window,
        "lookback": lookback,
        "recursive_steps": recursive_steps,
    }
