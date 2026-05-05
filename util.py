#This file contains all the utility functions for loading and preprocessing the data, as well as some helper functions for scaling and inverse scaling the data. It also includes a function to inspect the contents of the .mat file and get a summary of the variables it contains. These functions are used throughout the project to handle the data and prepare it for training and evaluation.
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
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



#This function is used keep the .mat file within the proper format, and to make sure that the data is in the right shape and format for the rest of the code. It also checks for any non-finite values in the data, which could cause issues during training and evaluation.
def keep_1d (x: ArrayLike, name: str= "array") -> np.ndarray:
    
    #To make sure that there is one column in the dataframe
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"{name} must have exactly one column")
        return x.iloc[:, 0].to_numpy()

    elif isinstance(x,pd.Series):
        return x.to_numpy()
    
    else:
        arr = np.asarray(x)

    arr = np.squeeze(arr)
    
    #Keeping consistancy across
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1 dimensional after squeezing, but got shape {arr.shape}")    
    #Ensuring right datatype
    arr = arr.astype(float)
    
    #Making sure there is nothing that can cause errors
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non finite values (NaN or Inf)")
    
    return arr
    
    
#This functions is used to keep the data in the right shape for the model, which expects 2D arrays with the time dimension as one of the dimensions. It also checks for non-finite values and ensures that the data is in a numeric format.
def keep_2d_with_time(x: ArrayLike, name: str= "array") -> np.ndarray:
    
    arr = np.asarray(x, dtype=float )
    arr = np.squeeze(arr)
    
    if arr.ndim ==1:
        arr = arr.reshape(-1, 1)
        
    elif arr.ndim == 2:
        if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 10:
            arr = arr.T
        
    else:
        raise ValueError(f"{name} must be 1 or 2 dimensional after squeezing, but got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non finite values (NaN or Inf)")
                         
    return arr


#This funciton loads the .mat file and extracts all the variables in it. 
def load_mat_file(
    file_path: Union[str, Path],
    variable_name: Optional[str] = None,
    squeeze: bool = True,
) -> Dict[str, np.ndarray]:
    # Placeholder for file loading logic
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if path.suffix.lower() != ".mat":
        raise ValueError(f"Unsupported file type: {path.suffix}. Only .mat files are supported.")
    
    try:
        
        raw = loadmat(file_path)
        data = {
            key: value
            for key, value in raw.items()
            if not key.startswith("__") and isinstance(value, np.ndarray)
        }
    except NotImplementedError as e:
       import h5py
       
       data= {}
       with h5py.File(file_path, 'r') as f:
            keys = [variable_name] if variable_name is not None else list(f.keys())

            for key in keys:
                if key not in f:
                    raise KeyError(f"Variable {key!r} not found in {path}.")
                data[key] = np.array(f[key]).T

               
    if variable_name is not None:
        if variable_name not in data:
            raise KeyError(f"Variable {variable_name!r} not found in {path}.")
        return {variable_name: data[variable_name]}
    
        data = {variable_name: data[variable_name]}
        
    cleaned = {}
    
    for key,value in data.items():
        arr = np.asarray(value)
        
        if squeeze:
            arr = np.squeeze(arr)
            
        if np.issubdtype(arr.dtype, np.number):
            cleaned[key] = arr.astype(float)
            
    if not cleaned:
        raise ValueError(f"No valid numeric data found in {path}.")
    return cleaned



#This lets us inpsect the contents of the .mat file and get a summary of the variables it contains, including their shapes, data types, and basic statistics. 
def inspect_mat_file(file_path: Union[str, Path]) -> pd.DataFrame:
 

    data = load_mat_file(
        file_path=file_path,
        variable_name=None,
        squeeze=False,
    )

    rows = []

    for name, arr in data.items():
        arr = np.asarray(arr, dtype=float)

        rows.append(
            {
                "variable": name,
                "shape": tuple(arr.shape),
                "dtype": str(arr.dtype),
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "mean": float(np.nanmean(arr)),
                "std": float(np.nanstd(arr)),
            }
        )

    return pd.DataFrame(rows)


#This function is used to infer the number of input channels for the model based on the shape of the input data and the lookback window. It checks if the time dimension matches the lookback window and returns the other dimension as the number of input channels. This allows the model to be flexible with different input shapes, as long as one of the dimensions matches the lookback window.
def load_laser_array(
    file_path: Union[str, Path] = "Xtrain.mat",
    variable_name: Optional[str] = None,
    column: Optional[int] = None,
) -> np.ndarray:
    data = load_mat_file(file_path=file_path, variable_name=variable_name, squeeze=True)

    if variable_name is not None:
        if len(data) != 1:
            raise ValueError(f"Expected exactly one variable when variable_name is specified, but found {len(data)}")
        arr = next(iter(data.values()))
    else:
        arr = data[variable_name]
        
    arr = keep_2d_with_time(arr, name="laser_array")
    
    if column is not None:
        if column < 0 or column >= arr.shape[1]:
            raise ValueError(f"Column index {column} is out of bounds for array with shape {arr.shape}")
        arr = arr[:, column]
        
    return arr
        
        



#This make sure the that the timesetps are consistent across the different convolutional blocks, and that the modle can handle the input data correctly, even if it is in a different format (e.g. Keras-style vs PyTorch-style). It also ensures that the model can be trained and evaluated without errors due to shape mismatches or non-finite values in the data.
def chronological_train_val_split(
    data_array: ArrayLike, val_fraction: float = 0.2)-> Tuple[np.ndarray, np.ndarray]:
    x = keep_2d_with_time(data_array, name="data_array")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    
    split_idk = int(len(x) * (1 - val_fraction))
    
    if split_idk <= 1:
        raise ValueError("Not enough data points to split with the given val_fraction")
    
    if split_idk >= len(x):
        raise ValueError("val_fraction is too small, resulting in no validation data")
    
    train_array = x[:split_idk]
    val_fraction = x[split_idk:]
    return train_array, val_fraction


#get_scaler() chooses which scaler to use. 
#Standard: subtracts mean and divides by standard deviation
#Minmax:   rescales values into a fixed range, usually 0 to 1
#Robust:   scales using median and interquartile range, better if there are outliers
def get_scaler(scalar_type: str = "standard") -> Any:
    scalar_type = scalar_type.lower()
    if scalar_type == "standard":
        return StandardScaler()
    elif scalar_type == "minmax":
        return MinMaxScaler()
    elif scalar_type == "robust":
        return RobustScaler()
    
    raise ValueError(f"Unsupported scaler type: {scalar_type}. Supported types are 'standard', 'minmax', and 'robust'.")



#add_feature_to_scaler() fits a scaler on the training features. 
#1. Makes sure the training data is 2D.
#2. Creates the scaler.
#3. Fits the scaler using only the training data. 
#the scaler should be fitted only on training data, not validation or test data, because fitting on future/test data would cause data leakage.
def add_feature_to_scaler(train_arry: ArrayLike, scalar_type: str = "standard") -> Any:
    x_train = keep_2d_with_time(train_arry, name="train_array")
    scaler = get_scaler(scalar_type)
    scaler.fit(x_train)
    return scaler

#applies the fitted scaler to feature data. It does not fit a new scaler. It only transforms the data using a scaler that was already fitted. 
def scale_feature_arry(data_array:ArrayLike, scaler: Any,) -> np.ndarray:
    x = keep_2d_with_time(data_array, name="data_array")
    return scaler.transform(x)

#fits a scaler on the target values. It reshapes the target from: (samples,) to (samples, 1), because sklearn scalers expect 2D input. This is separate from the feature scaler because the target is the value the model predicts.
def fit_target_scaler(
    train_target: ArrayLike,
    scaler_type: str = "standard",
) -> Any:
    y = keep_1d(train_target, name="train_target").reshape(-1, 1)
    
    scaler = get_scaler(scaler_type)
    scaler.fit(y)
    return scaler


#It converts the target into the scaled space used during training. The model learns to predict scaled values, not raw 2-255 values.
def scale_target_series(
    target_series: ArrayLike,
    scaler: Any,
) -> np.ndarray:
    y = keep_1d(target_series, name="target_series").reshape(-1, 1)
    return scaler.transform(y).ravel()


#This is very important for your assignment because your final MAE/RMSE should be computed on the original laser scale, not the scaled values.
def inverse_scale_series(
    scaled_series: ArrayLike,
    scaler: Any,
) -> np.ndarray:
    y_scaled = keep_1d(scaled_series, name="scaled_series").reshape(-1, 1)
    return scaler.inverse_transform(y_scaled).ravel()



#figures out how many input channels/features the model has.
#Model can accept two possible input formats:
#Keras-style:   (samples, lookback, features)
#PyTorch-style: (samples, features, lookback)
#Scaling functions:
#Prepare the data so the model trains better.
#Inverse scaling:
#Converts predictions back to real values.
#infer_input_channels:
#Tells the model how many features/channels are in each time step.
def infer_input_channels(X, lookback):
    
    if X.ndim != 3:
        raise ValueError(f"Expected 3D input. Got shape {X.shape}.")

    if X.shape[1] == lookback:
        return X.shape[2]

    if X.shape[2] == lookback:
        return X.shape[1]

    raise ValueError(
        f"Cannot infer input channels from shape {X.shape} with lookback={lookback}."
    )
   











