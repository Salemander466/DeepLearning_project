
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
    
    if column is not Nones:
        if column < 0 or column >= arr.shape[1]:
            raise ValueError(f"Column index {column} is out of bounds for array with shape {arr.shape}")
        arr = arr[:, column]
        
    return arr
        
        




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


def get_scaler(scalar_type: str = "standard") -> Any:
    scalar_type = scalar_type.lower()
    if scalar_type == "standard":
        return StandardScaler()
    elif scalar_type == "minmax":
        return MinMaxScaler()
    elif scalar_type == "robust":
        return RobustScaler()
    
    raise ValueError(f"Unsupported scaler type: {scalar_type}. Supported types are 'standard', 'minmax', and 'robust'.")


def add_feature_to_scaler(train_arry: ArrayLike, scalar_type: str = "standard") -> Any:
    x_train = keep_2d_with_time(train_arry, name="train_array")
    scaler = get_scaler(scalar_type)
    scaler.fit(x_train)
    return scaler

def scale_feature_arry(data_array:ArrayLike, scaler: Any,) -> np.ndarray:
    x = keep_2d_with_time(data_array, name="data_array")
    return scaler.transform(x)


def fit_target_scaler(
    train_target: ArrayLike,
    scaler_type: str = "standard",
) -> Any:
    y = keep_1d(train_target, name="train_target").reshape(-1, 1)
    
    scaler = get_scaler(scaler_type)
    scaler.fit(y)
    return scaler

def scale_target_series(
    target_series: ArrayLike,
    scaler: Any,
) -> np.ndarray:
    y = keep_1d(target_series, name="target_series").reshape(-1, 1)
    return scaler.transform(y).ravel()

def inverse_scale_series(
    scaled_series: ArrayLike,
    scaler: Any,
) -> np.ndarray:
    y_scaled = keep_1d(scaled_series, name="scaled_series").reshape(-1, 1)
    return scaler.inverse_transform(y_scaled).ravel()




   











