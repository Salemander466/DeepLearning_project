
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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