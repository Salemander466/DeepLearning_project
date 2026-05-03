
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

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

from util import keep_1d



def align_true_pred(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]: 
    
    yt = keep_1d(y_true, name="y_true")
    yp = keep_1d(y_pred, name="y_pred")
    
    if len(yt) != len(yp):
        raise ValueError(f"Length of y_true ({len(yt)}) and y_pred ({len(yp)}) must be the same.")
    
    return yt, yp


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
    
def metrics_to_dataframe(metrics: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([metrics])

