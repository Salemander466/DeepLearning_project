
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np
from numpy.typing import ArrayLike
from util import keep_2d_with_time, keep_1d, get_scaler
from eval import align_true_pred, compute_forecast_metrics, metrics_to_dataframe
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


def make_output_dir(output_dir: Union[str, Path]) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


#This funciton makes the input shape consistent for the model. It checks if the input is in the correct format and permutes it if necessary. This allows the model to accept both PyTorch and Keras style inputs without errors.
def save_or_show(
    fig: plt.Figure,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Optional[Path]:
    saved_path = None
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        saved_path = save_path
        
    if show:
        plt.show()
        
    plt.close(fig)
    
    return saved_path


#This function plots all the analysis that we will need for the report. It includes the real vs predicted values, the residuals over time, the error distribution, the predicted vs actual scatter plot, and a binned regression confusion graph. This will help us visualize the performance of our model in different ways and identify any patterns or issues in the predictions.
#Shows the actual signal and the model prediction together.
def plot_real_vs_predicted(    y_true: ArrayLike,
    y_pred: ArrayLike,
    title: str = "Real vs Predicted Values",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Optional[Path]:
    
    yt ,yp = align_true_pred(y_true, y_pred)
    x = np.arange(len(yt))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, yt, label="True", marker="o")
    ax.plot(x, yp, label="Predicted", marker="x")
    ax.set_title(title)
    ax.set_xlabel("time step")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return save_or_show(fig, save_path=save_path, show=show)


#plot_residuals() shows the error over time. This can help us identify if there are any patterns in the errors, such as increasing error over time or specific time steps where the model struggles.
def plot_residuals(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    title: str = "Residuals Over Time",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Optional[Path]:
    yt, yp = align_true_pred(y_true, y_pred)
    residuals = yt - yp
    x = np.arange(len(residuals))
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, residuals, label="Residuals", marker="o")
    ax.axhline(0, color="red", linestyle="--", label="Zero Error")
    ax.set_title(title)
    ax.set_xlabel("time step")
    ax.set_ylabel("Residual")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return save_or_show(fig, save_path=save_path, show=show)


#Plot error into a histogram
def plot_error_histogram(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    title: str = "Error Distribution",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Optional[Path]:
    yt, yp = align_true_pred(y_true, y_pred)
    errors = yt - yp
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(errors, bins=20, alpha=0.7, color="blue", edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel("Error")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    
    return save_or_show(fig, save_path=save_path, show=show)


#This function plots the predicted values against the actual values in a scatter plot. This can help us see how well the predictions align with the true values and identify any systematic biases or patterns in the predictions.
def plot_predicted_vs_actual_scatter(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    title: str = "Predicted vs Actual Scatter",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Optional[Path]:
    yt, yp = align_true_pred(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yt, yp, alpha=0.7, color="green", edgecolor="black")
    ax.plot([min(yt), max(yt)], [min(yt), max(yt)], color="red", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Actual Value")
    ax.set_ylabel("Predicted Value")
    ax.grid(True, alpha=0.3)
    
    return save_or_show(fig, save_path=save_path, show=show)



#These three functions make and Plot the Confusion matrix for regression. This is a way to visualize how well the model is doing in different ranges of the target variable. It bins the true and predicted values into categories and then counts how many predictions fall into each category compared to the true categories. This can help us see if the model is systematically overestimating or underestimating in certain ranges.
def make_regression_bins(
    values: ArrayLike,
    n_bins: int = 5,
    strategy: str = "quantile",
) -> np.ndarray: 
    values = keep_1d(values, name="values")
    
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2. Got {n_bins}.")
    
    if strategy == "quantile":
        edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
        
    elif strategy == "uniform":
        edges = np.linspace(np.min(values), np.max(values), n_bins + 1)
        
    else:
        raise ValueError(f"Invalid strategy '{strategy}'. Must be 'quantile' or 'uniform'.")    
    
    edges = np.unique(edges)
    
    if len(edges) <= 2:
        raise ValueError(f"Not enough unique bin edges ({len(edges)}) to create bins. Try reducing n_bins or using a different strategy.")
    
    edges[0] = -np.inf
    edges[-1] = np.inf
    
    return edges

def regression_confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    n_bins: int = 5,
    strategy: str = "quantile",
) -> np.ndarray:
    yt, yp = align_true_pred(y_true, y_pred)
    
    all_values = np.concatenate([yt, yp])
    bin_edges = make_regression_bins(all_values, n_bins=n_bins, strategy=strategy)
    
    y_true_binned = np.digitize(yt, bin_edges[1:-1], right=False)
    y_pred_binned = np.digitize(yp, bin_edges[1:-1], right=False)
    labels = np.arange(len(bin_edges) - 1)
    
    cm = confusion_matrix(y_true_binned, y_pred_binned, labels=labels)
    return cm , bin_edges

def plot_regression_confusion_graph(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    n_bins: int = 5,
    strategy: str = "quantile",
    title: str = "Binned Regression Confusion Graph",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Optional[Path]:
    
    cm, edges = regression_confusion_matrix(y_true, y_pred, n_bins=n_bins, strategy=strategy)
    
    labels = []
    for i in range(len(edges) - 1):
        left = edges[i]
        right = edges[i + 1]

        if np.isneginf(left):
            labels.append(f"<= {right:.3g}")
        elif np.isposinf(right):
            labels.append(f"> {left:.3g}")
        else:
            labels.append(f"{left:.3g} to {right:.3g}")
            
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, aspect="auto")
    fig.colorbar(image, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted bin")
    ax.set_ylabel("Actual bin")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                fontsize=8,
            )

    return save_or_show(fig, save_path=save_path, show=show)



# This is the final report after running the hyperprameter tuning and training the final model. It then generates all the plots and metrics.
def create_single_run_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    model_name: str = "model",
    output_dir: Union[str, Path] = "reports/single_run",
    n_confusion_bins: int = 5,
    show: bool = False,
) -> Dict[str, Any]:

    output_dir = make_output_dir(output_dir)
    safe_name = model_name.replace(" ", "_").replace("/", "_")

    metrics = compute_forecast_metrics(y_true, y_pred)
    metrics["model_name"] = model_name

    metrics_df = metrics_to_dataframe(metrics)
    metrics_df.to_csv(output_dir / f"{safe_name}_metrics.csv", index=False)

    with open(output_dir / f"{safe_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_real_vs_predicted(
        y_true=y_true,
        y_pred=y_pred,
        title=f"{model_name}: Real vs Predicted Values",
        save_path=output_dir / f"{safe_name}_real_vs_predicted.png",
        show=show,
    )

    plot_residuals(
        y_true=y_true,
        y_pred=y_pred,
        title=f"{model_name}: Residuals Over Time",
        save_path=output_dir / f"{safe_name}_residuals.png",
        show=show,
    )

    plot_error_histogram(
        y_true=y_true,
        y_pred=y_pred,
        title=f"{model_name}: Prediction Error Distribution",
        save_path=output_dir / f"{safe_name}_error_histogram.png",
        show=show,
    )

    plot_predicted_vs_actual_scatter(
        y_true=y_true,
        y_pred=y_pred,
        title=f"{model_name}: Predicted vs Actual",
        save_path=output_dir / f"{safe_name}_predicted_vs_actual.png",
        show=show,
    )

    plot_regression_confusion_graph(
        y_true=y_true,
        y_pred=y_pred,
        n_bins=n_confusion_bins,
        strategy="quantile",
        title=f"{model_name}: Binned Regression Confusion Graph",
        save_path=output_dir / f"{safe_name}_regression_confusion_graph.png",
        show=show,
    )

    return metrics