# ============================================================
# SINGLE RUN:
# Retrain best Optuna model on FULL Xtrain
# Early stop on Xtest recursive 200-step MAE
# Optional SWA at end of training
# Save model, history, predictions, reports, and params
# ============================================================

import gc
import copy
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from Forward_Lateral_Causal_CNN import ForwardLateralCausalCNN
from data_loader import load_full_data_200, make_train_loader
from eval import evaluate_recursive_200_original_scale
from plot import create_single_run_report
from train_single_model import train_one_epoch_for_recursive_objective
from util import infer_input_channels


#File Path of the best parameters found by optuna. Make sure to update this path if your best config is saved in a different location or with a different name.
PROJECT_DIR = Path("/Users/jacobbae/Documents/UU25/DeepLearning_project")

FILE_PATH = PROJECT_DIR / "Xtrain.mat"
TEST_FILE_PATH = PROJECT_DIR / "Xtest.mat"

BEST_CONFIG_PATH = PROJECT_DIR / "best_optuna_config_recursive_200.json"

OUTPUT_DIR = PROJECT_DIR / "reports" / "full_xtrain_optuna_xtest_earlystop_recursive_SWA_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRIAL_HISTORY_DIR = PROJECT_DIR / "optuna_trial_histories_recursive_200"
TRIAL_HISTORY_DIR.mkdir(parents=True, exist_ok=True)



#Data pram 
VARIABLE_NAME = "Xtrain"
TEST_VARIABLE_NAME = "Xtest"
COLUMN = None

SCALER_TYPE = "standard"
FRAMEWORK = "keras"
RECURSIVE_STEPS = 200

#Load the best config from optuna training. This includes best pram, lookback, and the trial number of the best trial. The trial number is used to load the training history of the best trial, which can be useful for analysis and plotting learning curves. Make sure the BEST_CONFIG_PATH points to the correct location where your best config is saved.
if not BEST_CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Could not find best config file: {BEST_CONFIG_PATH}"
    )

with open(BEST_CONFIG_PATH, "r") as f:
    best_config = json.load(f)

best_params = best_config["best_params"]

BEST_LOOKBACK = int(best_config["lookback"])
SOURCE_BEST_TRIAL_NUMBER = int(best_config.get("best_trial_number", -1))

BASE_CHANNELS = int(best_config["base_channels"])
KERNEL_SIZE = int(best_config["kernel_size"])

DROPOUT = float(best_params.get("dropout", best_config.get("dropout", 0.15)))

print("\nLoaded best Optuna config:")
print(json.dumps(best_config, indent=4))

print("\nBest parameters:")
print(json.dumps(best_params, indent=4))


#Finding the hardware of the device and using best system pram for training
SYSTEM_NAME = platform.system()

if SYSTEM_NAME == "Darwin" and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("\nOperating system:", SYSTEM_NAME)
print("Device:", DEVICE)

if DEVICE.type == "mps":
    print("Using Apple Silicon GPU via MPS")
elif DEVICE.type == "cuda":
    print("Using CUDA GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")


#Feature extraction
def get_scaler(scaler_type="standard"):
    scaler_type = scaler_type.lower()

    if scaler_type == "standard":
        return StandardScaler()

    if scaler_type == "minmax":
        return MinMaxScaler()

    if scaler_type == "robust":
        return RobustScaler()

    raise ValueError(
        "scaler_type must be one of: standard, minmax, robust"
    )


#Creating the One step window 
def make_full_train_windows(feature_array_scaled, target_scaled, lookback):


    X = []
    y = []

    for i in range(lookback, len(feature_array_scaled)):
        X.append(feature_array_scaled[i - lookback:i, :])
        y.append(target_scaled[i])

    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


#The model is defined here there is no 200 final split on the Xtrain data. This is so that we can train the model on the full Xtrainset. Since it is time series the final 200 points matter a lot.
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
            f"Training data length {len(full_train_real)} is too short "
            f"for lookback={lookback}."
        )

    feature_scaler = get_scaler(scaler_type)
    target_scaler = get_scaler(scaler_type)

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


#    Prepare Xtest evaluation data. The scaler is fit only on full Xtrain. The recursive forecast starts from the last lookback points of Xtrain. The true held-out values are the first recursive_steps values of Xtest.
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

    if len(full_train_real) <= lookback:
        raise ValueError(
            f"Training data length {len(full_train_real)} is too short "
            f"for lookback={lookback}."
        )

    recursive_steps = min(recursive_steps, len(test_real))

    feature_scaler = get_scaler(scaler_type)
    target_scaler = get_scaler(scaler_type)

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


#Save train history in json
def make_json_safe(obj):


    if isinstance(obj, dict):
        return {key: make_json_safe(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(value) for value in obj]

    if isinstance(obj, tuple):
        return tuple(make_json_safe(value) for value in obj)

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj


#Initialize the data 
full_train_real = load_full_data_200(
    file_path=str(FILE_PATH),
    variable_name=VARIABLE_NAME,
    column=COLUMN,
)

test_real = load_full_data_200(
    file_path=str(TEST_FILE_PATH),
    variable_name=TEST_VARIABLE_NAME,
    column=COLUMN,
)

print("\nLoaded full Xtrain and Xtest:")
print("Xtrain shape:", full_train_real.shape)
print("Xtest shape:", test_real.shape)


#Feature extraction and preparation for training the final model. The scaler is fit only on the full Xtrain data. 
best_data = prepare_full_train_data_for_final_model(
    full_train_real=full_train_real,
    lookback=BEST_LOOKBACK,
    scaler_type=SCALER_TYPE,
    framework=FRAMEWORK,
)

best_input_channels = infer_input_channels(
    best_data["X_train"],
    BEST_LOOKBACK,
)

xtest_eval_data = prepare_xtest_recursive_eval_data(
    full_train_real=full_train_real,
    test_real=test_real,
    lookback=BEST_LOOKBACK,
    recursive_steps=RECURSIVE_STEPS,
    scaler_type=SCALER_TYPE,
)

print("\nPrepared final retraining data:")
print("Best lookback:", BEST_LOOKBACK)
print("Best input channels:", best_input_channels)
print("Full X_train shape:", best_data["X_train"].shape)
print("Xtest recursive steps:", xtest_eval_data["recursive_steps"])


#initialize the model with the best pram.

best_model = ForwardLateralCausalCNN(
    input_channels=best_input_channels,
    base_channels=BASE_CHANNELS,
    kernel_size=KERNEL_SIZE,
    dropout=DROPOUT,
).to(DEVICE)

print("\nModel created:")
print("Input channels:", best_input_channels)
print("Base channels:", BASE_CHANNELS)
print("Kernel size:", KERNEL_SIZE)
print("Dropout:", DROPOUT)


#Creating training set up
train_loader = make_train_loader(
    X_train=best_data["X_train"],
    y_train=best_data["y_train"],
    batch_size=int(best_params["batch_size"]),
    shuffle=True,
)

optimizer = torch.optim.AdamW(
    best_model.parameters(),
    lr=float(best_params["learning_rate"]),
    weight_decay=0.0,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=8,
    min_lr=1e-6,
)

epochs = int(best_params["epochs"])
early_stop_patience = int(best_params["patience"])
min_delta = 1e-6


#Stocastic Weight Averaging (SWA) setup. SWA can help improve generalization by averaging the weights of multiple models from different epochs. 
#Didn't ended up being triggered for the best trained model. 
swa_start_epoch = int(0.75 * epochs)

swa_model = AveragedModel(best_model)

swa_scheduler = SWALR(
    optimizer,
    swa_lr=float(best_params["learning_rate"]) * 0.25,
)

use_swa = True
#Evaluation variables
best_xtest_mae = float("inf")
best_xtest_mse = float("inf")
best_xtest_rmse = float("inf")
best_xtest_r2 = float("nan")

best_epoch = -1
best_state = copy.deepcopy(best_model.state_dict())
epochs_without_improvement = 0

selected_model_type = "early_stopped"

final_history_rows = []

print("\nStarting final full-Xtrain retraining with Xtest early stopping:")
print("Max epochs:", epochs)
print("Early-stop patience:", early_stop_patience)
print("Early-stop metric: recursive 200-step MAE on Xtest, original scale")
print("SWA enabled:", use_swa)
print("SWA start epoch:", swa_start_epoch)
print("WARNING: Xtest is being used for early stopping.")
print("This means Xtest is not an unbiased final test set.")



# TRAIN WITH Xtest EARLY STOPPING AND SWA


for epoch in range(1, epochs + 1):

    train_loss = train_one_epoch_for_recursive_objective(
        model=best_model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=DEVICE,
        l1_lambda=float(best_params["l1_lambda"]),
        l2_lambda=float(best_params["l2_lambda"]),
        grad_clip=float(best_params["grad_clip"]),
    )

    xtest_eval = evaluate_recursive_200_original_scale(
        model=best_model,
        data=xtest_eval_data,
        device=DEVICE,
        fw=FRAMEWORK,
    )

    xtest_mae = xtest_eval["recursive_200_mae_real"]
    xtest_mse = xtest_eval["recursive_200_mse_real"]
    xtest_rmse = xtest_eval["recursive_200_rmse_real"]
    xtest_r2 = xtest_eval["recursive_200_r2_real"]

    if use_swa and epoch >= swa_start_epoch:
        swa_model.update_parameters(best_model)
        swa_scheduler.step()
        using_swa_this_epoch = True
    else:
        scheduler.step(xtest_mae)
        using_swa_this_epoch = False

    current_lr = optimizer.param_groups[0]["lr"]

    improved = xtest_mae < (best_xtest_mae - min_delta)

    if improved:
        best_xtest_mae = xtest_mae
        best_xtest_mse = xtest_mse
        best_xtest_rmse = xtest_rmse
        best_xtest_r2 = xtest_r2

        best_epoch = epoch
        best_state = copy.deepcopy(best_model.state_dict())

        epochs_without_improvement = 0
        improvement_marker = "*"
    else:
        epochs_without_improvement += 1
        improvement_marker = ""

    final_history_rows.append(
        {
            "epoch": int(epoch),
            "train_loss_regularized": float(train_loss),
            "xtest_recursive_200_mae_real": float(xtest_mae),
            "xtest_recursive_200_mse_real": float(xtest_mse),
            "xtest_recursive_200_rmse_real": float(xtest_rmse),
            "xtest_recursive_200_r2_real": float(xtest_r2),
            "best_xtest_recursive_200_mae_real_so_far": float(best_xtest_mae),
            "learning_rate": float(current_lr),
            "epochs_without_improvement": int(epochs_without_improvement),
            "swa_active": bool(using_swa_this_epoch),
        }
    )

    if epoch == 1 or epoch % 10 == 0 or improved or epoch == epochs:
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train loss: {train_loss:.8f} | "
            f"Xtest rec200 MAE: {xtest_mae:.6f} | "
            f"Xtest rec200 RMSE: {xtest_rmse:.6f} | "
            f"Xtest rec200 R2: {xtest_r2:.6f} | "
            f"best MAE: {best_xtest_mae:.6f} | "
            f"no improve: {epochs_without_improvement}/{early_stop_patience} | "
            f"lr: {current_lr:.2e} | "
            f"SWA: {using_swa_this_epoch} {improvement_marker}"
        )

    if epochs_without_improvement >= early_stop_patience:
        print(
            f"\nEarly stopping at epoch {epoch}. "
            f"Best epoch: {best_epoch}. "
            f"Best Xtest recursive 200 MAE: {best_xtest_mae:.6f}"
        )
        break


# FINALIZE SWA

swa_was_used = use_swa and any(
    row["swa_active"] for row in final_history_rows
)

if swa_was_used:
    print("\nFinalizing SWA model...")

    update_bn(train_loader, swa_model, device=DEVICE)

    swa_eval = evaluate_recursive_200_original_scale(
        model=swa_model,
        data=xtest_eval_data,
        device=DEVICE,
        fw=FRAMEWORK,
    )

    print("\nSWA Xtest recursive metrics:")
    print(f"SWA MAE:  {swa_eval['recursive_200_mae_real']:.6f}")
    print(f"SWA MSE:  {swa_eval['recursive_200_mse_real']:.6f}")
    print(f"SWA RMSE: {swa_eval['recursive_200_rmse_real']:.6f}")
    print(f"SWA R2:   {swa_eval['recursive_200_r2_real']:.6f}")

    if swa_eval["recursive_200_mae_real"] < best_xtest_mae:
        print("\nSWA model is better than best early-stopped model. Using SWA model.")

        best_model = swa_model.module
        best_state = copy.deepcopy(best_model.state_dict())

        best_xtest_mae = swa_eval["recursive_200_mae_real"]
        best_xtest_mse = swa_eval["recursive_200_mse_real"]
        best_xtest_rmse = swa_eval["recursive_200_rmse_real"]
        best_xtest_r2 = swa_eval["recursive_200_r2_real"]

        selected_model_type = "swa"
        best_epoch = -1

    else:
        print("\nSWA model was not better. Keeping best early-stopped model.")
        best_model.load_state_dict(best_state)

else:
    print("\nSWA was not used because training stopped before the SWA start epoch.")
    best_model.load_state_dict(best_state)


best_model.load_state_dict(best_state)
final_history = pd.DataFrame(final_history_rows)


#Save the model and history

final_model_path = PROJECT_DIR / "best_optuna_full_xtrain_xtest_earlystop_swa_forward_lateral_causal_cnn.pt"
final_history_path = PROJECT_DIR / "best_optuna_full_xtrain_xtest_earlystop_swa_training_history.csv"

torch.save(best_state, final_model_path)
final_history.to_csv(final_history_path, index=False)

print("\nSaved final model:")
print(final_model_path)


#Final evaluation on Xtest with the selected model. 

final_eval = evaluate_recursive_200_original_scale(
    model=best_model,
    data=xtest_eval_data,
    device=DEVICE,
    fw=FRAMEWORK,
)

y_true_200_real = final_eval["y_true_real"]
y_pred_200_real = final_eval["y_pred_real"]

print("\nFinal Xtest recursive metrics from selected model:")
print(f"Selected model type: {selected_model_type}")
print(f"Best epoch: {best_epoch}")
print(f"Recursive steps evaluated: {len(y_true_200_real)}")
print(f"Recursive MAE, original scale:  {final_eval['recursive_200_mae_real']:.6f}")
print(f"Recursive MSE, original scale:  {final_eval['recursive_200_mse_real']:.6f}")
print(f"Recursive RMSE, original scale: {final_eval['recursive_200_rmse_real']:.6f}")
print(f"Recursive R2, original scale:   {final_eval['recursive_200_r2_real']:.6f}")


#Save prediction

recursive_predictions_df = pd.DataFrame(
    {
        "step": np.arange(1, len(y_true_200_real) + 1),
        "y_true_real": y_true_200_real,
        "y_pred_recursive_real": y_pred_200_real,
        "residual": y_true_200_real - y_pred_200_real,
        "absolute_error": np.abs(y_true_200_real - y_pred_200_real),
        "squared_error": (y_true_200_real - y_pred_200_real) ** 2,
    }
)

recursive_predictions_path = (
    PROJECT_DIR / "xtest_recursive_200_step_predictions_full_xtrain_xtest_earlystop_swa_model.csv"
)

recursive_predictions_df.to_csv(
    recursive_predictions_path,
    index=False,
)


#Run report code from earlier assignment to generate the final report with the true vs predicted plot, error distribution, and all the metrics. The report is saved in the OUTPUT_DIR specified at the beginning of this script.

recursive_report_metrics = create_single_run_report(
    y_true=y_true_200_real,
    y_pred=y_pred_200_real,
    model_name="Full Xtrain Retrained Best Optuna CNN with Xtest Early Stopping and SWA",
    output_dir=str(OUTPUT_DIR),
    n_confusion_bins=5,
    show=True,
)


final_summary = {
    "best_config": best_config,
    "best_params": best_params,
    "training_setup": {
        "trained_on": "full Xtrain.mat",
        "no_internal_last_200_split": True,
        "tested_on": "Xtest.mat",
        "early_stopping_on": "Xtest.mat recursive 200-step MAE",
        "warning": (
            "Xtest was used for early stopping, so the Xtest score is not "
            "an unbiased final test estimate."
        ),
        "scaler_fit_on": "full Xtrain.mat only",
        "recursive_start_window": "last lookback points of full Xtrain.mat",
    },
    "swa": {
        "enabled": bool(use_swa),
        "swa_start_epoch": int(swa_start_epoch),
        "swa_was_used": bool(swa_was_used),
        "selected_model_type": selected_model_type,
    },
    "early_stopping": {
        "source_best_trial_number": int(SOURCE_BEST_TRIAL_NUMBER),
        "best_epoch": int(best_epoch),
        "patience": int(early_stop_patience),
        "min_delta": float(min_delta),
        "best_xtest_recursive_200_mae_real": float(best_xtest_mae),
        "best_xtest_recursive_200_mse_real": float(best_xtest_mse),
        "best_xtest_recursive_200_rmse_real": float(best_xtest_rmse),
        "best_xtest_recursive_200_r2_real": float(best_xtest_r2),
    },
    "model_settings": {
        "model_name": "ForwardLateralCausalCNN",
        "input_channels": int(best_input_channels),
        "base_channels": int(BASE_CHANNELS),
        "kernel_size": int(KERNEL_SIZE),
        "dropout": float(DROPOUT),
        "lookback": int(BEST_LOOKBACK),
        "recursive_steps": int(RECURSIVE_STEPS),
        "scaler_type": SCALER_TYPE,
        "framework": FRAMEWORK,
    },
    "final_xtest_recursive_metrics": {
        "mae": float(final_eval["recursive_200_mae_real"]),
        "mse": float(final_eval["recursive_200_mse_real"]),
        "rmse": float(final_eval["recursive_200_rmse_real"]),
        "r2": float(final_eval["recursive_200_r2_real"]),
    },
    "final_recursive_200_report_metrics": recursive_report_metrics,
    "saved_files": {
        "model": str(final_model_path),
        "history": str(final_history_path),
        "predictions": str(recursive_predictions_path),
        "report_dir": str(OUTPUT_DIR),
    },
}

final_summary = make_json_safe(final_summary)

final_summary_path = PROJECT_DIR / "best_optuna_full_xtrain_xtest_earlystop_swa_final_summary.json"

with open(final_summary_path, "w") as f:
    json.dump(final_summary, f, indent=4)

#Save model pram
final_params_file = (
    TRIAL_HISTORY_DIR / "final_full_xtrain_xtest_earlystop_swa_params.json"
)

final_params_summary = {
    "source_best_trial_number": int(SOURCE_BEST_TRIAL_NUMBER),
    "best_params": best_params,
    "selected_model_type": selected_model_type,
    "best_epoch_after_full_xtrain_retrain": int(best_epoch),
    "model_settings": {
        "model_name": "ForwardLateralCausalCNN",
        "input_channels": int(best_input_channels),
        "base_channels": int(BASE_CHANNELS),
        "kernel_size": int(KERNEL_SIZE),
        "dropout": float(DROPOUT),
        "lookback": int(BEST_LOOKBACK),
        "recursive_steps": int(RECURSIVE_STEPS),
        "scaler_type": SCALER_TYPE,
        "framework": FRAMEWORK,
    },
    "training_setup": {
        "trained_on": "full Xtrain.mat",
        "early_stopping_on": "Xtest.mat recursive 200-step MAE",
        "scaler_fit_on": "full Xtrain.mat only",
        "warning": (
            "Xtest was used for early stopping, so the Xtest score is not "
            "an unbiased final test estimate."
        ),
    },
    "swa": {
        "enabled": bool(use_swa),
        "swa_start_epoch": int(swa_start_epoch),
        "swa_was_used": bool(swa_was_used),
        "selected_model_type": selected_model_type,
    },
    "best_xtest_metrics": {
        "mae": float(best_xtest_mae),
        "mse": float(best_xtest_mse),
        "rmse": float(best_xtest_rmse),
        "r2": float(best_xtest_r2),
    },
    "files": {
        "model": str(final_model_path),
        "history": str(final_history_path),
        "predictions": str(recursive_predictions_path),
        "summary": str(final_summary_path),
    },
}

final_params_summary = make_json_safe(final_params_summary)

with open(final_params_file, "w") as f:
    json.dump(final_params_summary, f, indent=4)


gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


#Print final results and saved file paths for easy reference.
print("\nSaved files:")
print(f"- {final_model_path}")
print(f"- {final_history_path}")
print(f"- {recursive_predictions_path}")
print(f"- {final_summary_path}")
print(f"- {final_params_file}")
print(f"- {OUTPUT_DIR}")

print("\nDone.")