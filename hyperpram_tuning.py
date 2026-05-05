import os
import gc
import copy
import json
import random
import subprocess
import sys
from datetime import datetime
from IPython.display import display
import numpy as np
import optuna
import pandas as pd
import platform

import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Forward_Lateral_Causal_CNN import ForwardLateralCausalCNN, set_seed, l1_l2_regularized_loss
from data_loader import load_full_data_200, make_train_loader, prepare_recursive_train_data
from eval import evaluate_recursive_200_original_scale
from plot import create_single_run_report
from train_single_model import train_one_epoch_for_recursive_objective
from util import infer_input_channels


#Required Checks for 
required_objects = [
    "ForwardLateralCausalCNN",
    "l1_l2_regularized_loss",
    "create_single_run_report",
]

for object_name in required_objects:
    if object_name not in globals():
        raise NameError(
            f"{object_name} is not defined. Run the model and reporting cells first."
        )
#settings

SEED = 42

FILE_PATH = "Xtrain.mat"
VARIABLE_NAME = "Xtrain"
COLUMN = None

LOOKBACK_CANDIDATES = [ 90, 100, 110, 120, 130]
RECURSIVE_STEPS = 200

SCALER_TYPE = "standard"
FRAMEWORK = "keras"

N_TRIALS = 25
STUDY_DIRECTION = "minimize"

BASE_CHANNELS = 64
KERNEL_SIZE = 3
DROPOUT = 0.15

TRIAL_PRINT_EVERY = 1


#BASED OF OS and device gpu avaliablity, our team had both Mac and Windows so we want to make sure it worked for both.
SYSTEM_NAME = platform.system()

if SYSTEM_NAME == "Darwin":
    OS_NAME = "macOS"
elif SYSTEM_NAME == "Windows":
    OS_NAME = "Windows"
elif SYSTEM_NAME == "Linux":
    OS_NAME = "Linux"
else:
    OS_NAME = SYSTEM_NAME

if OS_NAME == "macOS" and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Operating system:", OS_NAME)
print("Device:", DEVICE)

if DEVICE.type == "mps":
    print("Using Apple Silicon GPU via MPS")
elif DEVICE.type == "cuda":
    print("Using CUDA GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

full_data_real = load_full_data_200(
    file_path=FILE_PATH,
    variable_name=VARIABLE_NAME,
    column=COLUMN,
)

print("Full data shape:", full_data_real.shape)


#Hyperprameter tuning with 200 of the last points as the point of optimization. This function uses early stop to make sure the we don't waste training time. It also uses Optuna for hyperprameter tuning
def train_model_for_recursive_trial(
    trial,
    model,
    data,
    device,
    batch_size,
    epochs,
    learning_rate,
    l1_lambda,
    l2_lambda,
    dropout,
    grad_clip,
    patience,
    min_delta=1e-6,
    print_every=1,
):

    model = model.to(device)


    #Load dataa
    train_loader = make_train_loader(
        X_train=data["X_train"],
        y_train=data["y_train"],
        batch_size=batch_size,
        shuffle=True,
    )

    #load Optimizer and scheduler
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


    # Initialize tracking variables for best metrics and early stopping
    best_recursive_mae = float("inf")
    best_recursive_mse = float("inf")
    best_recursive_rmse = float("inf")
    best_recursive_r2 = float("nan")

    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    history_rows = []

    
    #print to for testing
    print("\n" + "=" * 100)
    print(f"Starting Trial {trial.number}")
    print(f"lookback p       = {data['lookback']}")
    print(f"batch_size       = {batch_size}")
    print(f"epochs           = {epochs}")
    print(f"learning_rate    = {learning_rate:.3e}")
    print(f"l1_lambda        = {l1_lambda:.3e}")
    print(f"l2_lambda        = {l2_lambda:.3e}")
    print(f"dropout          = {dropout:.3e}")
    print(f"grad_clip        = {grad_clip}")
    print(f"patience         = {patience}")
    print("Training loss    = one-step MSE plus L1/L2")
    print("Early stopping   = recursive 200-step MAE, original scale")
    print("Optuna objective = recursive 200-step MAE, original scale")
    print("=" * 100)

    for epoch in range(1, epochs + 1):

        train_loss = train_one_epoch_for_recursive_objective(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            l1_lambda=l1_lambda,
            l2_lambda=l2_lambda,
            grad_clip=grad_clip,
        )

        recursive_metrics = evaluate_recursive_200_original_scale(
            model=model,
            data=data,
            device=device,
        )

        
        #Using 200-step recursive MAE on the original scale as the main metric for optimization, but also tracking MSE, RMSE, and R2 for more comprehensive evaluation and analysis of the model's performance on the recursive forecasting task.
        recursive_mae = recursive_metrics["recursive_200_mae_real"]
        recursive_mse = recursive_metrics["recursive_200_mse_real"]
        recursive_rmse = recursive_metrics["recursive_200_rmse_real"]
        recursive_r2 = recursive_metrics["recursive_200_r2_real"]

        scheduler.step(recursive_mae)

        current_lr = optimizer.param_groups[0]["lr"]
        
        #Print the current trial's progress and results at each epoch, including the training loss, recursive 200-step MAE, MSE, RMSE, R2 on the original scale, best recursive 200-step MAE so far, number of epochs without improvement, and current learning rate. This provides real-time feedback on how the trial is progressing and allows us to monitor the optimization process closely.
        trial.report(recursive_mae, step=epoch)

        if trial.should_prune():
            history = pd.DataFrame(history_rows)
            history.to_csv(
                f"optuna_trial_histories_recursive_200/trial_{trial.number}_history_pruned.csv",
                index=False,
            )

            print(
                f"Trial {trial.number} pruned at epoch {epoch} | "
                f"recursive 200 MAE real: {recursive_mae:.6f}"
            )

            raise optuna.TrialPruned()

        improved = recursive_mae < (best_recursive_mae - min_delta)

        #Make sure that the model is keep improving on the recursive 200 step and counts how many tries of nonimprovement
        if improved:
            best_recursive_mae = recursive_mae
            best_recursive_mse = recursive_mse
            best_recursive_rmse = recursive_rmse
            best_recursive_r2 = recursive_r2

            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            improvement_marker = "*"
        else:
            epochs_without_improvement += 1
            improvement_marker = ""

        history_rows.append(
            {
                "trial": trial.number,
                "lookback": data["lookback"],
                "epoch": epoch,
                "train_loss_regularized": train_loss,

                "recursive_200_mae_real": recursive_mae,
                "recursive_200_mse_real": recursive_mse,
                "recursive_200_rmse_real": recursive_rmse,
                "recursive_200_r2_real": recursive_r2,

                "best_recursive_200_mae_real_so_far": best_recursive_mae,
                "learning_rate": current_lr,
                "epochs_without_improvement": epochs_without_improvement,
            }
        )

        if epoch == 1 or epoch % print_every == 0 or improved or epoch == epochs:
            print(
                f"Trial {trial.number:03d} | "
                f"p={data['lookback']} | "
                f"Epoch {epoch:03d}/{epochs} | "
                f"train loss: {train_loss:.8f} | "
                f"rec200 MAE real: {recursive_mae:.6f} | "
                f"rec200 MSE real: {recursive_mse:.6f} | "
                f"rec200 RMSE real: {recursive_rmse:.6f} | "
                f"best rec200 MAE: {best_recursive_mae:.6f} | "
                f"no improve: {epochs_without_improvement}/{patience} | "
                f"lr: {current_lr:.2e} {improvement_marker}"
            )

        #After each epoch this displays if there was no improvement
        if epochs_without_improvement >= patience:
            print(
                f"Trial {trial.number} early stopped at epoch {epoch}. "
                f"Best epoch: {best_epoch}. "
                f"Best recursive 200 MAE real: {best_recursive_mae:.6f}"
            )
            break

    history = pd.DataFrame(history_rows)

    history.to_csv(
        f"optuna_trial_histories_recursive_200/trial_{trial.number}_history.csv",
        index=False,
    )

    model.load_state_dict(best_state)

    best_metrics = {
        "best_epoch": int(best_epoch),
        "best_recursive_200_mae_real": float(best_recursive_mae),
        "best_recursive_200_mse_real": float(best_recursive_mse),
        "best_recursive_200_rmse_real": float(best_recursive_rmse),
        "best_recursive_200_r2_real": float(best_recursive_r2),
    }

    return model, best_state, best_metrics, history


#output Folder

os.makedirs("optuna_trial_histories_recursive_200", exist_ok=True)
os.makedirs("reports/optuna_recursive_200_forward_lateral_causal_cnn", exist_ok=True)



#Past trial hyperparameters and results for reference:
# def objective(trial):
#     """
#     Optuna minimizes:
#         200-step recursive MAE on original scale.

#     Tuned parameters:
#         lookback
#         batch_size
#         epochs
#         learning_rate
#         l1_lambda
#         l2_lambda
#         grad_clip
#         patience
#     """

#     set_seed(SEED + trial.number)

#     lookback = trial.suggest_categorical(
#         "lookback",
#         [100, 125, 150,175],
#     )

#     batch_size = trial.suggest_categorical(
#         "batch_size",
#         [32],
#     )

#     epochs = trial.suggest_int(
#         "epochs",
#         170,
#         240,
#         step=10,
#     )

#     learning_rate = trial.suggest_float(
#         "learning_rate",
#         2.5e-4,
#         8.5e-4,
#         log=True,
#     )

#     l1_lambda = trial.suggest_float(
#         "l1_lambda",
#         4e-7,
#         7e-6,
#         log=True,
#     )

#     l2_lambda = trial.suggest_float(
#         "l2_lambda",
#         6e-6,
#         2.5e-5,
#         log=True,
#     )

#     grad_clip = trial.suggest_categorical(
#         "grad_clip",
#         [0.10, 0.15, 0.20, 0.25],
#     )

#     patience = trial.suggest_categorical(
#         "patience",
#         [35, 40, 45, 50],
#     )
#     trial_data = prepare_recursive_train_data(
#         full_data_real=full_data_real,
#         lookback=lookback,
#         recursive_steps=RECURSIVE_STEPS,
#         scaler_type=SCALER_TYPE,
#         framework=FRAMEWORK,
#     )

#     input_channels = infer_input_channels(
#         trial_data["X_train"],
#         lookback,
#     )

#     trial.set_user_attr("lookback", int(lookback))
#     trial.set_user_attr("input_channels", int(input_channels))
#     trial.set_user_attr("n_train_windows", int(len(trial_data["X_train"])))

#     trial_model = ForwardLateralCausalCNN(
#         input_channels=input_channels,
#         base_channels=BASE_CHANNELS,
#         kernel_size=KERNEL_SIZE,
#         dropout=DROPOUT,
#     ).to(DEVICE)

#     try:
#         trial_model, best_state, best_metrics, history = train_model_for_recursive_trial(
#             trial=trial,
#             model=trial_model,
#             data=trial_data,
#             device=DEVICE,
#             batch_size=batch_size,
#             epochs=epochs,
#             learning_rate=learning_rate,
#             l1_lambda=l1_lambda,
#             l2_lambda=l2_lambda,
#             grad_clip=grad_clip,
#             patience=patience,
#             min_delta=1e-6,
#             print_every=TRIAL_PRINT_EVERY,
#         )

#     except RuntimeError as error:
#         if "out of memory" in str(error).lower():
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#             print(f"Trial {trial.number} pruned because CUDA ran out of memory.")
#             raise optuna.TrialPruned("CUDA out of memory.")
#         raise error

#     trial.set_user_attr("best_epoch", best_metrics["best_epoch"])
#     trial.set_user_attr("recursive_200_mae_real", best_metrics["best_recursive_200_mae_real"])
#     trial.set_user_attr("recursive_200_mse_real", best_metrics["best_recursive_200_mse_real"])
#     trial.set_user_attr("recursive_200_rmse_real", best_metrics["best_recursive_200_rmse_real"])
#     trial.set_user_attr("recursive_200_r2_real", best_metrics["best_recursive_200_r2_real"])
#     trial.set_user_attr(
#         "history_file",
#         f"optuna_trial_histories_recursive_200/trial_{trial.number}_history.csv",
#     )

#     objective_value = best_metrics["best_recursive_200_mae_real"]

#     del trial_model
#     del trial_data

#     gc.collect()

#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

#     return objective_value


#Hyperpramerter to tune for the ForwardLateralCausalCNN model on the recursive 200-step forecasting task. After the objective function above that is commented was run. With more optimized ranged of hyperparmeter 
def objective(trial):
    """
    Optuna minimizes:
        200-step recursive MAE on original scale.

    Tighter search around the best observed trials.
    """

    set_seed(SEED + trial.number)

    # Best region was 125, with 150 also competitive.
    lookback = trial.suggest_categorical(
        "lookback",
        [115, 125, 135, 145, 150],
    )

    # All good trials used 32.
    batch_size = trial.suggest_categorical(
        "batch_size",
        [32],
    )

    # Best trial was 180 epochs; strong 150-region trials used 220-240.
    epochs = trial.suggest_categorical(
        "epochs",
        [170, 180, 190, 200, 220, 230, 240],
    )

    # Tight around best trial and strong 150-region trials.
    learning_rate = trial.suggest_float(
        "learning_rate",
        2.7e-4,
        8.2e-4,
        log=True,
    )

    # Best trial had very low L1; 150-region liked mid L1.
    l1_lambda = trial.suggest_float(
        "l1_lambda",
        4.0e-7,
        3.2e-6,
        log=True,
    )

    # Most top trials are clustered near 1.9e-5 to 2.5e-5.
    l2_lambda = trial.suggest_float(
        "l2_lambda",
        1.7e-5,
        2.6e-5,
        log=True,
    )
    
    dropout = trial.suggest_float(
        "dropout",
        0.10,
        0.35,
        step=0.05,
    )

    # Best trials mostly used 0.25.
    grad_clip = trial.suggest_categorical(
        "grad_clip",
        [0.20, 0.25, 0.30],
    )

    # Best trial used 35; strong 150-region used 45.
    patience = trial.suggest_categorical(
        "patience",
        [35, 40, 45],
    )

    trial_data = prepare_recursive_train_data(
        full_data_real=full_data_real,
        lookback=lookback,
        recursive_steps=RECURSIVE_STEPS,
        scaler_type=SCALER_TYPE,
        framework=FRAMEWORK,
    )

    input_channels = infer_input_channels(
        trial_data["X_train"],
        lookback,
    )

    trial.set_user_attr("lookback", int(lookback))
    trial.set_user_attr("dropout", float(dropout))
    trial.set_user_attr("input_channels", int(input_channels))
    trial.set_user_attr("n_train_windows", int(len(trial_data["X_train"])))

    trial_model = ForwardLateralCausalCNN(
        input_channels=input_channels,
        base_channels=BASE_CHANNELS,
        kernel_size=KERNEL_SIZE,
        dropout=dropout,
    ).to(DEVICE)

    try:
        trial_model, best_state, best_metrics, history = train_model_for_recursive_trial(
            trial=trial,
            model=trial_model,
            data=trial_data,
            device=DEVICE,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            l1_lambda=l1_lambda,
            l2_lambda=l2_lambda,
            dropout=dropout,
            grad_clip=grad_clip,
            patience=patience,
            min_delta=1e-6,
            print_every=TRIAL_PRINT_EVERY,
        )

    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Trial {trial.number} pruned because CUDA ran out of memory.")
            raise optuna.TrialPruned("CUDA out of memory.")
        raise error

    trial.set_user_attr("best_epoch", best_metrics["best_epoch"])
    trial.set_user_attr("recursive_200_mae_real", best_metrics["best_recursive_200_mae_real"])
    trial.set_user_attr("recursive_200_mse_real", best_metrics["best_recursive_200_mse_real"])
    trial.set_user_attr("recursive_200_rmse_real", best_metrics["best_recursive_200_rmse_real"])
    trial.set_user_attr("recursive_200_r2_real", best_metrics["best_recursive_200_r2_real"])
    
    trial.set_user_attr(
        "history_file",
        f"optuna_trial_histories_recursive_200/trial_{trial.number}_history.csv",
    )

    # Save exact best state for this trial.
    trial_model_path = f"optuna_trial_histories_recursive_200/trial_{trial.number}_best_model.pt"
    torch.save(best_state, trial_model_path)
    trial.set_user_attr("best_model_path", trial_model_path)

    objective_value = best_metrics["best_recursive_200_mae_real"]

    del trial_model
    del trial_data

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return objective_value





#This prints the training process to make sure everything is working. 
def print_trial_callback(study, trial):
    print("\n" + "-" * 100)
    print(f"Completed trial: {trial.number}")
    print(f"State: {trial.state}")

    if trial.value is not None:
        print(
            f"Trial objective score: {trial.value:.6f} "
            "[recursive 200-step MAE, original scale]"
        )

    print(
        f"Best objective so far: {study.best_value:.6f} "
        "[recursive 200-step MAE, original scale]"
    )

    print(f"Best trial so far: {study.best_trial.number}")

    print("Current trial parameters and metrics:")
    print("lookback:", trial.user_attrs.get("lookback", None))
    print("input_channels:", trial.user_attrs.get("input_channels", None))
    print("n_train_windows:", trial.user_attrs.get("n_train_windows", None))
    print("best_epoch:", trial.user_attrs.get("best_epoch", None))
    print("recursive_200_mae_real:", trial.user_attrs.get("recursive_200_mae_real", None))
    print("recursive_200_mse_real:", trial.user_attrs.get("recursive_200_mse_real", None))
    print("recursive_200_rmse_real:", trial.user_attrs.get("recursive_200_rmse_real", None))
    print("recursive_200_r2_real:", trial.user_attrs.get("recursive_200_r2_real", None))

    print("-" * 100)


#THis funciton runs the optuna study with the defined objective and prints the results. It also saves the trials and the best config for later analysis and retraining the final model with the best parameters.

sampler = optuna.samplers.TPESampler(seed=SEED)

pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,
    n_warmup_steps=10,
    interval_steps=5,
)

study = optuna.create_study(
    direction=STUDY_DIRECTION,
    sampler=sampler,
    pruner=pruner,
    study_name="forward_lateral_causal_cnn_recursive_200_tuning",
)

study.optimize(
    objective,
    n_trials=N_TRIALS,
    callbacks=[print_trial_callback],
    show_progress_bar=True,
)


#This saves the best trials 

trials_df = study.trials_dataframe()
trials_df.to_csv(
    "optuna_recursive_200_forward_lateral_causal_cnn_trials.csv",
    index=False,
)

print("\nBest trial:")
print("Trial number:", study.best_trial.number)
print("Best recursive 200-step MAE real scale:", study.best_value)
print("Best parameters:")
print(study.best_params)

print("\nTop trials:")
display(trials_df.sort_values("value", ascending=True).head(10))


#Saves the best parameter into a json file for later analysis and to tighten the range of hyperparmeter tuning

best_params = study.best_params
BEST_LOOKBACK = int(best_params["lookback"])

best_config = {
    "model_name": "ForwardLateralCausalCNN",
    "study_name": "forward_lateral_causal_cnn_recursive_200_tuning",
    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

    "file_path": FILE_PATH,
    "variable_name": VARIABLE_NAME,
    "column": COLUMN,

    "lookback": BEST_LOOKBACK,
    "lookback_candidates": LOOKBACK_CANDIDATES,
    "recursive_steps": RECURSIVE_STEPS,
    "scaler_type": SCALER_TYPE,
    "framework": FRAMEWORK,

    "input_channels": int(study.best_trial.user_attrs.get("input_channels", 1)),
    "base_channels": BASE_CHANNELS,
    "kernel_size": KERNEL_SIZE,
    "dropout": float(best_params["dropout"]),

    "n_trials": N_TRIALS,
    "objective": "recursive_200_mae_real_scale",
    "early_stopping_metric": "recursive_200_mae_real_scale",

    "best_trial_number": study.best_trial.number,
    "best_recursive_200_mae_real": float(study.best_value),
    "best_recursive_200_mse_real": study.best_trial.user_attrs.get("recursive_200_mse_real", None),
    "best_recursive_200_rmse_real": study.best_trial.user_attrs.get("recursive_200_rmse_real", None),
    "best_recursive_200_r2_real": study.best_trial.user_attrs.get("recursive_200_r2_real", None),

    "best_params": best_params,
    "best_epoch_from_trial": study.best_trial.user_attrs.get("best_epoch", None),

    "note": (
        "This Optuna study optimizes the assignment metric directly: "
        "200-step recursive MAE after inverse-transforming predictions "
        "to the original scale. The lookback window p is also tuned."
    ),
}

with open("best_optuna_config_recursive_200.json", "w") as f:
    json.dump(best_config, f, indent=4)

print("\nSaved config to: best_optuna_config_recursive_200.json")


#This retrains the model with the best hyperparmeters from the tuning so that ican be saved. 

best_data = prepare_recursive_train_data(
    full_data_real=full_data_real,
    lookback=BEST_LOOKBACK,
    recursive_steps=RECURSIVE_STEPS,
    scaler_type=SCALER_TYPE,
    framework=FRAMEWORK,
)

best_input_channels = infer_input_channels(
    best_data["X_train"],
    BEST_LOOKBACK,
)

print("\nRetraining final model using best recursive-200 parameters:")
print(json.dumps(best_params, indent=4))
print("Best lookback:", BEST_LOOKBACK)
print("Best input channels:", best_input_channels)
print("Best X_train shape:", best_data["X_train"].shape)

best_model = ForwardLateralCausalCNN(
    input_channels=best_input_channels,
    base_channels=int(best_config["base_channels"]),
    kernel_size=int(best_config["kernel_size"]),
    dropout=float(best_params["dropout"]),
).to(DEVICE)

best_model, best_state, best_metrics, best_history = train_model_for_recursive_trial(
    trial=study.best_trial,
    model=best_model,
    data=best_data,
    device=DEVICE,
    batch_size=int(best_params["batch_size"]),
    epochs=int(best_params["epochs"]),
    learning_rate=float(best_params["learning_rate"]),
    l1_lambda=float(best_params["l1_lambda"]),
    l2_lambda=float(best_params["l2_lambda"]),
    grad_clip=float(best_params["grad_clip"]),
    patience=int(best_params["patience"]),
    min_delta=1e-6,
    print_every=TRIAL_PRINT_EVERY,
)

torch.save(best_state, "best_optuna_recursive_200_forward_lateral_causal_cnn.pt")
best_history.to_csv("best_optuna_recursive_200_training_history.csv", index=False)

print("\nFinal retrained best recursive 200 metrics:")
print(best_metrics)


#This evaluates the model that is about to be saved. To see if it kept consistency. 

final_eval = evaluate_recursive_200_original_scale(
    model=best_model,
    data=best_data,
    device=DEVICE,
)

y_true_200_real = final_eval["y_true_real"]
y_pred_200_real = final_eval["y_pred_real"]

recursive_predictions_df = pd.DataFrame(
    {
        "step": np.arange(1, RECURSIVE_STEPS + 1),
        "y_true_real": y_true_200_real,
        "y_pred_recursive_real": y_pred_200_real,
        "residual": y_true_200_real - y_pred_200_real,
        "absolute_error": np.abs(y_true_200_real - y_pred_200_real),
        "squared_error": (y_true_200_real - y_pred_200_real) ** 2,
    }
)

recursive_predictions_df.to_csv(
    "recursive_200_step_predictions_best_optuna.csv",
    index=False,
)

recursive_report_metrics = create_single_run_report(
    y_true=y_true_200_real,
    y_pred=y_pred_200_real,
    model_name="Best Optuna Recursive 200 Forward Lateral Causal CNN",
    output_dir="reports/optuna_recursive_200_forward_lateral_causal_cnn",
    n_confusion_bins=5,
    show=True,
)

final_summary = {
    "best_config": best_config,
    "final_retrained_best_metrics": best_metrics,
    "final_recursive_200_report_metrics": recursive_report_metrics,
}

with open("best_optuna_recursive_200_final_summary.json", "w") as f:
    json.dump(final_summary, f, indent=4)

print("\nFinal assignment metric:")
print(f"Best lookback p: {BEST_LOOKBACK}")
print(f"Recursive 200-step MAE, original scale:  {recursive_report_metrics['mae']:.6f}")
print(f"Recursive 200-step MSE, original scale:  {recursive_report_metrics['mse']:.6f}")
print(f"Recursive 200-step RMSE, original scale: {recursive_report_metrics['rmse']:.6f}")
print(f"Recursive 200-step R2, original scale:   {recursive_report_metrics['r2']:.6f}")

print("\nSaved files:")
print("- best_optuna_config_recursive_200.json")
print("- best_optuna_recursive_200_final_summary.json")
print("- best_optuna_recursive_200_forward_lateral_causal_cnn.pt")
print("- optuna_recursive_200_forward_lateral_causal_cnn_trials.csv")
print("- best_optuna_recursive_200_training_history.csv")
print("- recursive_200_step_predictions_best_optuna.csv")
print("- reports/optuna_recursive_200_forward_lateral_causal_cnn/")