
from data_loader import prepare_train_val_data
from util import inspect_mat_file

#Load Data

print(inspect_mat_file("Xtrain.mat"))

data = prepare_train_val_data(
    file_path="Xtrain.mat",
    variable_name=None,
    column=None,
    lookback=30,
    val_fraction=0.2,
    scaler_type="standard",
    framework="keras",
)

print("X_train shape:", data["X_train"].shape)
print("y_train shape:", data["y_train"].shape)
print("X_val shape:", data["X_val"].shape)
print("y_val shape:", data["y_val"].shape)

#infer input channels form current data format



X_shape = data["X_train"].shape

if len(X_shape) != 3:
    raise ValueError(
        f"Expected X_train to be 3D. Got shape {X_shape}."
    )

lookback = data["lookback"]

# Keras-style: (samples, lookback, features)
if X_shape[1] == lookback:
    input_channels = X_shape[2]

# PyTorch-style: (samples, features, lookback)
elif X_shape[2] == lookback:
    input_channels = X_shape[1]

else:
    raise ValueError(
        f"Cannot infer input channels from X_train shape {X_shape} "
        f"and lookback {lookback}."
    )

print("Detected input channels:", input_channels)