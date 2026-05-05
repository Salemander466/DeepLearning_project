import copy
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# For reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    
#Model Components: TCN-inspired Causal Residual CNN
#First, CausalConv1d is the basic time-series convolution layer. It pads only on the left side of the sequence:
#That means the model only looks backward in time. It does not look at future values. This is important because forecasting must only use past information.
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            dilation = dilation,
            padding=0,
        )
        
    def forward(self,x):
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)
    
# The CausalResidualBlock is a standard residual block with two causal convolutional layers, batch normalization, and dropout. The residual connection ensures that the model can learn identity mappings if needed, which helps with training deeper networks. 
class CausalResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        dropout=0.15,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.conv2 = CausalConv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        if in_channels != out_channels:
            self.residual_projection = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )
        else:
            self.residual_projection = nn.Identity()
    
    #How the model is connected
    def forward(self, x):
        residual = self.residual_projection(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.dropout(out)
        
        #That is the residual, or skip, connection. It means the block keeps the original information and adds the new learned pattern on top of it. This helps the model train more stably and prevents useful earlier information from being lost
        out = out + residual
        out = self.activation(out)
        
        return out
    

# The ForwardLateralCausalCNN is the main model class. It consists of an initial causal convolutional layer (the "stem") followed by 6 residual blocks with increasing dilation rates. The dilation rates (1, 2, 4, 8, 16, 32) allow the model to capture patterns at multiple time scales. After the convolutional layers, it applies global average pooling and takes the last time step's features, concatenates them, and passes through a fully connected regressor to produce the final forecast.
class ForwardLateralCausalCNN(nn.Module):
    def __init__(
        self,
        input_channels,
        base_channels=32,
        kernel_size=3,
        dropout=0.15,
    ):
        super().__init__()

        self.input_channels = input_channels

        self.stem = nn.Sequential(
            CausalConv1d(
                in_channels=input_channels,
                out_channels=base_channels,
                kernel_size=kernel_size,
                dilation=1,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=base_channels,
            ),
            nn.GELU(),
        )

        #The dilation values are important: they allow the model to capture patterns at different time scales. The first block captures short-term patterns, while the later blocks can capture longer-term dependencies without needing a very deep network.
        #base_channels
        self.block1 = CausalResidualBlock(
            in_channels=base_channels,
            out_channels=base_channels,
            kernel_size=kernel_size,
            dilation=1,
            dropout=dropout,
        )
        #base_channels
        self.block2 = CausalResidualBlock(
            in_channels=base_channels,
            out_channels=base_channels,
            kernel_size=kernel_size,
            dilation=2,
            dropout=dropout,
        )

        #base_channels * 2
        self.block3 = CausalResidualBlock(
            in_channels=base_channels,
            out_channels=base_channels * 2,
            kernel_size=kernel_size,
            dilation=4,
            dropout=dropout,
        )

        #base_channels * 2
        self.block4 = CausalResidualBlock(
            in_channels=base_channels * 2,
            out_channels=base_channels * 2,
            kernel_size=kernel_size,
            dilation=8,
            dropout=dropout,
        )

        #base_channels * 4
        self.block5 = CausalResidualBlock(
            in_channels=base_channels * 2,
            out_channels=base_channels * 4,
            kernel_size=kernel_size,
            dilation=16,
            dropout=dropout,
        )
        
        #base_channels * 4
        self.block6 = CausalResidualBlock(
            in_channels=base_channels * 4,
            out_channels=base_channels * 4,
            kernel_size=kernel_size,
            dilation=32,
            dropout=dropout,
        )

        final_channels = base_channels * 4

        self.regressor = nn.Sequential(
            nn.Linear(final_channels * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.GELU(),

            nn.Linear(64, 1),
        )
        
    #Makes sure the input has the correct shape before it goes into Conv1d.
    def _fix_input_shape(self, x):
        if x.ndim != 3:
            raise ValueError(
                f"Expected 3D input. Got shape {tuple(x.shape)}."
            )

        # Already PyTorch format: (batch, channels, lookback)
        if x.shape[1] == self.input_channels:
            return x

        # Keras format: (batch, lookback, channels)
        if x.shape[2] == self.input_channels:
            return x.permute(0, 2, 1)

        raise ValueError(
            f"Input shape {tuple(x.shape)} does not match "
            f"input_channels={self.input_channels}."
        )
    def forward(self, x):
        x = self._fix_input_shape(x)

        out1 = self.stem(x)
        out2 = self.block1(out1)
        out3 = self.block2(out2)
        out4 = self.block3(out3)
        out5 = self.block4(out4)
        out6 = self.block5(out5)
        out7 = self.block6(out6)

        #After the convolution blocks, the model creates two summaries:
        #global_average summarizes what the model learned across the whole input window.
        #last_state summarizes what the model learned at the most recent time step.
        global_average = out7.mean(dim=-1)
        last_state = out7[:, :, -1]


        #Then it combines them:
        features = torch.cat([global_average, last_state], dim=1)
        #So the final regressor gets both: overall window information + most recent time information. This allows the model to make a more informed forecast by considering both the general patterns in the window and the latest state.
        
        output = self.regressor(features)

        return output.squeeze(-1)
    
    
    


#L1 and L2 regularized loss function for training the model. This is used in the training loop to compute the loss with both MSE and regularization penalties.
def l1_l2_regularized_loss(
    model,
    y_pred,
    y_true,
    l1_lambda=1e-6,
    l2_lambda=1e-5,
):
    
    mse_loss = F.mse_loss(y_pred, y_true)
    
    l1_penalty = torch.tensor(0.0, device=y_pred.device)
    l2_penalty = torch.tensor(0.0, device=y_pred.device)
    
    for name, pram in model.named_parameters():
        if not pram.requires_grad:
            continue
        if pram.requires_grad:
            l1_penalty += pram.abs().sum()
            l2_penalty += pram.pow(2).sum()
            
    total_loss = mse_loss + l1_lambda * l1_penalty + l2_lambda * l2_penalty
    
    return total_loss