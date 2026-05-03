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
    
    
#Model Components

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
            self.residual_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )
        else:
            self.residual_projection = nn.Identity()
            
    def forward(self, x):
        residual = self.residual_projection(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.dropout(out)
        
        out = out + residual
        out = self.activation(out)
        
        return out
    
    
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

        self.block1 = CausalResidualBlock(
            in_channels=base_channels,
            out_channels=base_channels,
            kernel_size=kernel_size,
            dilation=1,
            dropout=dropout,
        )

        self.block2 = CausalResidualBlock(
            in_channels=base_channels,
            out_channels=base_channels,
            kernel_size=kernel_size,
            dilation=2,
            dropout=dropout,
        )

        self.block3 = CausalResidualBlock(
            in_channels=base_channels,
            out_channels=base_channels * 2,
            kernel_size=kernel_size,
            dilation=4,
            dropout=dropout,
        )

        self.block4 = CausalResidualBlock(
            in_channels=base_channels * 2,
            out_channels=base_channels * 2,
            kernel_size=kernel_size,
            dilation=8,
            dropout=dropout,
        )

        self.block5 = CausalResidualBlock(
            in_channels=base_channels * 2,
            out_channels=base_channels * 4,
            kernel_size=kernel_size,
            dilation=16,
            dropout=dropout,
        )

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
        
    def _fix_input_shape(self, x):
        if x.ndim != 3:
            
            raise ValueError(f"Input must be 3D (batch_size, channels, timesteps). Got {x.shape}D.")
        
        if x.shape[1] != self.input_channels:
            return x 
        
        if x.shape[2] == self.input_channels:
            return x.permute(0, 2, 1)
        
        raise ValueError(f"Input has {x.shape[1]} channels and {x.shape[2]} timesteps, but expected {self.input_channels} channels.")
    
    def foward(self, x):
        x = self._fix_input_shape(x)
        
        out1 = self.stem(x)
        out2 = self.block1(out1)
        out3 = self.block2(out2)
        out4 = self.block3(out3)
        out5 = self.block4(out4)
        out6 = self.block5(out5)
        out7 = self.block6(out6)

        global_average = x.mean(dim=1)
        last_state = out7[:, :, -1]
        features = torch.cat([global_average, last_state], dim=1)
        output = self.regressor(features)
        
        return output.squeeze(1)
    
    
    

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