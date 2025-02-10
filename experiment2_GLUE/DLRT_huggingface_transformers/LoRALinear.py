import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=8, alpha=16, lora_dropout=0.0):
        super(LoRALinear, self).__init__()
        self.original_linear = original_linear

        self.r = min(
            original_linear.out_features, min(original_linear.in_features, rank)
        )

        self.alpha = alpha

        # Creating low-rank adaptation matrices
        self.lora_A = nn.Parameter(torch.zeros((original_linear.in_features, rank)))
        self.lora_B = nn.Parameter(torch.zeros((rank, original_linear.out_features)))

        if lora_dropout > 0.0:
            self.lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout_layer = nn.Identity()

        # Scaling factor
        self.scaling = alpha / rank

        # Initialize parameters
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.original_linear(x) + self.scaling * (
            self.lora_dropout_layer(x) @ self.lora_A @ self.lora_B
        )

    def step(self, learning_rate, weight_decay):
        return 0

    def basis_grad_zero(self):
        return 0

    def coeff_grad_zero(self):
        return 0

    def Truncate(self):
        return 0
