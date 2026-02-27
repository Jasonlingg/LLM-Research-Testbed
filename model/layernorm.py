"""
Layer Normalization from first principles.

GPT-2 uses pre-norm (LayerNorm before attention/MLP, not after).
This is a detail that matters for correctness against HF weights.
"""
import torch
import torch.nn as nn
from torch import Tensor


class LayerNorm(nn.Module):
    """
    Layer normalization with learnable affine parameters.
    
    Normalizes across the last dimension (d_model), then applies
    learned scale (gamma) and shift (beta).
    
    Math:
        y = gamma * (x - mean) / sqrt(var + eps) + beta
    
    Why from scratch: GPT-2's LayerNorm uses a specific eps=1e-5 and 
    the weight/bias naming must match HF's checkpoint keys exactly.
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # HF GPT-2 names these "weight" and "bias"
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x: Tensor) -> Tensor:
        # Compute mean and variance across last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Affine transform with learned parameters
        return self.weight * x_norm + self.bias
