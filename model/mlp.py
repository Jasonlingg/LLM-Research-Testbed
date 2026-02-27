"""
Position-wise Feed-Forward Network (MLP) for GPT-2.

GPT-2 specifics:
- Uses GELU activation (not ReLU)
- Expands from d_model to 4*d_model, then back
- HF names these c_fc and c_proj
- Uses the "approximate" GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
"""
import torch
import torch.nn as nn
from torch import Tensor
import math


class MLP(nn.Module):
    """
    Two-layer feed-forward network with GELU activation.
    
    Architecture: d_model → 4*d_model → d_model
    
    This is applied independently to each position (token),
    which is why it's called "position-wise."
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # GPT-2 naming convention (matches HF checkpoint keys)
        self.c_fc = nn.Linear(d_model, d_ff)
        self.c_proj = nn.Linear(d_ff, d_model)
    
    def gelu(self, x: Tensor) -> Tensor:
        """
        GELU activation — Gaussian Error Linear Unit.
        
        GPT-2 uses the approximate version, not the exact erf-based one.
        This is the "tanh approximation" from the original GELU paper.
        
        Exact:   0.5 * x * (1 + erf(x / sqrt(2)))
        Approx:  0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        
        We use the approximate version to match HF's GPT-2 outputs exactly.
        """
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
        ))
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]
        """
        x = self.c_fc(x)       # [B, T, d_ff]   — expand
        x = self.gelu(x)       # [B, T, d_ff]   — activate
        x = self.c_proj(x)     # [B, T, d_model] — compress back
        return x
