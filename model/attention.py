"""
Multi-Head Attention from raw matrix operations.

No nn.MultiheadAttention. No F.scaled_dot_product_attention.
Just matrix math you can walk through in an interview.

GPT-2 specifics:
- Uses a combined QKV projection (one big linear layer, not three separate ones)
- Causal mask (can only attend to previous tokens)
- 12 heads, 64-dim per head for GPT-2 124M
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention computed from raw operations.
    
    The forward pass does exactly this:
        1. Project input to Q, K, V via learned linear layers
        2. Split into heads: [B, T, d_model] → [B, n_heads, T, d_head]
        3. Compute attention scores: Q @ K^T / sqrt(d_head)
        4. Apply causal mask (prevent attending to future)
        5. Softmax over key dimension
        6. Weighted sum of values: weights @ V
        7. Concatenate heads and project output
    
    GPT-2 uses a COMBINED c_attn projection (QKV in one matrix) and
    a c_proj output projection. We split them for clarity but the
    weight_loader maps them correctly.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        
        # GPT-2 uses a single combined projection for Q, K, V
        # Shape: [d_model, 3 * d_model] — we'll split the output
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        
        # Output projection
        self.c_proj = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        x: Tensor,                                  # [batch, seq_len, d_model]
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,  # (cached_k, cached_v)
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Args:
            x: Input tensor [B, T, d_model]
            kv_cache: Optional tuple of (cached_keys, cached_values)
                      each with shape [B, n_heads, T_cached, d_head]
            use_cache: Whether to return updated KV cache
        
        Returns:
            output: [B, T, d_model]
            new_kv_cache: Optional tuple of (keys, values) for next step
        """
        B, T, D = x.shape
        
        # === Step 1: Project to Q, K, V ===
        # Combined projection, then split
        qkv = self.c_attn(x)                           # [B, T, 3 * d_model]
        q, k, v = qkv.split(self.d_model, dim=-1)      # each [B, T, d_model]
        
        # === Step 2: Reshape into heads ===
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, n_heads, T, d_head]
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # === Step 3: KV Cache — append new K, V to history ===
        if kv_cache is not None:
            k_cached, v_cached = kv_cache
            k = torch.cat([k_cached, k], dim=2)   # [B, n_heads, T_total, d_head]
            v = torch.cat([v_cached, v], dim=2)
        
        T_total = k.shape[2]  # Full sequence length including cache
        
        # === Step 4: Attention scores ===
        # Q @ K^T, scaled by sqrt(d_head) to prevent large dot products
        # that push softmax into saturated regions with tiny gradients
        scale = self.d_head ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        # scores shape: [B, n_heads, T_query, T_total]
        
        # === Step 5: Causal mask ===
        # Each token can only attend to itself and previous tokens
        # We create a mask where position i can attend to positions 0..i
        causal_mask = torch.triu(
            torch.ones(T, T_total, dtype=torch.bool, device=x.device),
            diagonal=T_total - T + 1
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # === Step 6: Softmax over key dimension ===
        weights = F.softmax(scores, dim=-1)
        # weights shape: [B, n_heads, T_query, T_total]
        
        # === Step 7: Weighted sum of values ===
        out = torch.matmul(weights, v)  # [B, n_heads, T_query, d_head]
        
        # === Step 8: Concatenate heads and project ===
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, d_model]
        out = self.c_proj(out)
        
        # Return updated cache if requested
        new_cache = (k, v) if use_cache else None
        
        return out, new_cache
