"""
Grouped-Query Attention (GQA) — Ainslie et al., 2023

Standard MHA: 12 query heads, 12 KV heads  (1:1 mapping)
GQA:          12 query heads, 4 KV heads   (3:1 mapping)
MQA:          12 query heads, 1 KV head    (12:1 mapping)

Memory savings: KV cache shrinks by factor of (n_heads / n_kv_groups).
For GPT-2 with 12 heads → 4 KV groups: 3x smaller KV cache.

Quality impact: minimal, because nearby attention heads tend to learn
similar key-value patterns. Sharing KV heads within a group preserves
most of the representational capacity.

KEY IMPLEMENTATION: Converting pretrained MHA weights to GQA.
Most tutorials skip this. We average K,V projection weights within
each group, which is the standard "uptraining-free" conversion.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


class GroupedQueryAttention(nn.Module):
    """
    GQA: multiple query heads share each KV head.
    
    For GPT-2 124M with n_kv_groups=4:
        - 12 query heads (same as standard)
        - 4 KV heads (reduced from 12)
        - Each KV head serves 3 query heads
        - KV cache per layer: 4 × d_head instead of 12 × d_head
    """
    def __init__(self, d_model: int, n_heads: int, n_kv_groups: int):
        super().__init__()
        assert n_heads % n_kv_groups == 0, \
            f"n_heads ({n_heads}) must be divisible by n_kv_groups ({n_kv_groups})"
        
        self.d_model = d_model
        self.n_heads = n_heads              # Query heads (12)
        self.n_kv_groups = n_kv_groups      # KV heads (4)
        self.heads_per_group = n_heads // n_kv_groups  # 3
        self.d_head = d_model // n_heads    # 64
        
        # Query projection: full n_heads (unchanged from MHA)
        self.W_q = nn.Linear(d_model, n_heads * self.d_head)
        
        # Key and Value projections: REDUCED to n_kv_groups
        # This is where the memory savings come from
        self.W_k = nn.Linear(d_model, n_kv_groups * self.d_head)
        self.W_v = nn.Linear(d_model, n_kv_groups * self.d_head)
        
        # Output projection (unchanged)
        self.c_proj = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        x: Tensor,                                   # [B, T, d_model]
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        B, T, D = x.shape
        
        # Project Q (full heads), K and V (reduced heads)
        q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # q: [B, 12, T, 64]
        
        k = self.W_k(x).view(B, T, self.n_kv_groups, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_kv_groups, self.d_head).transpose(1, 2)
        # k, v: [B, 4, T, 64]
        
        # KV cache update (cache is smaller because fewer KV heads!)
        if kv_cache is not None:
            k_cached, v_cached = kv_cache
            k = torch.cat([k_cached, k], dim=2)
            v = torch.cat([v_cached, v], dim=2)
        
        T_total = k.shape[2]
        
        # CRITICAL: Expand K,V to match Q's head count
        # Each KV group is "broadcast" to serve multiple query heads
        # [B, 4, T, 64] → [B, 12, T, 64]
        #
        # repeat_interleave keeps heads in order:
        # KV head 0 → query heads 0,1,2
        # KV head 1 → query heads 3,4,5
        # etc.
        k_expanded = k.repeat_interleave(self.heads_per_group, dim=1)
        v_expanded = v.repeat_interleave(self.heads_per_group, dim=1)
        
        # Standard attention math from here
        scale = self.d_head ** -0.5
        scores = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T_total, dtype=torch.bool, device=x.device),
            diagonal=T_total - T + 1
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v_expanded)
        
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.c_proj(out)
        
        # Cache the REDUCED K,V (not expanded), saving memory
        new_cache = (k, v) if use_cache else None
        
        return out, new_cache
    
    @classmethod
    def from_pretrained_mha(
        cls,
        c_attn_weight: Tensor,   # [d_model, 3 * d_model] combined QKV from GPT-2
        c_attn_bias: Tensor,     # [3 * d_model]
        c_proj_weight: Tensor,   # [d_model, d_model]
        c_proj_bias: Tensor,     # [d_model]
        d_model: int,
        n_heads: int,
        n_kv_groups: int,
    ) -> "GroupedQueryAttention":
        """
        Convert pretrained standard MHA weights to GQA weights.
        
        Strategy: average K,V projection weights within each group.
        
        For GPT-2 with 12 heads → 4 groups:
            Group 0: average K weights of heads 0, 1, 2
            Group 1: average K weights of heads 3, 4, 5
            Group 2: average K weights of heads 6, 7, 8
            Group 3: average K weights of heads 9, 10, 11
        Same for V weights.
        
        Why averaging works: nearby heads tend to learn similar K,V patterns.
        Averaging preserves the "center" of each group's representation.
        
        This is the part that shows you understand what's actually happening,
        not just how to use a library.
        """
        d_head = d_model // n_heads
        heads_per_group = n_heads // n_kv_groups
        
        gqa = cls(d_model, n_heads, n_kv_groups)
        
        # Split combined QKV weight into Q, K, V
        # GPT-2 c_attn stores [Q|K|V] concatenated along the output dimension
        # Weight shape after transpose: [3*d_model, d_model] → split on dim 0
        q_weight = c_attn_weight[:, :d_model]               # [d_model, d_model]
        k_weight = c_attn_weight[:, d_model:2*d_model]      # [d_model, d_model]
        v_weight = c_attn_weight[:, 2*d_model:]             # [d_model, d_model]
        
        q_bias = c_attn_bias[:d_model]
        k_bias = c_attn_bias[d_model:2*d_model]
        v_bias = c_attn_bias[2*d_model:]
        
        # Q projection: keep as-is (full n_heads)
        gqa.W_q.weight.data = q_weight.T.contiguous()  # Transpose for Conv1D → Linear
        gqa.W_q.bias.data = q_bias.clone()
        
        # K projection: average within groups
        # Reshape: [d_model, d_model] → [d_model, n_heads, d_head]
        # → [d_model, n_kv_groups, heads_per_group, d_head]
        # → mean over heads_per_group → [d_model, n_kv_groups, d_head]
        k_weight_heads = k_weight.view(d_model, n_heads, d_head)
        k_weight_grouped = k_weight_heads.view(d_model, n_kv_groups, heads_per_group, d_head)
        k_weight_avg = k_weight_grouped.mean(dim=2)  # Average within groups
        gqa.W_k.weight.data = k_weight_avg.reshape(d_model, n_kv_groups * d_head).T.contiguous()
        
        # Average K bias within groups too
        k_bias_heads = k_bias.view(n_heads, d_head)
        k_bias_grouped = k_bias_heads.view(n_kv_groups, heads_per_group, d_head)
        gqa.W_k.bias.data = k_bias_grouped.mean(dim=1).reshape(-1)
        
        # V projection: same averaging
        v_weight_heads = v_weight.view(d_model, n_heads, d_head)
        v_weight_grouped = v_weight_heads.view(d_model, n_kv_groups, heads_per_group, d_head)
        v_weight_avg = v_weight_grouped.mean(dim=2)
        gqa.W_v.weight.data = v_weight_avg.reshape(d_model, n_kv_groups * d_head).T.contiguous()
        
        v_bias_heads = v_bias.view(n_heads, d_head)
        v_bias_grouped = v_bias_heads.view(n_kv_groups, heads_per_group, d_head)
        gqa.W_v.bias.data = v_bias_grouped.mean(dim=1).reshape(-1)
        
        # Output projection: keep as-is
        gqa.c_proj.weight.data = c_proj_weight.T.contiguous()
        gqa.c_proj.bias.data = c_proj_bias.clone()
        
        return gqa
    
    @property
    def kv_cache_size_per_token(self) -> int:
        """Bytes of KV cache per token per layer."""
        # K + V, each with n_kv_groups heads × d_head dimensions
        return 2 * self.n_kv_groups * self.d_head * 4  # 4 bytes for float32
    
    @property
    def memory_reduction_factor(self) -> float:
        """How much smaller the KV cache is vs standard MHA."""
        return self.n_heads / self.n_kv_groups
