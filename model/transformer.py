"""
GPT-2 124M Transformer — Full architecture from components.

This assembles embedding, attention, MLP, and layernorm into the
complete GPT-2 model. The architecture exactly mirrors HuggingFace's
GPT2LMHeadModel so we can load their pretrained weights.

Architecture (GPT-2 uses pre-norm, not post-norm):
    Input → Embedding → [LayerNorm → Attention → + → LayerNorm → MLP → +] × 12 → LayerNorm → LM Head

Key GPT-2 detail: the LM head (output projection to vocabulary) shares
weights with the token embedding. This is called "weight tying."
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, List, Tuple

from model.attention import MultiHeadAttention
from model.mlp import MLP
from model.layernorm import LayerNorm
from model.embedding import GPT2Embedding
from optimizations.grouped_query_attention import GroupedQueryAttention
from config import ModelConfig


class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-norm architecture.
    
    Pre-norm (GPT-2):  x → LayerNorm → Attention → + residual → LayerNorm → MLP → + residual
    Post-norm (orig):  x → Attention → + residual → LayerNorm → MLP → + residual → LayerNorm
    
    Pre-norm is more stable for training deep networks. The residual
    connections ensure gradients can flow directly through the network.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.d_model)
        if config.use_gqa:
            self.attn = GroupedQueryAttention(config.d_model, config.n_heads, config.gqa_num_kv_groups)
        else:
            self.attn = MultiHeadAttention(config.d_model, config.n_heads)
        self.ln_2 = LayerNorm(config.d_model)
        self.mlp = MLP(config.d_model, config.d_ff)
    
    def forward(
        self,
        x: Tensor,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        # Pre-norm attention with residual
        residual = x
        x = self.ln_1(x)
        attn_out, new_cache = self.attn(x, kv_cache=kv_cache, use_cache=use_cache)
        x = residual + attn_out
        
        # Pre-norm MLP with residual
        residual = x
        x = self.ln_2(x)
        x = residual + self.mlp(x)
        
        return x, new_cache


class GPT2(nn.Module):
    """
    Complete GPT-2 124M language model.
    
    Parameters: ~124M
    - Embedding: 50257 × 768 = ~38.6M
    - 12 transformer blocks × ~7.1M each = ~85.1M
    - Final LayerNorm: ~1.5K
    - LM Head: tied with embedding (0 extra params)
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token + positional embeddings
        self.embedding = GPT2Embedding(config.vocab_size, config.d_model, config.max_seq_len)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final layer norm (before LM head)
        self.ln_f = LayerNorm(config.d_model)
        
        # LM head — projects hidden states to vocabulary logits
        # Weight-tied with token embedding (standard GPT-2 practice)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights: lm_head.weight = embedding.token_embedding.weight
        self.lm_head.weight = self.embedding.token_embedding.weight
    
    def forward(
        self,
        input_ids: Tensor,                           # [batch, seq_len]
        position_ids: Optional[Tensor] = None,       # [batch, seq_len]
        kv_caches: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[List[Tuple[Tensor, Tensor]]]]:
        """
        Full forward pass: tokens → logits.
        
        Args:
            input_ids:    Token indices [B, T]
            position_ids: Position indices [B, T] (auto-generated if None)
            kv_caches:    List of (K, V) tuples, one per layer
            use_cache:    Whether to return updated KV caches
        
        Returns:
            logits:       [B, T, vocab_size] — unnormalized predictions
            new_caches:   List of (K, V) tuples if use_cache=True
        """
        B, T = input_ids.shape
        
        # Handle position IDs for cached generation
        # When using KV cache, position_ids should start from the cached length
        if position_ids is None and kv_caches is not None and kv_caches[0] is not None:
            past_len = kv_caches[0][0].shape[2]  # T_cached
            position_ids = torch.arange(past_len, past_len + T, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(B, -1)
        
        # Embed tokens + positions
        x = self.embedding(input_ids, position_ids)  # [B, T, d_model]
        
        # Pass through transformer blocks
        new_caches = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, kv_cache=layer_cache, use_cache=use_cache)
            new_caches.append(new_cache)
        
        # Final layernorm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)  # [B, T, vocab_size]
        
        return logits, new_caches if use_cache else None
    
    @property
    def num_parameters(self) -> int:
        """Count total parameters (excluding tied weights)."""
        # Don't double-count tied lm_head weights
        return sum(p.numel() for p in self.parameters()) - self.lm_head.weight.numel()

    def get_num_layers(self) -> int:
        return len(self.blocks)
