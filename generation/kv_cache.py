"""
KV Cache Manager with memory tracking.

Key insight: during PREFILL we process all prompt tokens at once.
During DECODE we process one token at a time, appending to cache.

This implementation pre-allocates a contiguous buffer to avoid
repeated memory allocation (torch.cat every token is O(n²) total).
"""
import torch
from torch import Tensor
from typing import Optional, Tuple, List
from config import ModelConfig


class KVCache:
    """
    Pre-allocated Key-Value cache for efficient autoregressive generation.
    
    Memory layout:
        For each layer, we store K and V in contiguous buffers of shape
        [batch, n_heads, max_seq_len, d_head].
        
        Tokens 0..current_len are valid; the rest is pre-allocated zeros.
    
    Why pre-allocate:
        torch.cat([cached, new]) allocates a new tensor every step.
        Over 256 tokens, that's 256 allocations + copies = O(n²) memory ops.
        Pre-allocation: O(1) per step, just write into the next slot.
    """
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        d_head: int,
        max_seq_len: int,
        batch_size: int = 1,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.current_len = 0
        
        # Pre-allocate buffers: [n_layers, 2 (K+V), batch, n_heads, max_seq_len, d_head]
        self.cache = torch.zeros(
            n_layers, 2, batch_size, n_heads, max_seq_len, d_head,
            device=device, dtype=dtype
        )
    
    def update(
        self,
        layer_idx: int,
        k_new: Tensor,    # [batch, n_heads, T_new, d_head]
        v_new: Tensor,    # [batch, n_heads, T_new, d_head]
    ) -> Tuple[Tensor, Tensor]:
        """
        Write new K, V into cache and return full cached K, V.
        
        During prefill: T_new = prompt_length (many tokens at once)
        During decode:  T_new = 1 (one token at a time)
        
        Returns:
            k_full: [batch, n_heads, current_len + T_new, d_head]
            v_full: [batch, n_heads, current_len + T_new, d_head]
        """
        T_new = k_new.shape[2]
        start = self.current_len
        end = start + T_new
        
        assert end <= self.max_seq_len, \
            f"Cache overflow: trying to write position {end} but max is {self.max_seq_len}"
        
        # Write into pre-allocated buffer
        self.cache[layer_idx, 0, :, :, start:end, :] = k_new
        self.cache[layer_idx, 1, :, :, start:end, :] = v_new
        
        # Advance position pointer after the LAST layer processes
        # (all layers process the same tokens in one forward pass)
        if layer_idx == self.n_layers - 1:
            self.current_len = end
        
        # Return valid portion of cache
        k_full = self.cache[layer_idx, 0, :, :, :end, :]
        v_full = self.cache[layer_idx, 1, :, :, :end, :]
        
        return k_full, v_full
    
    def get_layer_cache(self, layer_idx: int) -> Optional[Tuple[Tensor, Tensor]]:
        """Get current cached K, V for a specific layer."""
        if self.current_len == 0:
            return None
        k = self.cache[layer_idx, 0, :, :, :self.current_len, :]
        v = self.cache[layer_idx, 1, :, :, :self.current_len, :]
        return (k, v)
    
    def reset(self):
        """Clear cache for new sequence."""
        self.current_len = 0
        # No need to zero out — we track valid length
    
    @property
    def memory_bytes(self) -> int:
        """Exact memory footprint of the full pre-allocated cache."""
        return self.cache.nelement() * self.cache.element_size()
    
    @property
    def memory_mb(self) -> float:
        return self.memory_bytes / (1024 * 1024)
    
    @property
    def utilization(self) -> float:
        """Fraction of allocated cache currently in use."""
        return self.current_len / self.max_seq_len
    
    def __repr__(self) -> str:
        return (
            f"KVCache(layers={self.n_layers}, heads={self.n_heads}, "
            f"len={self.current_len}/{self.max_seq_len}, "
            f"mem={self.memory_mb:.1f}MB, util={self.utilization:.1%})"
        )


def create_kv_cache(config: ModelConfig, batch_size: int = 1, device: str = "cpu") -> KVCache:
    """Convenience constructor from model config."""
    return KVCache(
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_head=config.d_head,
        max_seq_len=config.max_seq_len,
        batch_size=batch_size,
        device=device,
    )
