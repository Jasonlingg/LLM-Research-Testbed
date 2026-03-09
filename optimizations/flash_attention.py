"""
Flash Attention — tiled online softmax attention (Dao et al. 2022).

The key insight: standard attention materializes the full [B, H, T, S] score matrix,
which is O(N²) in memory. Flash Attention avoids this by computing attention in tiles,
using the online softmax recurrence to accumulate the correct result without ever
storing the full score matrix.

Online softmax recurrence (per Q-block, iterating over K/V blocks j):
    m_new = max(m_old, rowmax(Sij))
    l_new = exp(m_old - m_new) * l_old + sum(exp(Sij - m_new), dim=-1)
    O_new = exp(m_old - m_new) * O_old + exp(Sij - m_new) @ Vj
Then normalize: O_final = O / l

This gives the SAME result as standard softmax attention, but:
- Memory: O(N) instead of O(N²)   ← demonstrably true in pure PyTorch
- Speed: requires Triton/CUDA kernel to actually fuse tiles into SRAM
         (pure PyTorch cannot force tile residence, so wall-clock time
         may not improve and can be slightly worse due to overhead)

This implementation demonstrates the algorithm and achieves the memory property.
To get the IO speedup shown in the paper, you'd rewrite the tile loop as a
Triton kernel so each tile truly stays in SRAM.

Drop-in replacement for MultiHeadAttention:
- Same c_attn / c_proj attribute names → weight loading works unchanged
- Same forward signature: (x, kv_cache, use_cache) → (out, new_cache)
"""
import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple


class FlashAttention(nn.Module):
    """
    Flash Attention: tiled online-softmax self-attention.

    Identical interface and attribute names to MultiHeadAttention so the
    weight loader can populate c_attn / c_proj with zero changes.

    Algorithm detail (forward pass, no cache):
        Block sizes Br = Bc = min(block_size, T).
        For each Q tile qi (rows [r_start, r_end]):
            m_i = -inf, l_i = 0, O_i = 0      # running stats
            For each KV tile (kj, vj):
                Sij = qi @ kj.T / sqrt(d_head)  # [Br, Bc]
                apply causal mask to Sij
                m_ij = rowmax(Sij)
                P̃ij = exp(Sij - m_ij)          # unnormalized, numerically stable
                m_new = max(m_i, m_ij)
                l_new = exp(m_i - m_new)*l_i + exp(m_ij - m_new)*rowsum(P̃ij)
                O_new = exp(m_i - m_new)*O_i  + exp(m_ij - m_new)*(P̃ij @ vj)
                m_i, l_i, O_i = m_new, l_new, O_new
            O[r_start:r_end] = O_i / l_i[:, None]

    Causal masking: Sij[r, c] = -inf when absolute position of c > absolute
    position of r. The absolute offset from cached tokens is tracked via
    `kv_offset` so this works correctly during cached decode.
    """

    def __init__(self, d_model: int, n_heads: int, block_size: int = 64):
        super().__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.block_size = block_size

        # Same attribute names as MultiHeadAttention for weight-loading compatibility
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)

    def _tiled_attention(
        self,
        q: Tensor,   # [B, H, T_q, d_head]
        k: Tensor,   # [B, H, T_kv, d_head]
        v: Tensor,   # [B, H, T_kv, d_head]
        kv_offset: int = 0,
    ) -> Tensor:
        """
        Tiled online-softmax attention with causal masking.

        Args:
            q, k, v:   Attention tensors in [B, H, T, d_head] layout
            kv_offset: Number of cached tokens already seen before k/v start.
                       Used to compute absolute positions for causal masking.

        Returns:
            out: [B, H, T_q, d_head]

        NOTE on IO complexity:
            Pure PyTorch cannot pin tiles to SRAM; the actual speedup described
            in Dao et al. requires a Triton/CUDA kernel. What IS demonstrably
            correct here is the O(N) peak memory allocation — we never create
            the [B, H, T_q, T_kv] score tensor.
        """
        B, H, T_q, d = q.shape
        T_kv = k.shape[2]
        scale = d ** -0.5

        Br = min(self.block_size, T_q)
        Bc = min(self.block_size, T_kv)

        # Output accumulator — same shape as q, stays on same device/dtype
        out = torch.zeros_like(q)

        for r_start in range(0, T_q, Br):
            r_end = min(r_start + Br, T_q)
            qi = q[:, :, r_start:r_end, :]   # [B, H, Br, d_head]
            actual_Br = r_end - r_start

            # Running statistics per query row
            # m: running max  [B, H, Br]
            # l: running sum of exp  [B, H, Br]
            # O: running weighted sum of V  [B, H, Br, d_head]
            m_i = torch.full(
                (B, H, actual_Br), float("-inf"), device=q.device, dtype=q.dtype
            )
            l_i = torch.zeros((B, H, actual_Br), device=q.device, dtype=q.dtype)
            O_i = torch.zeros((B, H, actual_Br, d), device=q.device, dtype=q.dtype)

            for c_start in range(0, T_kv, Bc):
                c_end = min(c_start + Bc, T_kv)
                kj = k[:, :, c_start:c_end, :]   # [B, H, Bc, d_head]
                vj = v[:, :, c_start:c_end, :]   # [B, H, Bc, d_head]

                # Attention scores for this tile: [B, H, Br, Bc]
                Sij = torch.matmul(qi, kj.transpose(-2, -1)) * scale

                # Causal mask — absolute sequence positions:
                #   The Q sequence starts at absolute position kv_offset in the
                #   final sequence (0 during prefill, past_len during cached decode).
                #   The K sequence always starts at absolute position 0 because
                #   k = cat([k_cached (positions 0..kv_offset-1), k_new]).
                #
                #   q_abs[r] = kv_offset + r_start + r
                #   k_abs[c] = c_start + c
                #   mask where k_abs > q_abs (key is strictly after query → future)
                q_abs = torch.arange(
                    kv_offset + r_start, kv_offset + r_end, device=q.device
                )  # [Br]
                k_abs = torch.arange(c_start, c_end, device=q.device)  # [Bc]
                # mask[r, c] = True if k_abs[c] > q_abs[r]  (future token → mask)
                causal_mask = k_abs.unsqueeze(0) > q_abs.unsqueeze(1)  # [Br, Bc]
                Sij = Sij.masked_fill(
                    causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                )

                # Online softmax update
                m_ij = Sij.amax(dim=-1)          # [B, H, Br]  rowmax of this tile
                #
                # Numerical safety: when ALL keys in this tile are masked for a query
                # row, m_ij = -inf.  Using m_ij directly in Sij - m_ij gives
                # (-inf) - (-inf) = NaN.  Replace -inf with 0 just for the
                # subtraction so exp(-inf - 0) = 0 (masked entries stay zero).
                m_ij_safe = torch.where(
                    m_ij == float("-inf"), torch.zeros_like(m_ij), m_ij
                )
                P_tilde = torch.exp(Sij - m_ij_safe.unsqueeze(-1))  # [B, H, Br, Bc]

                m_new = torch.maximum(m_i, m_ij)   # [B, H, Br]

                # Correction factors for previously accumulated stats.
                # NaN can arise when both m_i and m_ij (hence m_new) are -inf,
                # meaning no valid (non-masked) key has been seen yet.
                # In that case O_i = 0 and l_i = 0, so the nan_to_num → 0 is correct.
                alpha = torch.nan_to_num(torch.exp(m_i  - m_new), nan=0.0)  # [B, H, Br]
                beta  = torch.nan_to_num(torch.exp(m_ij - m_new), nan=0.0)  # [B, H, Br]

                l_new = alpha * l_i + beta * P_tilde.sum(dim=-1)           # [B, H, Br]
                O_new = (
                    alpha.unsqueeze(-1) * O_i
                    + beta.unsqueeze(-1) * torch.matmul(P_tilde, vj)
                )  # [B, H, Br, d_head]

                m_i, l_i, O_i = m_new, l_new, O_new

            # Normalize by accumulated sum
            # Guard against all-masked rows (l_i == 0) which arise when T=1
            # and the only key is in the future (shouldn't happen with causal, but safe)
            l_safe = l_i.clamp(min=1e-6)
            out[:, :, r_start:r_end, :] = O_i / l_safe.unsqueeze(-1)

        return out

    def forward(
        self,
        x: Tensor,                                          # [B, T, d_model]
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,  # (cached_k, cached_v)
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Flash Attention forward pass.

        Args:
            x:        [B, T, d_model]
            kv_cache: Optional (cached_k, cached_v) each [B, H, T_cached, d_head]
            use_cache: Whether to return updated KV cache for next step

        Returns:
            out:      [B, T, d_model]
            new_cache: (k_full, v_full) if use_cache else None
        """
        B, T, D = x.shape

        # Project to Q, K, V
        qkv = self.c_attn(x)                           # [B, T, 3*d_model]
        q, k, v = qkv.split(self.d_model, dim=-1)      # each [B, T, d_model]

        # Reshape to [B, H, T, d_head]
        def _split_heads(t: Tensor) -> Tensor:
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q = _split_heads(q)
        k = _split_heads(k)
        v = _split_heads(v)

        # Prepend cached K, V if available
        kv_offset = 0
        if kv_cache is not None:
            k_cached, v_cached = kv_cache
            kv_offset = k_cached.shape[2]              # number of previously cached tokens
            k = torch.cat([k_cached, k], dim=2)        # [B, H, T_total, d_head]
            v = torch.cat([v_cached, v], dim=2)

        # Tiled online-softmax attention
        out = self._tiled_attention(q, k, v, kv_offset=kv_offset)  # [B, H, T, d_head]

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, d_model]
        out = self.c_proj(out)

        new_cache = (k, v) if use_cache else None
        return out, new_cache
