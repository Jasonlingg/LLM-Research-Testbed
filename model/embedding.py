"""
Token and positional embeddings for GPT-2.

GPT-2 uses learned positional embeddings (not sinusoidal like the
original Transformer). This means position information is stored in
the pretrained weights, not computed at runtime.
"""
import torch
import torch.nn as nn
from torch import Tensor


class GPT2Embedding(nn.Module):
    """
    Combined token + positional embedding.
    
    GPT-2 specifics:
    - Vocabulary: 50257 tokens (BPE)
    - Max positions: 1024
    - Learned positional embeddings (not sinusoidal)
    - No dropout during inference
    """
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
    
    def forward(self, input_ids: Tensor, position_ids: Tensor = None) -> Tensor:
        """
        Args:
            input_ids: [batch, seq_len] token indices
            position_ids: [batch, seq_len] position indices (optional, auto-generated)
        
        Returns:
            [batch, seq_len, d_model] embedded representations
        """
        B, T = input_ids.shape
        
        if position_ids is None:
            position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        
        token_emb = self.token_embedding(input_ids)      # [B, T, d_model]
        pos_emb = self.position_embedding(position_ids)   # [B, T, d_model]
        
        return token_emb + pos_emb
