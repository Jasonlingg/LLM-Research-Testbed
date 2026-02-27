"""
Sampling strategies for text generation.

Greedy decoding (always pick most likely) is boring and repetitive.
Real generation uses temperature scaling and truncation methods.
"""
import torch
import torch.nn.functional as F
from torch import Tensor


def sample_token(
    logits: Tensor,         # [batch, vocab_size] — raw logits for next token
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
) -> Tensor:
    """
    Sample next token from logits with temperature, top-k, and top-p filtering.
    
    Pipeline:
        1. Temperature: divide logits by T (higher T = more random, lower T = more greedy)
        2. Top-k: zero out everything except the k most likely tokens
        3. Top-p (nucleus): keep smallest set of tokens whose cumulative prob >= p
        4. Sample from the filtered distribution
    
    Args:
        logits: [B, vocab_size] unnormalized logits
        temperature: Scaling factor (1.0 = normal, 0.0 = greedy)
        top_k: Keep only top k tokens (0 = disabled)
        top_p: Keep tokens with cumulative prob >= p (1.0 = disabled)
    
    Returns:
        [B, 1] sampled token indices
    """
    # Greedy shortcut
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    
    # Step 1: Temperature scaling
    logits = logits / temperature
    
    # Step 2: Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        # Find the k-th largest value as threshold
        threshold = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        # Zero out (set to -inf) anything below threshold
        logits = logits.masked_fill(logits < threshold, float('-inf'))
    
    # Step 3: Top-p (nucleus) filtering
    if top_p < 1.0:
        # Sort logits descending
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Find where cumulative prob exceeds top_p
        # Shift right so we keep the token that crosses the threshold
        sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        
        # Scatter mask back to original ordering
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float('-inf'))
        logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
    
    # Step 4: Sample from filtered distribution
    probs = F.softmax(logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1)
    
    return token


def greedy_decode(logits: Tensor) -> Tensor:
    """Simple greedy: always pick highest probability token."""
    return logits.argmax(dim=-1, keepdim=True)
