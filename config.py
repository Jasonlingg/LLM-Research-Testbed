"""
Centralized configuration for the inference engine.
Every optimization is a toggle. The benchmark harness iterates over configs.
"""
from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class ModelConfig:
    """GPT-2 124M architecture constants."""
    vocab_size: int = 50257
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072          # 4 * d_model
    max_seq_len: int = 1024
    dropout: float = 0.0       # No dropout during inference

    # Optimization flags (set from InferenceConfig)
    use_gqa: bool = False
    gqa_num_kv_groups: int = 4
    use_flash_attn: bool = False

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


@dataclass
class InferenceConfig:
    """What to toggle on/off during generation."""
    # Model
    model: ModelConfig = field(default_factory=ModelConfig)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    
    # KV Cache
    use_kv_cache: bool = True
    
    # Grouped-Query Attention
    use_gqa: bool = False
    gqa_num_kv_groups: int = 4    # 12 query heads → 4 KV groups

    # Flash Attention
    use_flash_attn: bool = False
    flash_block_size: int = 64    # Tile size for online softmax (Br = Bc)

    # Speculative Decoding
    use_speculative: bool = False
    spec_draft_n_layers: int = 4   # Draft model: first 4 layers of GPT-2
    spec_lookahead: int = 5        # Propose 5 tokens per step
    
    # Generation
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    
    def describe(self) -> str:
        """Human-readable description for benchmark labels."""
        parts = ["Baseline"]
        if self.use_kv_cache:
            parts = ["KV Cache"]
        if self.use_flash_attn:
            parts.append("FlashAttn")
        if self.use_gqa:
            parts.append(f"GQA({self.gqa_num_kv_groups}g)")
        if self.use_speculative:
            parts.append(f"Spec({self.spec_lookahead})")
        return " + ".join(parts)


# Pre-defined configurations for benchmarking
BENCHMARK_CONFIGS = {
    "baseline": InferenceConfig(use_kv_cache=False),
    "kv_cache": InferenceConfig(use_kv_cache=True),
    "flash_attn": InferenceConfig(use_kv_cache=True, use_flash_attn=True),
    "kv_cache_gqa": InferenceConfig(use_kv_cache=True, use_gqa=True, gqa_num_kv_groups=4),
    "kv_cache_spec": InferenceConfig(use_kv_cache=True, use_speculative=True),
    "all": InferenceConfig(use_kv_cache=True, use_gqa=True, gqa_num_kv_groups=4, use_speculative=True),
}
