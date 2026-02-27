"""
Autoregressive text generation engine.

This is the main generation loop. It supports:
- Naive generation (recompute everything each step) — baseline
- KV-cached generation (cache K,V from previous steps) — optimized
- Metrics collection (tokens/sec, TTFT, memory) — for benchmarking

The loop is intentionally explicit (not wrapped in a library) so you
can see exactly what happens at each step.
"""
import time
import torch
import tiktoken
from torch import Tensor
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

from model.transformer import GPT2
from generation.kv_cache import KVCache, create_kv_cache
from generation.sampling import sample_token
from config import InferenceConfig, ModelConfig


@dataclass
class GenerationMetrics:
    """Metrics collected during a single generation run."""
    total_time_s: float = 0.0
    time_to_first_token_ms: float = 0.0
    tokens_generated: int = 0
    prompt_length: int = 0
    peak_memory_bytes: int = 0
    per_step_time_ms: List[float] = field(default_factory=list)
    cache_memory_mb: float = 0.0
    
    @property
    def tokens_per_sec(self) -> float:
        if self.total_time_s == 0:
            return 0.0
        return self.tokens_generated / self.total_time_s
    
    @property
    def avg_step_time_ms(self) -> float:
        if not self.per_step_time_ms:
            return 0.0
        return sum(self.per_step_time_ms) / len(self.per_step_time_ms)
    
    def summary(self) -> str:
        return (
            f"Tokens: {self.tokens_generated} | "
            f"Tok/s: {self.tokens_per_sec:.1f} | "
            f"TTFT: {self.time_to_first_token_ms:.1f}ms | "
            f"Avg step: {self.avg_step_time_ms:.1f}ms | "
            f"Cache: {self.cache_memory_mb:.1f}MB"
        )


class Generator:
    """
    Autoregressive text generation with optional KV caching.
    
    Two modes:
    
    1. NAIVE (use_kv_cache=False):
       Each step feeds the ENTIRE sequence so far through the model.
       Cost per step: O(n) where n = current sequence length.
       Total cost: O(n²) over the full generation.
       
    2. CACHED (use_kv_cache=True):
       Prefill: process entire prompt once, cache K,V for all layers.
       Decode: each step feeds only the NEW token, reusing cached K,V.
       Cost per step: O(1) for the new token (attention still looks at full cache).
       Total cost: O(n) over the full generation.
    """
    def __init__(self, model: GPT2, config: InferenceConfig):
        self.model = model
        self.config = config
        self.device = config.device
        
        # Tokenizer — using tiktoken (lightweight, no HF dependency)
        self.tokenizer = tiktoken.get_encoding("gpt2")
    
    def encode(self, text: str) -> Tensor:
        """Tokenize text to tensor."""
        tokens = self.tokenizer.encode(text)
        return torch.tensor([tokens], dtype=torch.long, device=self.device)
    
    def decode(self, token_ids: Tensor) -> str:
        """Detokenize tensor to text."""
        return self.tokenizer.decode(token_ids.squeeze(0).tolist())
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tuple[str, GenerationMetrics]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text
            max_new_tokens: Override config.max_new_tokens
            temperature: Override config.temperature
            top_k: Override config.top_k
            top_p: Override config.top_p
        
        Returns:
            generated_text: The complete generated text (prompt + new tokens)
            metrics: Timing and memory metrics
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        
        # Tokenize prompt
        input_ids = self.encode(prompt)  # [1, prompt_len]
        prompt_len = input_ids.shape[1]
        
        metrics = GenerationMetrics(prompt_length=prompt_len)
        
        if self.config.use_kv_cache:
            output_ids, metrics = self._generate_with_cache(
                input_ids, max_new_tokens, temperature, top_k, top_p, metrics
            )
        else:
            output_ids, metrics = self._generate_naive(
                input_ids, max_new_tokens, temperature, top_k, top_p, metrics
            )
        
        generated_text = self.decode(output_ids)
        return generated_text, metrics
    
    def _generate_naive(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        metrics: GenerationMetrics,
    ) -> Tuple[Tensor, GenerationMetrics]:
        """
        Naive generation: feed entire sequence each step.
        
        This is the BASELINE. It's slow because at step n, the model
        processes all n tokens, even though it only needs the last logit.
        Total compute: O(1 + 2 + 3 + ... + n) = O(n²)
        """
        generated = input_ids.clone()
        gen_start = time.perf_counter()
        
        for i in range(max_new_tokens):
            step_start = time.perf_counter()
            
            # Feed ENTIRE sequence through model
            # This recomputes attention for ALL previous tokens every step
            logits, _ = self.model(generated, use_cache=False)
            
            # Only need logits for the last position
            next_logits = logits[:, -1, :]
            
            # Sample next token
            next_token = sample_token(next_logits, temperature, top_k, top_p)
            
            step_time = (time.perf_counter() - step_start) * 1000
            metrics.per_step_time_ms.append(step_time)
            
            # Record TTFT
            if i == 0:
                metrics.time_to_first_token_ms = step_time
            
            # Append token
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop on EOS
            if next_token.item() == self.tokenizer.eot_token:
                break
            
            # Safety: don't exceed max sequence length
            if generated.shape[1] >= self.config.model.max_seq_len:
                break
        
        metrics.total_time_s = time.perf_counter() - gen_start
        metrics.tokens_generated = generated.shape[1] - input_ids.shape[1]
        
        return generated, metrics
    
    def _generate_with_cache(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        metrics: GenerationMetrics,
    ) -> Tuple[Tensor, GenerationMetrics]:
        """
        KV-cached generation: prefill once, then decode one token at a time.
        
        Phase 1 — PREFILL:
            Process entire prompt in one forward pass.
            Cache all K,V for every layer.
            This is compute-bound (lots of tokens to process).
        
        Phase 2 — DECODE:
            Process only the NEW token each step.
            Attention reuses cached K,V (just appends one new K,V pair).
            This is memory-bound (reading cached K,V dominates).
        """
        generated_tokens = []
        gen_start = time.perf_counter()
        
        # === PHASE 1: Prefill ===
        # Process entire prompt, cache K,V
        prefill_start = time.perf_counter()
        logits, kv_caches = self.model(input_ids, use_cache=True)
        prefill_time = (time.perf_counter() - prefill_start) * 1000
        
        # Sample first new token from last position's logits
        next_logits = logits[:, -1, :]
        next_token = sample_token(next_logits, temperature, top_k, top_p)
        generated_tokens.append(next_token)
        
        metrics.time_to_first_token_ms = prefill_time
        metrics.per_step_time_ms.append(prefill_time)
        
        # === PHASE 2: Decode ===
        # One token at a time, reusing cached K,V
        for i in range(max_new_tokens - 1):
            step_start = time.perf_counter()
            
            # Feed ONLY the new token (not the whole sequence!)
            logits, kv_caches = self.model(
                next_token,          # [1, 1] — just the new token
                kv_caches=kv_caches, # reuse cached K,V
                use_cache=True,
            )
            
            next_logits = logits[:, -1, :]
            next_token = sample_token(next_logits, temperature, top_k, top_p)
            generated_tokens.append(next_token)
            
            step_time = (time.perf_counter() - step_start) * 1000
            metrics.per_step_time_ms.append(step_time)
            
            # Stop on EOS
            if next_token.item() == self.tokenizer.eot_token:
                break
        
        # Reconstruct full sequence
        new_tokens = torch.cat(generated_tokens, dim=1)
        output_ids = torch.cat([input_ids, new_tokens], dim=1)
        
        metrics.total_time_s = time.perf_counter() - gen_start
        metrics.tokens_generated = new_tokens.shape[1]
        
        # Cache memory tracking
        if kv_caches and kv_caches[0] is not None:
            k, v = kv_caches[0]
            per_layer_bytes = k.nelement() * k.element_size() * 2  # K + V
            metrics.cache_memory_mb = (per_layer_bytes * len(kv_caches)) / (1024 * 1024)
        
        return output_ids, metrics


def main():
    """Quick test: generate text and print metrics."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate text with GPT-2")
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence is")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--no_cache", action="store_true", help="Disable KV cache (naive baseline)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    from model.weight_loader import create_model
    
    config = InferenceConfig(
        use_kv_cache=not args.no_cache,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        device=args.device,
    )
    
    print("Loading model...")
    model = create_model("gpt2", device=args.device)
    
    print(f"\nGenerating with {'KV Cache' if config.use_kv_cache else 'Naive (no cache)'}...")
    print(f"Prompt: \"{args.prompt}\"\n")
    
    generator = Generator(model, config)
    text, metrics = generator.generate(args.prompt)
    
    print(f"--- Generated Text ---")
    print(text)
    print(f"\n--- Metrics ---")
    print(metrics.summary())


if __name__ == "__main__":
    main()
