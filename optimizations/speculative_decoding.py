"""
Speculative Decoding — Leviathan et al., 2023
"Fast Inference from Transformers via Speculative Decoding"

Core idea:
    A small, fast "draft" model proposes K tokens autoregressively.
    The large "target" model scores ALL K tokens in a single forward pass.
    We accept/reject each proposed token using rejection sampling.
    
    Expected speedup: ~K × acceptance_rate
    Where acceptance_rate depends on how well the draft approximates the target.

Why it's LOSSLESS:
    The rejection sampling scheme guarantees that the output distribution
    is EXACTLY the same as standard autoregressive sampling from the target
    model alone. This is not an approximation — it's mathematically exact.
    
    Proof sketch:
    - If p_target(x) >= p_draft(x): always accept → same as target
    - If p_target(x) < p_draft(x): accept with prob p_target/p_draft
    - On rejection: sample from residual distribution max(0, p_target - p_draft)
    - This is classic rejection sampling from probability theory

Draft model strategy:
    We use a "truncated" version of GPT-2: same architecture but only
    the first N layers (e.g., 4 out of 12). This shares the same weights
    and tokenizer, so there's no conversion needed.
"""
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, List
from dataclasses import dataclass, field

from model.transformer import GPT2, TransformerBlock
from model.layernorm import LayerNorm
from model.embedding import GPT2Embedding
from generation.sampling import sample_token
from config import ModelConfig


@dataclass
class SpeculativeMetrics:
    """Track speculative decoding statistics."""
    total_proposed: int = 0
    total_accepted: int = 0
    total_steps: int = 0
    
    @property
    def acceptance_rate(self) -> float:
        if self.total_proposed == 0:
            return 0.0
        return self.total_accepted / self.total_proposed
    
    @property
    def avg_accepted_per_step(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.total_accepted / self.total_steps
    
    def summary(self) -> str:
        return (
            f"Spec decode: {self.acceptance_rate:.1%} acceptance rate, "
            f"{self.avg_accepted_per_step:.1f} tokens/step avg "
            f"({self.total_accepted}/{self.total_proposed} accepted)"
        )


class DraftModel(GPT2):
    """
    Draft model: first N layers of GPT-2.
    
    Uses the same weights as the target model for shared layers.
    Fewer layers = faster inference, but less accurate predictions.
    
    The key insight is that early layers capture most of the "easy"
    token predictions (common words, simple patterns). Later layers
    refine the distribution for harder predictions. So a 4-layer model
    can often predict the same token as a 12-layer model.
    """
    @classmethod
    def from_target_model(cls, target: GPT2, n_layers: int) -> "DraftModel":
        """
        Create a draft model by taking the first n_layers from the target.
        Shares weights — no additional memory for shared parameters.
        """
        config = ModelConfig(
            vocab_size=target.config.vocab_size,
            d_model=target.config.d_model,
            n_heads=target.config.n_heads,
            n_layers=n_layers,
            d_ff=target.config.d_ff,
            max_seq_len=target.config.max_seq_len,
        )
        
        draft = cls(config)
        
        # Share embedding weights
        draft.embedding = target.embedding
        
        # Copy first n_layers of transformer blocks
        for i in range(n_layers):
            draft.blocks[i] = target.blocks[i]
        
        # Share final layernorm and LM head
        draft.ln_f = target.ln_f
        draft.lm_head = target.lm_head
        
        draft.eval()
        return draft


class SpeculativeDecoder:
    """
    Speculative decoding engine.
    
    Algorithm per step:
        1. Draft model generates K tokens autoregressively (fast)
        2. Target model scores all K tokens in ONE forward pass (parallel)
        3. Accept/reject each token using rejection sampling
        4. On first rejection: sample corrected token from residual distribution
        5. Return all accepted tokens + corrected token
    
    This produces 1 to K+1 tokens per "step" instead of exactly 1,
    with the same distribution as standard autoregressive decoding.
    """
    def __init__(
        self,
        target_model: GPT2,
        draft_model: DraftModel,
        K: int = 5,                # Lookahead: propose K tokens
        temperature: float = 1.0,
    ):
        self.target = target_model
        self.draft = draft_model
        self.K = K
        self.temperature = temperature
        self.metrics = SpeculativeMetrics()
        self.device = next(target_model.parameters()).device
    
    @torch.no_grad()
    def speculative_step(
        self,
        input_ids: Tensor,             # [B, current_len]
        target_caches: Optional[List],  # KV caches for target model
        draft_caches: Optional[List],   # KV caches for draft model
    ) -> Tuple[Tensor, List, List]:
        """
        One speculative decoding step.
        
        Returns:
            new_tokens: [B, n_accepted] — between 1 and K+1 tokens
            updated target_caches
            updated draft_caches
        """
        B = input_ids.shape[0]
        assert B == 1, "Speculative decoding currently supports batch_size=1"
        
        # ============================================================
        # PHASE 1: Draft model proposes K tokens (autoregressive, fast)
        # ============================================================
        draft_tokens = []       # K proposed token ids
        draft_probs_list = []   # K probability distributions
        
        draft_input = input_ids[:, -1:]  # Start from last token
        
        for k in range(self.K):
            draft_logits, draft_caches = self.draft(
                draft_input, kv_caches=draft_caches, use_cache=True
            )
            
            draft_logits_last = draft_logits[:, -1, :]
            if self.temperature > 0:
                draft_logits_last = draft_logits_last / self.temperature
            
            draft_probs = F.softmax(draft_logits_last, dim=-1)  # [B, vocab]
            
            # Sample from draft distribution
            token = torch.multinomial(draft_probs, 1)  # [B, 1]
            
            draft_tokens.append(token)
            draft_probs_list.append(draft_probs)
            draft_input = token
        
        draft_token_ids = torch.cat(draft_tokens, dim=1)  # [B, K]
        
        # ============================================================
        # PHASE 2: Target model scores ALL K tokens at once (parallel)
        # ============================================================
        # Feed the last real token + all K draft tokens to the target
        # This is ONE forward pass instead of K sequential passes
        target_input = torch.cat([input_ids[:, -1:], draft_token_ids], dim=1)  # [B, K+1]
        
        target_logits, target_caches = self.target(
            target_input, kv_caches=target_caches, use_cache=True
        )
        # target_logits: [B, K+1, vocab]
        # Position 0: logits for token AFTER the last real token
        # Position k: logits for token AFTER draft_token[k-1]
        
        if self.temperature > 0:
            target_logits = target_logits / self.temperature
        
        # ============================================================
        # PHASE 3: Rejection sampling — accept/reject each draft token
        # ============================================================
        accepted_tokens = []
        n_accepted = 0
        
        for k in range(self.K):
            # Target's probability for this position
            p_target = F.softmax(target_logits[:, k, :], dim=-1)  # [B, vocab]
            
            # Draft's probability for this position
            p_draft = draft_probs_list[k]  # [B, vocab]
            
            # The proposed token
            token_k = draft_tokens[k]  # [B, 1]
            
            # Get probabilities for the specific proposed token
            p_t = p_target.gather(-1, token_k).squeeze(-1)  # [B]
            p_d = p_draft.gather(-1, token_k).squeeze(-1)   # [B]
            
            # Acceptance criterion:
            # Accept with probability min(1, p_target(x) / p_draft(x))
            #
            # Intuition:
            # - If target likes this token MORE than draft: always accept
            # - If target likes this token LESS than draft: accept proportionally
            accept_prob = torch.clamp(p_t / (p_d + 1e-10), max=1.0)
            
            # Stochastic acceptance
            r = torch.rand(B, device=self.device)
            
            if (r < accept_prob).all():
                # ACCEPT: this token matches target's preferences
                accepted_tokens.append(token_k)
                n_accepted += 1
            else:
                # REJECT: sample from the residual distribution
                # p_residual(x) = normalize(max(0, p_target(x) - p_draft(x)))
                #
                # This is the mathematical key to why speculative decoding
                # is LOSSLESS: the residual distribution corrects for any
                # mismatch between draft and target.
                residual = torch.clamp(p_target - p_draft, min=0)
                residual_sum = residual.sum(dim=-1, keepdim=True)
                
                if residual_sum.item() > 1e-10:
                    residual = residual / residual_sum
                    corrected_token = torch.multinomial(residual, 1)
                else:
                    # Fallback: sample from target distribution
                    corrected_token = torch.multinomial(p_target, 1)
                
                accepted_tokens.append(corrected_token)
                n_accepted += 0  # The corrected token doesn't count as "accepted"
                break  # Stop at first rejection
        else:
            # ALL K tokens accepted — sample one more from target
            # This is the "bonus" token from the last position's logits
            p_last = F.softmax(target_logits[:, self.K, :], dim=-1)
            bonus_token = torch.multinomial(p_last, 1)
            accepted_tokens.append(bonus_token)
        
        # Update metrics
        self.metrics.total_proposed += self.K
        self.metrics.total_accepted += n_accepted
        self.metrics.total_steps += 1
        
        # Concatenate all accepted (+ corrected/bonus) tokens
        new_tokens = torch.cat(accepted_tokens, dim=1)  # [B, 1..K+1]
        
        # IMPORTANT: We need to truncate the KV caches to only include
        # the tokens we actually accepted. The target processed K+1 tokens
        # but we may have only accepted fewer.
        n_to_keep = new_tokens.shape[1]
        n_to_remove = (self.K + 1) - n_to_keep
        
        if n_to_remove > 0 and target_caches is not None:
            # Trim the last n_to_remove positions from each layer's cache
            target_caches = [
                (k[:, :, :-n_to_remove, :], v[:, :, :-n_to_remove, :])
                if k is not None else None
                for k, v in target_caches
            ]
        
        # Also reset draft caches — draft will re-run from the new position
        draft_caches = None
        
        return new_tokens, target_caches, draft_caches
    
    def reset_metrics(self):
        self.metrics = SpeculativeMetrics()
