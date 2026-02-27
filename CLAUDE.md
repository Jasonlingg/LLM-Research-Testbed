# CLAUDE.md

## What This Project Is

An LLM inference engine for GPT-2 124M built entirely from raw PyTorch operations. No `model.generate()`, no HuggingFace at runtime. The forward pass, attention, KV cache, generation loop, and every optimization is implemented from scratch. HuggingFace is only used at setup time to download pretrained weights.

The goal is to demonstrate deep understanding of transformer inference for ML research engineer roles — not just usage of libraries.

## Project Structure

```
inference-engine/
├── config.py                          # Central config — all settings and optimization toggles
├── model/                             # GPT-2 transformer from scratch
│   ├── attention.py                   # Multi-head attention from raw matmul (no nn.MultiheadAttention)
│   ├── mlp.py                         # Feed-forward with GELU (approximate, to match HF)
│   ├── layernorm.py                   # Layer norm with learned affine params
│   ├── embedding.py                   # Token + learned positional embeddings
│   ├── transformer.py                 # Full GPT-2: embedding → 12 blocks → ln_f → lm_head (weight-tied)
│   └── weight_loader.py              # Maps HF checkpoint keys → our module names, handles Conv1D transpose
├── generation/                        # Inference engine
│   ├── base_generator.py             # Autoregressive loop (naive and KV-cached modes) + metrics collection
│   ├── kv_cache.py                   # Pre-allocated cache with memory tracking and utilization stats
│   └── sampling.py                   # Temperature, top-k, top-p (nucleus) sampling
├── optimizations/                     # Standalone optimization modules
│   ├── grouped_query_attention.py    # GQA with from_pretrained_mha() weight conversion
│   └── speculative_decoding.py       # Draft-verify with rejection sampling, acceptance rate tracking
├── benchmarks/                        # Measurement framework
│   ├── harness.py                    # Unified benchmark runner: warmup → measure → report
│   ├── visualize.py                  # Generates plots from benchmark JSON results
│   └── results/                      # Auto-generated JSON + PNGs
├── tests/
│   └── test_correctness.py           # Three tests: logits match HF, greedy match, cached matches naive
├── demo/
│   └── app.py                        # Gradio interface with optimization toggles and live metrics
├── requirements.txt
└── README.md
```

## How the Code Flows

### Model Architecture (model/)

GPT-2 uses **pre-norm** architecture: LayerNorm comes before attention and MLP, not after.

```
input_ids → Embedding (token + positional)
          → [LayerNorm → MultiHeadAttention → + residual → LayerNorm → MLP → + residual] × 12
          → LayerNorm (final)
          → LM Head (linear, weight-tied with token embedding)
          → logits [batch, seq_len, 50257]
```

Key details:
- `attention.py` uses a combined `c_attn` projection for Q, K, V (single Linear → split), matching GPT-2's convention. Causal masking via `torch.triu` + `masked_fill(-inf)`.
- `mlp.py` uses the **approximate** GELU (tanh version), not `F.gelu`, to match HF outputs exactly.
- `weight_loader.py` handles the Conv1D → Linear transpose: HF's GPT-2 stores weights as `[in_features, out_features]`, our Linear expects `[out_features, in_features]`.
- The LM head shares weights with the token embedding (`self.lm_head.weight = self.embedding.token_embedding.weight`).

### Generation (generation/)

Two modes controlled by `InferenceConfig.use_kv_cache`:

**Naive mode** (`use_kv_cache=False`): Feeds the entire sequence through the model every step. O(n²) total compute. This is the baseline.

**Cached mode** (`use_kv_cache=True`): Two phases:
1. **Prefill** — process entire prompt in one pass, cache K,V for all 12 layers
2. **Decode** — feed only the new token each step, reuse cached K,V

The KV cache in `kv_cache.py` is pre-allocated as a contiguous tensor `[n_layers, 2, batch, n_heads, max_seq_len, d_head]`. Position tracking via `current_len` avoids torch.cat per step.

Position IDs during cached decode start from `past_len` (handled in `transformer.py` forward method).

### Optimizations (optimizations/)

Each optimization is a standalone module. They are not yet wired into the main generation pipeline — that's the builder's job (see "What Still Needs Work" below).

**Grouped-Query Attention** (`grouped_query_attention.py`):
- Reduces KV heads from 12 to `n_kv_groups` (default 4)
- Query projection stays full size; K,V projections shrink
- `repeat_interleave` expands KV heads to match query heads during attention
- `from_pretrained_mha()` converts standard MHA weights by averaging K,V projections within each group
- Caches the **reduced** K,V (not expanded), so cache memory shrinks by `n_heads / n_kv_groups`

**Speculative Decoding** (`speculative_decoding.py`):
- `DraftModel` — first N layers of GPT-2, shares weights with target (no extra memory for shared params)
- `speculative_step()` runs three phases: draft proposes K tokens → target scores all K in one forward pass → rejection sampling accepts/rejects
- Acceptance criterion: `min(1, p_target(x) / p_draft(x))` per token
- On rejection: sample from residual distribution `normalize(max(0, p_target - p_draft))`
- After acceptance/rejection, KV caches must be truncated to only include accepted tokens
- Tracks `acceptance_rate` and `avg_accepted_per_step` for diagnostics

### Configuration (config.py)

Everything is controlled through `InferenceConfig`:

```python
config = InferenceConfig(
    use_kv_cache=True,        # Toggle KV caching
    use_gqa=False,            # Toggle grouped-query attention
    gqa_num_kv_groups=4,      # 12 heads → 4 KV groups
    use_speculative=False,    # Toggle speculative decoding
    spec_draft_n_layers=4,    # Draft model uses first 4 layers
    spec_lookahead=5,         # Propose 5 tokens per speculative step
    max_new_tokens=256,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
)
```

`BENCHMARK_CONFIGS` in config.py defines the standard comparison set: baseline, kv_cache, kv_cache_gqa, kv_cache_spec, all.

### Benchmarking (benchmarks/)

`harness.py` runs each config through the same pipeline:
1. Warmup runs (not counted)
2. Multiple trials across 5 diverse test prompts
3. Collects: tokens/sec, TTFT, avg step latency, memory
4. Outputs comparison table (markdown) + raw JSON

`visualize.py` reads the JSON and generates 4 plot PNGs: speedup bars, TTFT comparison, step latency, speedup factors.

### Tests (tests/)

`test_correctness.py` has three tests:
1. **Logits match** — our model's logits match HF's to <1e-3 max difference
2. **Greedy match** — greedy generation produces identical text
3. **Cache match** — KV-cached generation produces same output as naive

All three must pass before benchmarks are meaningful.

## What Still Needs Work

The scaffold is complete but the following needs to be wired up and debugged:

1. **Weight loading correctness** — The Conv1D transpose logic in `weight_loader.py` may need debugging. Run `test_correctness.py` first and fix until logits match. Common issue: the transpose condition might not catch all cases. Compare shapes carefully.

2. **GQA integration into transformer blocks** — `GroupedQueryAttention` exists as a standalone module but isn't plugged into `TransformerBlock` yet. Need to add logic in `transformer.py` (or a new file) that replaces `MultiHeadAttention` with `GroupedQueryAttention` when `config.use_gqa=True`, using `from_pretrained_mha()` to convert weights.

3. **Speculative decoding integration into generator** — `SpeculativeDecoder` has the algorithm but isn't called from `base_generator.py`. Need to add a `_generate_with_speculative()` method that creates the draft model, runs `speculative_step()` in a loop, and handles the token accumulation and cache management.

4. **Draft model KV cache handling** — Speculative decoding needs separate KV caches for draft and target models. The draft cache should be reset after each speculative step (since we don't know which tokens were accepted). The target cache needs truncation.

5. **Benchmark configs for GQA and speculative** — The config toggles exist but the generator doesn't route to the right code paths yet. Add branching in `Generator.generate()` based on config flags.

6. **Quality measurement** — `harness.py` has a placeholder for perplexity measurement on WikiText-2. Implement this to quantify quality degradation from GQA and other lossy optimizations.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run correctness tests (DO THIS FIRST)
python -m tests.test_correctness

# Generate text (with KV cache)
python -m generation.base_generator --prompt "The future of AI is" --max_tokens 100

# Generate text (naive baseline, for comparison)
python -m generation.base_generator --prompt "The future of AI is" --max_tokens 50 --no_cache

# Run full benchmark suite
python -m benchmarks.harness

# Generate plots from benchmark results
python benchmarks/visualize.py

# Launch Gradio demo
python -m demo.app
```

## Key Technical Decisions

- **tiktoken for tokenization** instead of HF tokenizer — lightweight, no HF runtime dependency.
- **Pre-allocated KV cache** instead of torch.cat per step — O(1) per step vs O(n) for cat.
- **Combined c_attn projection** matching GPT-2's convention — one `[d_model, 3*d_model]` linear instead of three separate Q/K/V projections. This matters for weight loading correctness.
- **Approximate GELU** — GPT-2 uses the tanh approximation, not the exact erf version. Using the wrong one causes logit mismatches.
- **Weight tying** — lm_head shares weight tensor with token embedding. `num_parameters` property accounts for this.
- **Speculative draft model shares layers** — `DraftModel.from_target_model()` points to the same `nn.Module` objects, not copies. No extra memory for shared parameters.

## Style and Conventions

- Every file has a module-level docstring explaining what it does and why.
- Every class has detailed docstrings with shapes annotated in comments (e.g., `# [B, n_heads, T, d_head]`).
- GPT-2-specific naming (c_attn, c_proj, c_fc) matches HuggingFace checkpoint keys.
- Metrics are always collected alongside generation — never run inference without measuring.
- Tests compare against HuggingFace as ground truth.