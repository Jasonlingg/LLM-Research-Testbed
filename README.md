# LLM Inference Engine — From Scratch

Fast, optimized inference for GPT-2 124M built entirely from raw PyTorch operations.  
No `model.generate()`. No HuggingFace at runtime.

## Results

| Configuration          | Tokens/sec | Memory (MB) | TTFT (ms) | Perplexity (WikiText-2) |
|------------------------|-----------|-------------|-----------|------------------------|
| Baseline (naive)       | —         | —           | —         | —                      |
| + KV Cache             | —         | —           | —         | —                      |
| + Speculative Decoding | —         | —           | —         | —                      |
| + GQA (12→4 heads)     | —         | —           | —         | —                      |
| All Combined           | —         | —           | —         | —                      |

> Fill in after running `python -m benchmarks.harness`

## Live Demo
[Try it on HuggingFace Spaces](#) *(deploy after completion)*

## Architecture

```
inference-engine/
├── model/                    # Transformer from scratch
│   ├── attention.py          # Raw multi-head attention (no nn.MultiheadAttention)
│   ├── mlp.py                # Feed-forward network
│   ├── layernorm.py          # Layer normalization
│   ├── transformer.py        # Full GPT-2 architecture
│   ├── embedding.py          # Token + positional embeddings
│   └── weight_loader.py      # Load HF pretrained weights into our modules
├── generation/               # Inference engine
│   ├── base_generator.py     # Autoregressive generation loop
│   ├── kv_cache.py           # Pre-allocated cache with memory tracking
│   └── sampling.py           # Top-k, top-p, temperature sampling
├── optimizations/            # Each optimization is a standalone module
│   ├── grouped_query_attention.py  # GQA with pretrained weight conversion
│   └── speculative_decoding.py     # Draft-verify with rejection sampling
├── benchmarks/               # Measurement framework
│   ├── harness.py            # Unified benchmark runner
│   ├── metrics.py            # Metric collection and reporting
│   └── results/              # Auto-generated tables and plots
├── tests/                    # Correctness tests
│   ├── test_correctness.py   # Outputs match HuggingFace exactly
│   └── test_speculative.py   # Speculative decoding distribution test
├── demo/
│   └── app.py                # Gradio interactive demo
├── config.py                 # All configuration in one place
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt

# Verify correctness against HuggingFace
python -m tests.test_correctness

# Generate text
python -m generation.base_generator --prompt "The future of AI is" --max_tokens 100

# Run benchmarks
python -m benchmarks.harness

# Launch demo
python -m demo.app
```

## Key Design Decisions

1. **From scratch, but not from random weights.** I load pretrained GPT-2 weights from HuggingFace, but the forward pass, attention computation, KV cache, and generation loop are all my code. This proves I understand the architecture while still producing meaningful text.

2. **Composable optimizations.** Each optimization is a toggle in the config. The benchmark harness tests every combination through the same pipeline. This lets me produce apples-to-apples comparisons.

3. **Correctness first.** Before any optimization, `test_correctness.py` verifies my implementation produces identical logits to HuggingFace's GPT-2. If the baseline is wrong, every optimization result is meaningless.


