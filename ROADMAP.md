# Roadmap: LLM Optimization Research Testbed

## Vision
Transform this GPT-2 inference engine into a **research platform for testing LLM optimization papers** with:
- Easy plugin architecture for new techniques
- Comprehensive benchmarking (speed, memory, quality)
- MCP server for programmatic/agentic experimentation
- Reference implementations of key papers

---

## Phase 1: Foundation — Complete Current Work
**Goal:** Finish existing optimizations, establish correctness baseline

### 1.1 Fix Weight Loading & Correctness Tests
- [ ] Debug `weight_loader.py` Conv1D transpose logic
- [ ] Run `python -m tests.test_correctness` until all 3 tests pass:
  - Logits match HuggingFace (<1e-3 error)
  - Greedy generation matches HF exactly
  - KV-cached output matches naive mode
- [ ] Document any HF checkpoint quirks discovered

### 1.2 Complete KV Cache Implementation
- [ ] Verify pre-allocated cache works correctly
- [ ] Add memory tracking and utilization stats
- [ ] Test with various sequence lengths (128, 512, 1024)

### 1.3 Integrate Grouped-Query Attention (GQA)
- [ ] Wire `GroupedQueryAttention` into `TransformerBlock`
- [ ] Add conditional logic: if `config.use_gqa=True`, replace MHA with GQA
- [ ] Implement `from_pretrained_mha()` weight conversion (average K,V heads)
- [ ] Test correctness: outputs should match baseline (it's lossy, but deterministic)
- [ ] Measure: memory savings (cache size), speed impact, perplexity change

### 1.4 Integrate Speculative Decoding
- [ ] Add `_generate_with_speculative()` method to `base_generator.py`
- [ ] Create draft model with shared weights (`DraftModel.from_target_model()`)
- [ ] Implement draft/target KV cache management:
  - Draft cache resets each step
  - Target cache truncates on rejection
- [ ] Track acceptance rate and avg tokens accepted per step
- [ ] Test: outputs should match baseline distribution (non-deterministic but statistically correct)

**Deliverable:** Baseline system with 3 working optimizations (KV cache, GQA, speculative)

---

## Phase 2: Plugin Architecture — Make It Extensible
**Goal:** Extract pattern from existing optimizations, create reusable framework

### 2.1 Design Plugin Interface
```python
# optimizations/base.py
class OptimizationPlugin:
    """Base class for all optimization techniques."""

    name: str
    requires_model_modification: bool
    is_lossy: bool  # Does it change output distribution?

    def apply_to_model(self, model: GPT2Model) -> GPT2Model:
        """Modify model architecture (e.g., replace attention layers)."""

    def apply_to_generator(self, generator: Generator) -> Generator:
        """Modify generation logic (e.g., speculative decoding loop)."""

    def get_config_requirements(self) -> dict:
        """Return required config parameters."""

    def get_metrics(self) -> dict:
        """Return technique-specific metrics for reporting."""
```

### 2.2 Refactor Existing Optimizations as Plugins
- [ ] Create `KVCachePlugin`
- [ ] Create `GQAPlugin(num_kv_groups=4)`
- [ ] Create `SpeculativeDecodingPlugin(draft_n_layers=4, lookahead=5)`
- [ ] Update `config.py` to accept list of plugins:
  ```python
  config = InferenceConfig(
      optimizations=[
          KVCachePlugin(),
          GQAPlugin(num_kv_groups=4),
      ]
  )
  ```

### 2.3 Add Plugin Composition & Validation
- [ ] Check plugin compatibility (e.g., speculative requires KV cache)
- [ ] Apply plugins in correct order (model modifications → generation logic)
- [ ] Merge metrics from all active plugins

### 2.4 Create Plugin Template & Documentation
- [ ] `optimizations/TEMPLATE/` with boilerplate
- [ ] `docs/ADDING_NEW_OPTIMIZATION.md` — step-by-step guide
- [ ] Example: "How to add Flash Attention in 1 hour"

**Deliverable:** Clean plugin system, existing optimizations refactored, guide for adding new ones

---

## Phase 3: Comprehensive Benchmarking
**Goal:** Multi-dimensional measurement framework

### 3.1 Expand Metrics Collection
Current: tokens/sec, TTFT, memory
Add:
- [ ] **Latency percentiles** (p50, p90, p99)
- [ ] **Cache utilization** (% of allocated cache used)
- [ ] **Memory breakdown** (model weights, activations, KV cache)
- [ ] **Warmup vs steady-state** performance

### 3.2 Add Quality Measurement
- [ ] Implement perplexity measurement on WikiText-2
- [ ] Add BLEU/ROUGE for generation quality
- [ ] Exact match rate vs baseline (for deterministic optimizations)
- [ ] Track quality degradation from lossy optimizations

### 3.3 Automated Comparison & Ablation Studies
```python
# benchmarks/ablation.py
def ablation_study(
    base_plugins: list[OptimizationPlugin],
    ablations: dict[str, list[OptimizationPlugin]],
    test_prompts: list[str]
):
    """Run ablation study and generate comparison table."""
```
- [ ] Test all plugin combinations automatically
- [ ] Generate markdown tables + plots
- [ ] Export to JSON for programmatic access

### 3.4 Visualization Improvements
- [ ] Speedup bar charts (current)
- [ ] Memory usage breakdown (stacked bars)
- [ ] Quality vs speed Pareto frontier
- [ ] Timeline plots (latency over generation steps)

**Deliverable:** `python -m benchmarks.harness --ablation` produces comprehensive analysis

---

## Phase 4: MCP Server — Programmatic Access
**Goal:** Expose inference engine as MCP tools for agentic experimentation

### 4.1 Basic MCP Server
```bash
mcp_server/
├── __init__.py
├── server.py              # Main FastMCP server
├── tools/
│   ├── generation.py      # generate_with_optimization()
│   ├── benchmark.py       # benchmark_optimization()
│   └── comparison.py      # compare_optimizations()
└── README.md              # How to connect from Claude Desktop
```

**Tools to implement:**
- [ ] `generate_with_optimization(prompt, optimization, max_tokens)`
  - Returns: text + metrics (tokens/sec, memory, TTFT)
- [ ] `benchmark_optimization(optimization, num_trials=10)`
  - Runs full benchmark suite, returns JSON results
- [ ] `compare_optimizations(optimizations: list, test_prompt)`
  - Side-by-side comparison of multiple techniques
- [ ] `list_available_optimizations()`
  - Returns metadata about all registered plugins

### 4.2 Configuration & Testing
- [ ] Add MCP server config to `claude_desktop_config.json` example
- [ ] Test connection from Claude Desktop
- [ ] Add error handling and helpful messages
- [ ] Document tool schemas with examples

### 4.3 Advanced Research Tools
- [ ] `ablation_study(base, ablations: dict)`
  - Automated ablation analysis
- [ ] `profile_optimization(optimization, metric="latency")`
  - Deep profiling of specific technique
- [ ] `find_best_optimization(constraint="memory<2GB", objective="speed")`
  - Search optimization space with constraints

**Deliverable:** Working MCP server, agents can generate/benchmark via tool calls

---

## Phase 5: Reference Implementations — Prove It Works
**Goal:** Implement 3-4 high-impact papers as plugins

### 5.1 Flash Attention
- [ ] Paper: https://arxiv.org/abs/2205.14135
- [ ] Implement tiled attention (fused kernel via Triton or torch.compile)
- [ ] Test: memory usage should drop significantly for long sequences
- [ ] Benchmark: speed vs standard attention
- [ ] Document in `optimizations/flash_attention/PAPER.md`

### 5.2 Quantization (Int8)
- [ ] Paper: LLM.int8() https://arxiv.org/abs/2208.07339
- [ ] Implement weight-only int8 quantization
- [ ] Test: model size should drop 4x, quality degradation <5% perplexity
- [ ] Benchmark: speed/memory tradeoffs
- [ ] Compare to bitsandbytes for validation

### 5.3 Multi-Query Attention (MQA)
- [ ] Paper: https://arxiv.org/abs/1911.02150
- [ ] Implement as extreme case of GQA (1 KV head)
- [ ] Test: KV cache size should drop 12x
- [ ] Measure quality impact vs GQA

### 5.4 PagedAttention (optional, advanced)
- [ ] Paper: https://arxiv.org/abs/2309.06180 (vLLM)
- [ ] Non-contiguous KV cache with virtual memory
- [ ] Test: batching efficiency, memory fragmentation
- [ ] This is complex — only if time permits

**Deliverable:** 3 new optimizations with paper citations, correctness tests, benchmarks

---

## Phase 6: Research Platform Features
**Goal:** Tools for serious ML research

### 6.1 Multi-Model Support
- [ ] GPT-2 Medium (355M)
- [ ] GPT-2 Large (774M)
- [ ] GPT-J 6B (stretch goal)
- [ ] Test: do optimizations generalize across model sizes?

### 6.2 Custom Dataset Support
- [ ] WikiText-2 (perplexity)
- [ ] Tool-calling benchmark (JSON validity, function accuracy)
- [ ] Long-context benchmark (needle-in-haystack)
- [ ] Code generation (HumanEval subset)

### 6.3 Numerical Stability Analysis
- [ ] Track activation norms per layer
- [ ] Gradient flow visualization (if fine-tuning)
- [ ] Quantization error distribution
- [ ] Compare fp32 vs fp16 vs int8

### 6.4 Automated Paper Replication
```python
# experiments/replicate_paper.py
def replicate(paper_name: str):
    """
    Load paper config, run experiments, compare to reported numbers.
    Example: replicate("flash_attention") runs all benchmarks from the paper.
    """
```

**Deliverable:** Platform for rigorous optimization research

---

## Phase 7: Documentation & Polish
**Goal:** Make it usable by others

### 7.1 Comprehensive Docs
- [ ] `docs/ARCHITECTURE.md` — how the codebase is structured
- [ ] `docs/ADDING_NEW_OPTIMIZATION.md` — step-by-step tutorial
- [ ] `docs/MCP_TOOLS.md` — guide to using the MCP server
- [ ] `docs/BENCHMARKING.md` — interpreting results, methodology

### 7.2 Example Notebooks
- [ ] `notebooks/quickstart.ipynb` — generate text, run benchmarks
- [ ] `notebooks/add_optimization.ipynb` — walkthrough of adding new technique
- [ ] `notebooks/ablation_study.ipynb` — example research workflow

### 7.3 Demo & Deployment
- [ ] Gradio app with live metrics (`demo/app.py`)
- [ ] Deploy to HuggingFace Spaces
- [ ] Add "Try it live" link to README

### 7.4 Blog Post / Technical Writeup
- [ ] "Building an LLM Optimization Testbed from Scratch"
- [ ] Key insights from implementing Flash Attention, quantization, etc.
- [ ] Benchmark comparisons
- [ ] Publish on personal blog or Medium

**Deliverable:** Production-ready research platform

---

## Success Metrics

**By Phase 3:** Baseline + 3 optimizations working, comprehensive benchmarks
**By Phase 4:** MCP server functional, agents can experiment programmatically
**By Phase 5:** 3+ paper implementations with published results
**By Phase 7:** Usable by other researchers, documentation complete

---

## Timeline Estimate

| Phase | Description | Estimated Time |
|-------|-------------|----------------|
| Phase 1 | Complete current work | 1-2 weeks |
| Phase 2 | Plugin architecture | 1 week |
| Phase 3 | Benchmarking framework | 1 week |
| Phase 4 | MCP server | 3-5 days |
| Phase 5 | 3 reference implementations | 2-3 weeks |
| Phase 6 | Research features | 2-3 weeks |
| Phase 7 | Documentation & polish | 1 week |

**Total: ~8-12 weeks** for full platform

**MVP (Phases 1-4):** ~3-4 weeks — functional testbed with MCP integration

---

## What This Enables

1. **Fast paper replication** — New optimization in <1 day instead of weeks
2. **Apples-to-apples comparisons** — Same model, same hardware, same prompts
3. **Agentic research** — AI agents can explore optimization space autonomously
4. **Portfolio piece** — Demonstrates deep ML systems knowledge to employers
5. **Open source contribution** — Useful tool for the research community

---

## Next Steps

1. Read this roadmap
2. Decide which phases to prioritize
3. Start with Phase 1.1 (fix weight loading, pass correctness tests)
4. Work through systematically

Want to get started? Run:
```bash
python -m tests.test_correctness
```

Fix any failures, then we'll move to Phase 1.2.
