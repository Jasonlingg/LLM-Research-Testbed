"""
PyTorch Profiler wrapper — shows where time actually goes during inference.

Wraps torch.profiler.profile around a generation run and prints a formatted
table of the top operations by total CPU time, plus a chrome trace export
that can be opened at ui.perfetto.dev for full flame-graph analysis.

Usage:
    python -m benchmarks.profile [--prompt "..."] [--max_tokens 30] [--use_flash_attn]

Output:
    - Console table: op name, total ms, %, call count
    - benchmarks/results/trace.json  (open at ui.perfetto.dev)
    - Human insight: where is the bottleneck?
"""
import argparse
import os
import json
from pathlib import Path

import torch
import torch.profiler
import tiktoken

from config import InferenceConfig
from model.weight_loader import create_model
from generation.sampling import sample_token


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate(model, input_ids: torch.Tensor, config: InferenceConfig) -> torch.Tensor:
    """
    Minimal autoregressive loop — just enough to trigger the ops we want to profile.
    Mirrors base_generator.py but without the full metrics machinery so the profiler
    output isn't cluttered with bookkeeping.
    """
    device = input_ids.device
    generated = input_ids.clone()

    with torch.no_grad():
        # Prefill
        logits, kv_caches = model(
            generated,
            use_cache=config.use_kv_cache,
        )

        # Sample first token
        next_token = sample_token(
            logits[:, -1, :], config.temperature, config.top_k, config.top_p
        )
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        for _ in range(config.max_new_tokens - 1):
            if config.use_kv_cache and kv_caches is not None:
                logits, kv_caches = model(
                    next_token.unsqueeze(0),
                    kv_caches=kv_caches,
                    use_cache=True,
                )
            else:
                logits, _ = model(generated)

            next_token = sample_token(
                logits[:, -1, :], config.temperature, config.top_k, config.top_p
            )
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

    return generated


def _print_table(events, top_n: int = 20) -> dict:
    """
    Print a human-readable table of the top ops by total CPU time.

    Returns a dict of {op_name: total_ms} for the top ops, used for
    the insight summary below.
    """
    # Filter to CPU self time > 0
    rows = [
        (e.key, e.cpu_time_total / 1000, e.count)
        for e in events
        if e.cpu_time_total > 0
    ]
    rows.sort(key=lambda x: -x[1])

    total_ms = sum(r[1] for r in rows)

    col_w = [50, 12, 8, 8]
    header = (
        f"{'Operation':<{col_w[0]}}"
        f"{'Time (ms)':>{col_w[1]}}"
        f"{'%':>{col_w[2]}}"
        f"{'Calls':>{col_w[3]}}"
    )
    sep = "-" * sum(col_w)
    print(f"\n{sep}")
    print(header)
    print(sep)

    top_rows = rows[:top_n]
    for name, ms, calls in top_rows:
        pct = 100 * ms / total_ms if total_ms > 0 else 0
        # Truncate long op names
        short = name if len(name) <= col_w[0] - 1 else name[: col_w[0] - 4] + "..."
        print(
            f"{short:<{col_w[0]}}"
            f"{ms:>{col_w[1]}.2f}"
            f"{pct:>{col_w[2]}.1f}"
            f"{calls:>{col_w[3]}}"
        )

    if len(rows) > top_n:
        rest_ms = sum(r[1] for r in rows[top_n:])
        pct = 100 * rest_ms / total_ms if total_ms > 0 else 0
        print(
            f"{'  ... (other ops)':<{col_w[0]}}"
            f"{rest_ms:>{col_w[1]}.2f}"
            f"{pct:>{col_w[2]}.1f}"
            f"{'':>{col_w[3]}}"
        )

    print(sep)
    print(f"{'TOTAL':<{col_w[0]}}{total_ms:>{col_w[1]}.2f}\n")

    return {name: ms for name, ms, _ in top_rows}


def _print_insight(op_times: dict, total_ms: float) -> None:
    """
    Translate raw op times into a plain-English bottleneck analysis.
    This is the systems-thinking layer on top of the raw numbers.
    """
    print("=" * 60)
    print("BOTTLENECK ANALYSIS")
    print("=" * 60)

    # Categorise ops
    matmul_ms = sum(
        ms for op, ms in op_times.items()
        if any(k in op for k in ("aten::mm", "aten::bmm", "aten::addmm", "aten::matmul"))
    )
    linear_ms = sum(ms for op, ms in op_times.items() if "aten::linear" in op)
    softmax_ms = sum(ms for op, ms in op_times.items() if "softmax" in op)
    layernorm_ms = sum(ms for op, ms in op_times.items() if "layer_norm" in op)
    cat_ms = sum(ms for op, ms in op_times.items() if "aten::cat" in op)

    def pct(ms):
        return 100 * ms / total_ms if total_ms > 0 else 0

    lines = [
        ("matmul (mm/bmm/addmm)", matmul_ms),
        ("linear projections",    linear_ms),
        ("softmax",               softmax_ms),
        ("layer_norm (×24/step)", layernorm_ms),
        ("cat (KV cache concat)", cat_ms),
    ]

    for label, ms in lines:
        if ms > 0:
            print(f"  {label:<30} {ms:7.2f} ms  ({pct(ms):.1f}%)")

    # Generate human insight
    print()
    dominant = max(lines, key=lambda x: x[1])
    print(f"  → {dominant[1]:.1f} ms ({pct(dominant[1]):.1f}%) in '{dominant[0]}'"
          " — that's your primary bottleneck.")

    if cat_ms > 0.05 * total_ms:
        print("  → KV cache torch.cat is significant: consider pre-allocated cache "
              "to eliminate O(n) copy per step.")

    if layernorm_ms > 0.15 * total_ms:
        print("  → LayerNorm is a notable cost: 24 calls per generate step "
              "(2 per block × 12 blocks). Fused kernels (e.g. apex) help here.")

    if softmax_ms > 0.1 * total_ms:
        print("  → Softmax is significant; Flash Attention's tiled approach avoids "
              "the full score matrix but needs a Triton kernel for true IO savings.")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Profile GPT-2 inference with PyTorch profiler")
    parser.add_argument("--prompt", type=str, default="The history of artificial intelligence began",
                        help="Prompt to generate from")
    parser.add_argument("--max_tokens", type=int, default=30,
                        help="Tokens to generate (keep low for fast profiling)")
    parser.add_argument("--use_flash_attn", action="store_true",
                        help="Use Flash Attention instead of standard MHA")
    parser.add_argument("--no_kv_cache", action="store_true",
                        help="Disable KV cache (baseline mode)")
    parser.add_argument("--top_n", type=int, default=20,
                        help="Number of ops to show in table")
    args = parser.parse_args()

    config = InferenceConfig(
        use_kv_cache=not args.no_kv_cache,
        use_flash_attn=args.use_flash_attn,
        max_new_tokens=args.max_tokens,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
    )

    device = config.device
    print(f"\nConfig: {config.describe()}")
    print(f"Device: {device}")
    print(f"Prompt: {args.prompt!r}")
    print(f"Tokens: {args.max_tokens}\n")

    # Load model
    model = create_model("gpt2", device=device, config=config)
    model.eval()

    # Tokenize
    enc = tiktoken.get_encoding("gpt2")
    input_ids = torch.tensor(
        [enc.encode(args.prompt)], dtype=torch.long, device=device
    )

    # Warmup (not profiled)
    print("Warming up...")
    with torch.no_grad():
        _generate(model, input_ids, config)

    # Output directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    trace_path = str(results_dir / "trace.json")

    # Profile
    print("Profiling...")
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            _generate(model, input_ids, config)

    # Export chrome trace
    prof.export_chrome_trace(trace_path)
    print(f"Chrome trace saved → {trace_path}")
    print("  (Open at https://ui.perfetto.dev to explore the flame graph)\n")

    # Print table
    events = prof.key_averages()
    op_times = _print_table(events, top_n=args.top_n)

    # Bottleneck insight
    total_ms = sum(e.cpu_time_total / 1000 for e in events if e.cpu_time_total > 0)
    _print_insight(op_times, total_ms)


if __name__ == "__main__":
    main()
