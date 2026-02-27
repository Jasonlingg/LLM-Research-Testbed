"""
Benchmark harness: measure and compare all configurations fairly.

This is what turns your implementations into EVIDENCE.
Every config goes through the same pipeline: warmup → measure → report.

Run: python -m benchmarks.harness
"""
import time
import json
import torch
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.weight_loader import create_model
from generation.base_generator import Generator, GenerationMetrics
from config import InferenceConfig, BENCHMARK_CONFIGS


@dataclass
class BenchmarkResult:
    """Aggregated results for one configuration."""
    config_name: str
    config_desc: str
    mean_tokens_per_sec: float
    std_tokens_per_sec: float
    mean_ttft_ms: float
    mean_step_ms: float
    total_tokens: int
    num_trials: int


# Standard test prompts (diverse lengths and styles)
TEST_PROMPTS = [
    "The",
    "Once upon a time in a land far away",
    "In recent years, artificial intelligence has",
    "The fundamental theorem of calculus states that",
    "def fibonacci(n):\n    \"\"\"Calculate the nth Fibonacci number.\"\"\"\n    if n <=",
]


def run_benchmark(
    model,
    config: InferenceConfig,
    config_name: str,
    prompts: List[str] = None,
    max_new_tokens: int = 50,
    num_warmup: int = 1,
    num_trials: int = 3,
) -> BenchmarkResult:
    """
    Benchmark a single configuration with warmup and multiple trials.
    
    Protocol:
        1. Warmup runs (not counted) — JIT compilation, cache warming
        2. Multiple trials — for statistical reliability
        3. Report mean ± std
    """
    prompts = prompts or TEST_PROMPTS
    config.max_new_tokens = max_new_tokens
    
    generator = Generator(model, config)
    
    # Warmup
    print(f"  Warming up ({num_warmup} runs)...")
    for _ in range(num_warmup):
        generator.generate(prompts[0], max_new_tokens=10, temperature=0.0)
    
    # Measure
    all_metrics: List[GenerationMetrics] = []
    
    for trial in range(num_trials):
        for prompt in prompts:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            
            _, metrics = generator.generate(prompt, temperature=0.0)
            all_metrics.append(metrics)
    
    # Aggregate
    tok_per_sec = [m.tokens_per_sec for m in all_metrics]
    ttft = [m.time_to_first_token_ms for m in all_metrics]
    step_ms = [m.avg_step_time_ms for m in all_metrics]
    
    mean_tps = sum(tok_per_sec) / len(tok_per_sec)
    std_tps = (sum((x - mean_tps) ** 2 for x in tok_per_sec) / len(tok_per_sec)) ** 0.5
    
    result = BenchmarkResult(
        config_name=config_name,
        config_desc=config.describe(),
        mean_tokens_per_sec=round(mean_tps, 1),
        std_tokens_per_sec=round(std_tps, 1),
        mean_ttft_ms=round(sum(ttft) / len(ttft), 1),
        mean_step_ms=round(sum(step_ms) / len(step_ms), 1),
        total_tokens=sum(m.tokens_generated for m in all_metrics),
        num_trials=len(all_metrics),
    )
    
    return result


def run_full_comparison(device: str = "cpu") -> Dict[str, BenchmarkResult]:
    """Run all benchmark configurations and produce comparison."""
    
    print("=" * 60)
    print("BENCHMARK SUITE — LLM Inference Engine")
    print("=" * 60)
    print(f"Device: {device}")
    print()
    
    # Load model once
    print("Loading model...")
    model = create_model("gpt2", device=device)
    print()
    
    results = {}
    
    for name, config in BENCHMARK_CONFIGS.items():
        config.device = device
        print(f"[{name}] {config.describe()}")
        
        try:
            result = run_benchmark(model, config, name)
            results[name] = result
            print(f"  → {result.mean_tokens_per_sec} ± {result.std_tokens_per_sec} tok/s, "
                  f"TTFT: {result.mean_ttft_ms}ms")
        except Exception as e:
            print(f"  → FAILED: {e}")
        
        print()
    
    return results


def print_comparison_table(results: Dict[str, BenchmarkResult]):
    """Print a markdown-formatted comparison table."""
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    
    # Header
    print(f"| {'Configuration':<25} | {'Tok/s':>10} | {'TTFT (ms)':>10} | {'Step (ms)':>10} |")
    print(f"|{'-'*27}|{'-'*12}|{'-'*12}|{'-'*12}|")
    
    baseline_tps = None
    for name, result in results.items():
        if baseline_tps is None:
            baseline_tps = result.mean_tokens_per_sec
        
        speedup = result.mean_tokens_per_sec / baseline_tps if baseline_tps > 0 else 0
        tps_str = f"{result.mean_tokens_per_sec} ({speedup:.1f}x)"
        
        print(f"| {result.config_desc:<25} | {tps_str:>10} | {result.mean_ttft_ms:>10} | {result.mean_step_ms:>10} |")
    
    print()


def save_results(results: Dict[str, BenchmarkResult], output_dir: str = "benchmarks/results"):
    """Save results as JSON for reproducibility and plotting."""
    os.makedirs(output_dir, exist_ok=True)
    
    data = {name: asdict(result) for name, result in results.items()}
    
    output_path = os.path.join(output_dir, "benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Results saved to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference benchmarks")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    
    results = run_full_comparison(device=args.device)
    print_comparison_table(results)
    save_results(results)
    
    print("\nDone! Check benchmarks/results/ for raw data.")
    print("Next: run `python benchmarks/visualize.py` to generate plots.")


if __name__ == "__main__":
    main()
