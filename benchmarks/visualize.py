"""
Generate benchmark visualizations from results JSON.

Produces four key plots:
1. Speedup comparison (bar chart)
2. Memory-quality tradeoff (scatter)
3. Latency breakdown (stacked bar)
4. Summary radar chart

Run: python benchmarks/visualize.py
"""
import json
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11

COLORS = {
    'baseline': '#6B7280',
    'kv_cache': '#3B82F6',
    'kv_cache_gqa': '#8B5CF6',
    'kv_cache_spec': '#F59E0B',
    'all': '#10B981',
}


def load_results(path: str = "benchmarks/results/benchmark_results.json") -> dict:
    with open(path) as f:
        return json.load(f)


def plot_speedup_comparison(results: dict, output_dir: str):
    """Bar chart: tokens/sec for each configuration."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    names = list(results.keys())
    tps = [results[n]['mean_tokens_per_sec'] for n in names]
    labels = [results[n]['config_desc'] for n in names]
    colors = [COLORS.get(n, '#6B7280') for n in names]
    
    bars = ax.bar(range(len(names)), tps, color=colors, edgecolor='white', linewidth=0.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, tps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Baseline reference line
    if tps:
        ax.axhline(y=tps[0], color='red', linestyle='--', alpha=0.5, label='Baseline')
    
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Tokens per Second')
    ax.set_title('Inference Throughput by Configuration', fontweight='bold')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved speedup_comparison.png")


def plot_ttft_comparison(results: dict, output_dir: str):
    """Bar chart: time to first token for each configuration."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    names = list(results.keys())
    ttft = [results[n]['mean_ttft_ms'] for n in names]
    labels = [results[n]['config_desc'] for n in names]
    colors = [COLORS.get(n, '#6B7280') for n in names]
    
    bars = ax.bar(range(len(names)), ttft, color=colors, edgecolor='white', linewidth=0.5)
    
    for bar, val in zip(bars, ttft):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}ms', ha='center', va='bottom', fontsize=10)
    
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Time to First Token (ms)')
    ax.set_title('TTFT by Configuration', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ttft_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved ttft_comparison.png")


def plot_step_latency(results: dict, output_dir: str):
    """Bar chart: average per-step latency."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    names = list(results.keys())
    step_ms = [results[n]['mean_step_ms'] for n in names]
    labels = [results[n]['config_desc'] for n in names]
    colors = [COLORS.get(n, '#6B7280') for n in names]
    
    bars = ax.bar(range(len(names)), step_ms, color=colors, edgecolor='white', linewidth=0.5)
    
    for bar, val in zip(bars, step_ms):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}ms', ha='center', va='bottom', fontsize=10)
    
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Avg Step Latency (ms)')
    ax.set_title('Per-Token Generation Latency', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'step_latency.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved step_latency.png")


def plot_speedup_factors(results: dict, output_dir: str):
    """Horizontal bar chart showing speedup factors vs baseline."""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    names = list(results.keys())
    tps = [results[n]['mean_tokens_per_sec'] for n in names]
    labels = [results[n]['config_desc'] for n in names]
    colors = [COLORS.get(n, '#6B7280') for n in names]
    
    baseline = tps[0] if tps[0] > 0 else 1
    speedups = [t / baseline for t in tps]
    
    # Reverse for horizontal bar (top = first)
    bars = ax.barh(range(len(names)-1, -1, -1), speedups, color=colors, edgecolor='white')
    
    for bar, val in zip(bars, reversed(speedups)):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}x', va='center', fontweight='bold', fontsize=11)
    
    ax.set_yticks(range(len(names)-1, -1, -1))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Speedup vs Baseline')
    ax.set_title('Speedup Factor by Configuration', fontweight='bold')
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup_factors.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved speedup_factors.png")


def main():
    output_dir = "benchmarks/results"
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, "benchmark_results.json")
    
    if not os.path.exists(results_path):
        print("No benchmark results found. Run `python -m benchmarks.harness` first.")
        return
    
    results = load_results(results_path)
    
    print("Generating visualizations...")
    plot_speedup_comparison(results, output_dir)
    plot_ttft_comparison(results, output_dir)
    plot_step_latency(results, output_dir)
    plot_speedup_factors(results, output_dir)
    
    print(f"\nAll plots saved to {output_dir}/")
    print("Add these to your README and blog post!")


if __name__ == "__main__":
    main()
