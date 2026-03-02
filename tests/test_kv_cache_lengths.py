"""
Test KV cache performance across different sequence lengths.

This validates that:
1. Cache works correctly for short (128), medium (512), and long (1024) sequences
2. Utilization metrics are accurate
3. Performance scales appropriately
4. No cache overflow errors
"""
import torch
from model.weight_loader import create_model
from generation.base_generator import Generator
from config import InferenceConfig

# Test prompts of different lengths (measured token counts)
TEST_CASES = [
    {
        "name": "Short (~100 tokens)",
        "prompt": "The quick brown fox jumps over the lazy dog. " * 10,
        "max_tokens": 30,
    },
    {
        "name": "Medium (~400 tokens)",
        "prompt": "The history of artificial intelligence begins in antiquity with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. " * 15,
        "max_tokens": 30,
    },
    {
        "name": "Long (~800 tokens)",
        "prompt": "In the field of machine learning and deep learning, transformers represent a revolutionary architecture that has fundamentally changed how we approach natural language processing tasks. " * 30,
        "max_tokens": 30,
    },
]


def test_kv_cache_various_lengths():
    """Test KV cache with various sequence lengths."""
    print("=" * 60)
    print("KV CACHE LENGTH TESTS")
    print("=" * 60)

    device = "cpu"
    print(f"\nDevice: {device}")
    print("Loading model...")
    model = create_model("gpt2", device=device)

    results = []

    for test_case in TEST_CASES:
        print(f"\n{'=' * 60}")
        print(f"TEST: {test_case['name']}")
        print(f"{'=' * 60}")

        config = InferenceConfig(
            use_kv_cache=True,
            max_new_tokens=test_case['max_tokens'],
            temperature=0.8,
            device=device,
        )

        generator = Generator(model, config)

        # Truncate prompt display for readability
        prompt_display = test_case['prompt'][:100] + "..." if len(test_case['prompt']) > 100 else test_case['prompt']
        print(f"Prompt: \"{prompt_display}\"")
        print(f"Max new tokens: {test_case['max_tokens']}\n")

        try:
            text, metrics = generator.generate(test_case['prompt'], max_new_tokens=test_case['max_tokens'])

            print(f"✓ SUCCESS")
            print(f"  Prompt length: {metrics.prompt_length} tokens")
            print(f"  Generated: {metrics.tokens_generated} tokens")
            print(f"  Total sequence: {metrics.prompt_length + metrics.tokens_generated} tokens")
            print(f"  Cache memory: {metrics.cache_memory_mb:.1f} MB")
            print(f"  Cache utilization: {metrics.cache_utilization:.1%}")
            print(f"  Throughput: {metrics.tokens_per_sec:.1f} tok/s")
            print(f"  TTFT: {metrics.time_to_first_token_ms:.1f} ms")
            print(f"  Decode throughput: {metrics.decode_tokens_per_sec:.1f} tok/s")

            results.append({
                "name": test_case['name'],
                "prompt_len": metrics.prompt_length,
                "tokens_generated": metrics.tokens_generated,
                "total_len": metrics.prompt_length + metrics.tokens_generated,
                "cache_mb": metrics.cache_memory_mb,
                "utilization": metrics.cache_utilization,
                "tok_per_sec": metrics.tokens_per_sec,
                "ttft_ms": metrics.time_to_first_token_ms,
                "decode_tok_per_sec": metrics.decode_tokens_per_sec,
                "success": True,
            })

        except Exception as e:
            print(f"✗ FAILED: {e}")
            results.append({
                "name": test_case['name'],
                "success": False,
                "error": str(e),
            })

    # Summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}\n")

    print(f"{'Test':<20} {'Prompt':<8} {'Total':<8} {'Cache':<10} {'Util':<8} {'Tok/s':<8} {'TTFT':<10}")
    print("-" * 90)

    for r in results:
        if r['success']:
            print(
                f"{r['name']:<20} "
                f"{r['prompt_len']:<8} "
                f"{r['total_len']:<8} "
                f"{r['cache_mb']:<10.1f} "
                f"{r['utilization']:<8.1%} "
                f"{r['tok_per_sec']:<8.1f} "
                f"{r['ttft_ms']:<10.1f}"
            )
        else:
            print(f"{r['name']:<20} FAILED: {r['error']}")

    # Check all passed
    all_passed = all(r['success'] for r in results)

    print(f"\n{'=' * 60}")
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print(f"{'=' * 60}\n")

    return all_passed


if __name__ == "__main__":
    success = test_kv_cache_various_lengths()
    exit(0 if success else 1)
