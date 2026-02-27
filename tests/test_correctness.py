"""
Correctness test: verify our implementation matches HuggingFace exactly.

This is the MOST IMPORTANT test. If this fails, every benchmark result
is meaningless. We compare:
    1. Raw logits (should match to float precision)
    2. Generated sequences with greedy decoding (should be identical)
    3. KV-cached generation matches non-cached generation

Run: python -m tests.test_correctness
"""
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_logits_match():
    """
    Test 1: Our model's logits match HuggingFace's for the same input.
    
    This is the ground truth test. If logits match, our attention math,
    layer norm, MLP, and weight loading are all correct.
    """
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from model.weight_loader import create_model
    
    print("=" * 60)
    print("TEST 1: Logits match HuggingFace")
    print("=" * 60)
    
    device = "cpu"  # Use CPU for deterministic comparison
    
    # Load both models
    print("  Loading HuggingFace model...")
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(device)
    hf_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    print("  Loading our model...")
    our_model = create_model("gpt2", device=device)
    
    # Test prompts
    prompts = [
        "Hello, world!",
        "The quick brown fox",
        "Machine learning is",
        "In the beginning, there was",
    ]
    
    all_passed = True
    for prompt in prompts:
        # Tokenize
        inputs = hf_tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        
        # Get logits from both models
        with torch.no_grad():
            hf_logits = hf_model(input_ids).logits
            our_logits, _ = our_model(input_ids)
        
        # Compare
        max_diff = (hf_logits - our_logits).abs().max().item()
        mean_diff = (hf_logits - our_logits).abs().mean().item()
        
        passed = max_diff < 1e-3  # Allow small float precision differences
        status = "✓ PASS" if passed else "✗ FAIL"
        
        print(f"  {status} | \"{prompt}\"")
        print(f"         max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        
        if not passed:
            all_passed = False
    
    return all_passed


def test_greedy_generation_match():
    """
    Test 2: Greedy generation produces identical sequences.
    
    If logits match, greedy decoding (always pick argmax) must also match.
    This is an end-to-end test of the full generation pipeline.
    """
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from model.weight_loader import create_model
    from generation.base_generator import Generator
    from config import InferenceConfig
    
    print("\n" + "=" * 60)
    print("TEST 2: Greedy generation matches HuggingFace")
    print("=" * 60)
    
    device = "cpu"
    max_tokens = 20
    
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(device)
    hf_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    our_model = create_model("gpt2", device=device)
    config = InferenceConfig(
        use_kv_cache=False,  # Test without cache first
        max_new_tokens=max_tokens,
        temperature=0.0,  # Greedy
        device=device,
    )
    generator = Generator(our_model, config)
    
    prompt = "The meaning of life is"
    
    # HuggingFace generation
    hf_inputs = hf_tokenizer(prompt, return_tensors="pt").to(device)
    hf_output = hf_model.generate(
        hf_inputs["input_ids"],
        max_new_tokens=max_tokens,
        do_sample=False,  # Greedy
    )
    hf_text = hf_tokenizer.decode(hf_output[0], skip_special_tokens=True)
    
    # Our generation
    our_text, metrics = generator.generate(prompt, max_new_tokens=max_tokens, temperature=0.0)
    
    match = hf_text.strip() == our_text.strip()
    status = "✓ PASS" if match else "✗ FAIL"
    
    print(f"  {status}")
    print(f"  HF:   \"{hf_text}\"")
    print(f"  Ours: \"{our_text}\"")
    
    if not match:
        print(f"  (Mismatch — debug by comparing token-by-token)")
    
    return match


def test_cached_matches_naive():
    """
    Test 3: KV-cached generation produces the same output as naive generation.
    
    This verifies our KV cache implementation is correct. If the cache
    introduces any errors, this test catches it.
    """
    from model.weight_loader import create_model
    from generation.base_generator import Generator
    from config import InferenceConfig
    
    print("\n" + "=" * 60)
    print("TEST 3: Cached generation matches naive generation")
    print("=" * 60)
    
    device = "cpu"
    max_tokens = 30
    
    our_model = create_model("gpt2", device=device)
    
    # Naive (no cache)
    naive_config = InferenceConfig(
        use_kv_cache=False, max_new_tokens=max_tokens, temperature=0.0, device=device,
    )
    naive_gen = Generator(our_model, naive_config)
    
    # Cached
    cached_config = InferenceConfig(
        use_kv_cache=True, max_new_tokens=max_tokens, temperature=0.0, device=device,
    )
    cached_gen = Generator(our_model, cached_config)
    
    prompt = "Artificial intelligence will"
    
    naive_text, naive_metrics = naive_gen.generate(prompt, temperature=0.0)
    cached_text, cached_metrics = cached_gen.generate(prompt, temperature=0.0)
    
    match = naive_text.strip() == cached_text.strip()
    status = "✓ PASS" if match else "✗ FAIL"
    
    print(f"  {status}")
    print(f"  Naive:  \"{naive_text[:80]}...\"")
    print(f"  Cached: \"{cached_text[:80]}...\"")
    print(f"  Naive:  {naive_metrics.tokens_per_sec:.1f} tok/s")
    print(f"  Cached: {cached_metrics.tokens_per_sec:.1f} tok/s")
    
    if match:
        speedup = cached_metrics.tokens_per_sec / max(naive_metrics.tokens_per_sec, 0.01)
        print(f"  Speedup: {speedup:.1f}x")
    
    return match


def main():
    print("=" * 60)
    print("CORRECTNESS TESTS — Verifying our implementation")
    print("=" * 60)
    print()
    
    results = {}
    
    results["logits_match"] = test_logits_match()
    results["greedy_match"] = test_greedy_generation_match()
    results["cache_match"] = test_cached_matches_naive()
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    all_pass = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} {name}")
        if not passed:
            all_pass = False
    
    if all_pass:
        print("\n  All tests passed! Your implementation is correct. ✓")
        print("  You can now trust your benchmark results.")
    else:
        print("\n  Some tests failed. Fix these before running benchmarks.")
        print("  Benchmark results are meaningless if correctness fails.")
    
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
