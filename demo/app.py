"""
Interactive Gradio demo: toggle optimizations and see metrics in real time.

Deploy to HuggingFace Spaces for a live URL anyone can try.

Run locally: python -m demo.app
"""
import time
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from model.weight_loader import create_model
from generation.base_generator import Generator
from config import InferenceConfig


# Load model once at startup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {DEVICE}...")
MODEL = create_model("gpt2", device=DEVICE)
print("Model loaded!\n")


def generate_with_config(
    prompt: str,
    max_tokens: int,
    temperature: float,
    use_kv_cache: bool,
    use_gqa: bool,
    use_speculative: bool,
):
    """Generate text with selected optimizations and return metrics."""
    if not prompt.strip():
        return "Please enter a prompt.", ""
    
    config = InferenceConfig(
        use_kv_cache=use_kv_cache,
        use_gqa=use_gqa,
        use_speculative=use_speculative,
        max_new_tokens=max_tokens,
        temperature=temperature,
        device=DEVICE,
    )
    
    generator = Generator(MODEL, config)
    
    start = time.perf_counter()
    text, metrics = generator.generate(prompt)
    total_time = time.perf_counter() - start
    
    # Format metrics display
    active_opts = []
    if use_kv_cache:
        active_opts.append("KV Cache")
    if use_gqa:
        active_opts.append("GQA (12→4)")
    if use_speculative:
        active_opts.append("Speculative")
    if not active_opts:
        active_opts.append("Baseline (naive)")
    
    metrics_text = f"""
━━━ Performance Metrics ━━━
Config:       {' + '.join(active_opts)}
Tokens:       {metrics.tokens_generated}
Throughput:   {metrics.tokens_per_sec:.1f} tokens/sec
TTFT:         {metrics.time_to_first_token_ms:.1f} ms
Avg step:     {metrics.avg_step_time_ms:.1f} ms
Total time:   {total_time:.2f}s
Cache memory: {metrics.cache_memory_mb:.1f} MB
Device:       {DEVICE}
"""
    
    return text, metrics_text


# Build Gradio interface
with gr.Blocks(
    title="LLM Inference Playground",
    theme=gr.themes.Base(primary_hue="blue"),
) as demo:
    gr.Markdown("""
    # ⚡ LLM Inference Playground
    
    Compare inference optimization techniques on GPT-2 124M.
    Toggle optimizations on/off and see the performance impact in real time.
    
    **Everything is implemented from scratch** — no `model.generate()`, no HuggingFace at runtime.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                value="The future of artificial intelligence is",
                lines=3,
            )
            
            with gr.Row():
                max_tokens = gr.Slider(10, 200, value=50, step=10, label="Max Tokens")
                temperature = gr.Slider(0.0, 2.0, value=0.8, step=0.1, label="Temperature")
            
            gr.Markdown("### Optimizations")
            with gr.Row():
                use_kv_cache = gr.Checkbox(label="KV Cache", value=True)
                use_gqa = gr.Checkbox(label="Grouped-Query Attention (12→4)", value=False)
                use_speculative = gr.Checkbox(label="Speculative Decoding", value=False)
            
            generate_btn = gr.Button("Generate ⚡", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            output_text = gr.Textbox(label="Generated Text", lines=8)
            metrics_display = gr.Textbox(label="Performance Metrics", lines=10)
    
    generate_btn.click(
        fn=generate_with_config,
        inputs=[prompt, max_tokens, temperature, use_kv_cache, use_gqa, use_speculative],
        outputs=[output_text, metrics_display],
    )
    
    gr.Markdown("""
    ---
    **How it works:** Each optimization is a standalone module that plugs into the same inference engine.
    The model forward pass, attention computation, KV cache, and generation loop are all implemented from 
    raw PyTorch operations.
    
    [GitHub](#) · [Blog Post](#) · Built by Jason Ling
    """)


if __name__ == "__main__":
    demo.launch(share=False)
