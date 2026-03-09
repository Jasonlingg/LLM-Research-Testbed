"""
Interactive Gradio demo: toggle optimizations and see metrics in real time.

Deploy to HuggingFace Spaces for a live URL anyone can try.

Run locally: python -m demo.app
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from model.weight_loader import create_model
from generation.base_generator import Generator
from config import InferenceConfig


# Preload both MHA and GQA models at startup so GQA toggle is instant
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading models on {DEVICE}...")
MODEL_MHA = create_model("gpt2", device=DEVICE)
MODEL_GQA = create_model("gpt2", device=DEVICE, config=InferenceConfig(use_gqa=True, gqa_num_kv_groups=4))
print("Models loaded!\n")


CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,300;0,400;0,500;0,700;1,400&display=swap');

*, *::before, *::after {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    box-sizing: border-box;
}

body, .gradio-container {
    background: #060809 !important;
    color: #c9d1d9 !important;
}

.gradio-container {
    max-width: 1100px !important;
    padding: 0 24px !important;
}

/* Kill default Gradio chrome */
footer { display: none !important; }
.built-with { display: none !important; }

/* Panels transparent */
.gr-panel, .gr-box, .gr-form, .contain, .gap, .form {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Inputs */
textarea, input[type="text"], input[type="number"] {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    color: #c9d1d9 !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    transition: border-color 0.2s !important;
    padding: 10px 12px !important;
}

textarea:focus, input:focus {
    border-color: #00d4ff !important;
    outline: none !important;
    box-shadow: 0 0 0 3px #00d4ff15 !important;
}

/* Labels */
label > span, .gr-block label span {
    color: #6e7681 !important;
    font-size: 10px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}

/* Sliders */
input[type="range"] { accent-color: #00d4ff !important; }
.wrap { background: transparent !important; }

/* Checkboxes */
input[type="checkbox"] { accent-color: #00d4ff !important; }
.gr-check-radio { background: transparent !important; border: none !important; }

/* Primary button */
.gr-button-primary, button.primary {
    background: #00d4ff !important;
    color: #000 !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 14px 28px !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    width: 100% !important;
}

.gr-button-primary:hover, button.primary:hover {
    background: #33ddff !important;
    box-shadow: 0 0 24px #00d4ff40 !important;
    transform: translateY(-1px) !important;
}

.gr-button-primary:active, button.primary:active {
    transform: translateY(0px) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #21262d; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #00d4ff44; }

/* Section separators */
hr { border-color: #21262d !important; margin: 24px 0 !important; }

/* Row gap */
.gr-row { gap: 20px !important; }

/* Animations */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

@keyframes slide-in {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes bar-fill {
    from { width: 0%; }
    to { width: var(--bar-w); }
}
"""


HEADER_HTML = """
<div style="
    padding: 40px 0 32px 0;
    border-bottom: 1px solid #21262d;
    margin-bottom: 28px;
">
    <div style="
        display: flex; align-items: center; gap: 10px; margin-bottom: 10px;
    ">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <polygon points="8,1 15,5 15,11 8,15 1,11 1,5" stroke="#00d4ff" stroke-width="1" fill="none"/>
            <polygon points="8,4 12,6.5 12,9.5 8,12 4,9.5 4,6.5" fill="#00d4ff22" stroke="#00d4ff44" stroke-width="0.5"/>
        </svg>
        <span style="font-size: 10px; letter-spacing: 0.25em; color: #00d4ff; font-weight: 700; text-transform: uppercase;">
            INFERENCE ENGINE
        </span>
        <div style="height: 1px; flex: 1; background: linear-gradient(90deg, #21262d 0%, transparent 100%);"></div>
        <span style="font-size: 10px; color: #30363d; letter-spacing: 0.1em;">GPT-2 · 124M · CPU/CUDA</span>
    </div>

    <div style="font-size: 38px; font-weight: 700; color: #e6edf3; line-height: 1.1; margin-bottom: 14px; letter-spacing: -0.02em;">
        LLM optimization,<br/>
        <span style="color: #00d4ff;">measured live.</span>
    </div>

    <div style="font-size: 13px; color: #6e7681; line-height: 1.7; max-width: 520px; margin-bottom: 20px;">
        KV caching. Grouped-Query Attention. Speculative decoding.
        Each optimization built from raw PyTorch — no shortcuts.
        Toggle them and <em style="color: #8b949e;">watch the numbers move</em>.
    </div>

    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
        <span style="
            background: #0d1117; border: 1px solid #21262d; color: #6e7681;
            font-size: 10px; padding: 4px 10px; border-radius: 4px; letter-spacing: 0.08em;
        ">no model.generate()</span>
        <span style="
            background: #0d1117; border: 1px solid #21262d; color: #6e7681;
            font-size: 10px; padding: 4px 10px; border-radius: 4px; letter-spacing: 0.08em;
        ">raw matmul attention</span>
        <span style="
            background: #0d1117; border: 1px solid #21262d; color: #6e7681;
            font-size: 10px; padding: 4px 10px; border-radius: 4px; letter-spacing: 0.08em;
        ">pre-allocated kv cache</span>
        <span style="
            background: #0d1117; border: 1px solid #21262d; color: #6e7681;
            font-size: 10px; padding: 4px 10px; border-radius: 4px; letter-spacing: 0.08em;
        ">validated vs huggingface</span>
    </div>
</div>
"""


EMPTY_STATE_HTML = """
<div style="
    border: 1px dashed #21262d; border-radius: 8px;
    padding: 48px 24px; text-align: center; color: #30363d;
    font-size: 11px; letter-spacing: 0.12em; line-height: 2;
">
    ◇ RESULTS APPEAR HERE<br/>
    <span style="font-size: 10px; color: #21262d;">
        configure optimizations → run benchmark
    </span>
</div>
"""


def make_metric_card(label, metrics, config_desc, is_baseline=False, speedup=None, ttft_delta=None):
    """Render a benchmark result card as HTML."""
    tok_s = metrics.tokens_per_sec
    ttft = metrics.time_to_first_token_ms
    step = metrics.avg_step_time_ms
    mem = metrics.cache_memory_mb

    accent = "#ff6b35" if is_baseline else "#00d4ff"
    icon = "◇" if is_baseline else "◆"
    bar_pct = min(100, tok_s / 150 * 100)
    bar_color = f"linear-gradient(90deg, {accent}66, {accent})"
    glow = "" if is_baseline else f"box-shadow: 0 0 40px #00d4ff08;"

    speedup_badge = ""
    if speedup is not None and not is_baseline:
        sign = "+" if speedup >= 1 else ""
        color = "#00d4ff" if speedup > 1 else "#ff6b35"
        speedup_badge = f"""
        <span style="
            background: {color}15; border: 1px solid {color}40;
            color: {color}; font-size: 10px; font-weight: 700;
            padding: 3px 8px; border-radius: 3px; margin-left: 10px;
            letter-spacing: 0.08em;
        ">↑ {speedup:.1f}× faster</span>
        """

    ttft_badge = ""
    if ttft_delta is not None and not is_baseline:
        pct = int((1 - ttft_delta) * 100)
        ttft_badge = f'<span style="font-size: 10px; color: #00d4ff66; margin-left: 6px;">−{pct}% TTFT</span>' if pct > 0 else ""

    return f"""
    <div style="
        background: #0d1117;
        border: 1px solid #21262d;
        border-left: 3px solid {accent};
        border-radius: 8px; padding: 22px 24px;
        margin-bottom: 14px;
        {glow}
        animation: slide-in 0.3s ease;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <span style="font-size: 10px; color: {accent}; letter-spacing: 0.18em; font-weight: 700; text-transform: uppercase;">
                {icon} {label}
            </span>
            {speedup_badge}
        </div>

        <div style="font-size: 11px; color: #30363d; margin-bottom: 18px; letter-spacing: 0.04em;">
            {config_desc}
        </div>

        <div style="font-size: 44px; font-weight: 700; color: {accent}; line-height: 1; letter-spacing: -0.02em;">
            {tok_s:.1f}
        </div>
        <div style="font-size: 10px; color: #30363d; letter-spacing: 0.18em; text-transform: uppercase; margin: 4px 0 16px 0;">
            tokens / second
        </div>

        <div style="
            height: 2px; background: #161b22; border-radius: 1px;
            margin-bottom: 22px; overflow: hidden;
        ">
            <div style="
                height: 100%; width: {bar_pct:.1f}%;
                background: {bar_color};
                border-radius: 1px;
            "></div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;">
            <div>
                <div style="font-size: 9px; color: #30363d; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 5px;">
                    TTFT {ttft_badge}
                </div>
                <div style="font-size: 20px; font-weight: 500; color: #8b949e;">
                    {ttft:.0f}<span style="font-size: 11px; color: #30363d; margin-left: 2px;">ms</span>
                </div>
            </div>
            <div>
                <div style="font-size: 9px; color: #30363d; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 5px;">
                    AVG STEP
                </div>
                <div style="font-size: 20px; font-weight: 500; color: #8b949e;">
                    {step:.1f}<span style="font-size: 11px; color: #30363d; margin-left: 2px;">ms</span>
                </div>
            </div>
            <div>
                <div style="font-size: 9px; color: #30363d; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 5px;">
                    KV MEM
                </div>
                <div style="font-size: 20px; font-weight: 500; color: #8b949e;">
                    {mem:.1f}<span style="font-size: 11px; color: #30363d; margin-left: 2px;">MB</span>
                </div>
            </div>
        </div>
    </div>
    """


def generate_with_config(
    prompt: str,
    max_tokens: int,
    temperature: float,
    use_kv_cache: bool,
    use_gqa: bool,
    use_speculative: bool,
):
    """Run baseline + optimized generation side-by-side, return HTML comparison."""
    if not prompt.strip():
        return "", EMPTY_STATE_HTML

    # Baseline: always naive (no cache), capped at 30 tokens so it's not painfully slow
    baseline_config = InferenceConfig(
        use_kv_cache=False,
        max_new_tokens=min(max_tokens, 30),
        temperature=temperature,
        device=DEVICE,
    )
    baseline_gen = Generator(MODEL_MHA, baseline_config)
    _, baseline_metrics = baseline_gen.generate(prompt)

    # Optimized: user's selected config
    opt_config = InferenceConfig(
        use_kv_cache=use_kv_cache,
        use_gqa=use_gqa,
        use_speculative=use_speculative,
        max_new_tokens=max_tokens,
        temperature=temperature,
        device=DEVICE,
    )
    model = MODEL_GQA if use_gqa else MODEL_MHA
    opt_gen = Generator(model, opt_config)
    opt_text, opt_metrics = opt_gen.generate(prompt)

    # Labels
    parts = []
    if use_kv_cache:  parts.append("KV Cache")
    if use_gqa:       parts.append("GQA 12→4")
    if use_speculative: parts.append("Speculative")
    if not parts:     parts.append("Naive")
    opt_label = " + ".join(parts)

    # Speedup ratios
    speedup = opt_metrics.tokens_per_sec / max(baseline_metrics.tokens_per_sec, 1e-6)
    ttft_ratio = opt_metrics.time_to_first_token_ms / max(baseline_metrics.time_to_first_token_ms, 1e-6)

    baseline_card = make_metric_card(
        "BASELINE",
        baseline_metrics,
        f"naive · full recompute every step · {baseline_metrics.tokens_generated} tokens",
        is_baseline=True,
    )
    opt_card = make_metric_card(
        "OPTIMIZED",
        opt_metrics,
        f"{opt_label} · {opt_metrics.tokens_generated} tokens · device: {DEVICE}",
        is_baseline=False,
        speedup=speedup,
        ttft_delta=ttft_ratio,
    )

    html = f"""
    <div style="font-size: 9px; color: #30363d; letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 16px;">
        ◆ BENCHMARK — BASELINE VS OPTIMIZED
    </div>
    {baseline_card}
    {opt_card}
    <div style="
        display: flex; align-items: center; gap: 8px;
        font-size: 10px; color: #21262d; margin-top: 4px;
    ">
        <div style="height: 1px; flex: 1; background: #21262d;"></div>
        baseline capped at {baseline_metrics.tokens_generated} tokens for speed
        <div style="height: 1px; flex: 1; background: #21262d;"></div>
    </div>
    """

    return opt_text, html


# ── Build interface ───────────────────────────────────────────────────────────

SECTION = "font-size: 9px; color: #30363d; letter-spacing: 0.18em; text-transform: uppercase; margin: 0 0 12px 0;"

with gr.Blocks(css=CSS, title="Inference Engine") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Row():

        # ── Left: controls ────────────────────────────────────────────────────
        with gr.Column(scale=1):

            gr.HTML(f'<div style="{SECTION}">◆ PROMPT</div>')
            prompt = gr.Textbox(
                label="",
                placeholder="Enter your prompt...",
                value="The future of artificial intelligence is",
                lines=4,
                show_label=False,
            )

            gr.HTML(f'<div style="{SECTION}; margin-top: 20px;">◆ PARAMETERS</div>')
            max_tokens  = gr.Slider(10, 200, value=60, step=10, label="Max Tokens")
            temperature = gr.Slider(0.0, 2.0, value=0.8, step=0.1, label="Temperature")

            gr.HTML(f'<div style="{SECTION}; margin-top: 20px;">◆ OPTIMIZATIONS</div>')

            use_kv_cache   = gr.Checkbox(value=True,  label="KV Cache  ·  O(1) decode vs O(n²) naive")
            use_gqa        = gr.Checkbox(value=False, label="Grouped-Query Attention  ·  3× smaller KV cache")
            use_speculative = gr.Checkbox(value=False, label="Speculative Decoding  ·  coming soon")

            gr.HTML('<div style="margin-top: 20px;"></div>')
            generate_btn = gr.Button("▶  RUN BENCHMARK", variant="primary", size="lg")

            gr.HTML("""
            <div style="
                margin-top: 20px; padding: 14px 16px;
                background: #0d1117; border: 1px solid #21262d;
                border-radius: 6px; font-size: 11px; color: #30363d; line-height: 1.8;
            ">
                Runs <span style="color: #6e7681;">baseline</span> (naive, no cache)<br/>
                then <span style="color: #00d4ff;">your config</span> — side by side.<br/>
                The speedup is real. Check the source.
            </div>
            """)

        # ── Right: outputs ────────────────────────────────────────────────────
        with gr.Column(scale=1):

            metrics_display = gr.HTML(EMPTY_STATE_HTML)

            gr.HTML(f'<div style="{SECTION}; margin-top: 20px;">◆ GENERATED TEXT</div>')
            output_text = gr.Textbox(
                label="",
                lines=7,
                show_label=False,
                placeholder="output appears here...",
            )

    generate_btn.click(
        fn=generate_with_config,
        inputs=[prompt, max_tokens, temperature, use_kv_cache, use_gqa, use_speculative],
        outputs=[output_text, metrics_display],
    )

    gr.HTML("""
    <div style="
        border-top: 1px solid #21262d; margin-top: 36px; padding-top: 20px;
        display: flex; justify-content: space-between; align-items: center;
        flex-wrap: wrap; gap: 12px;
    ">
        <div style="font-size: 10px; color: #21262d; letter-spacing: 0.06em;">
            raw pytorch · no model.generate() · no huggingface at runtime
        </div>
        <div style="font-size: 10px; color: #21262d;">
            Jason Ling ·
            <a href="https://github.com" style="color: #30363d; text-decoration: none; letter-spacing: 0.05em;">
                github ↗
            </a>
        </div>
    </div>
    """)


if __name__ == "__main__":
    demo.launch(share=False)
