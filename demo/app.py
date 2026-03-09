"""
Interactive Gradio demo: toggle optimizations and see metrics in real time.

Three tabs:
  1. Generate — streaming token output with live tok/s, baseline vs optimized comparison
  2. Profiler — torch.profiler breakdown showing where time actually goes
  3. About    — architecture notes

Run locally: python -m demo.app
Deploy:      push to HuggingFace Spaces (SDK: Gradio)
"""
import time
import torch
import tiktoken
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from model.weight_loader import create_model
from generation.base_generator import Generator, GenerationMetrics
from generation.sampling import sample_token
from config import InferenceConfig


# ── Model loading ─────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER = tiktoken.get_encoding("gpt2")

print(f"Loading models on {DEVICE}...")
MODEL_MHA   = create_model("gpt2", device=DEVICE)
MODEL_GQA   = create_model("gpt2", device=DEVICE, config=InferenceConfig(use_gqa=True))
MODEL_FLASH = create_model("gpt2", device=DEVICE, config=InferenceConfig(use_flash_attn=True))
print("All models ready.\n")


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,300;0,400;0,500;0,700;1,400&display=swap');
*, *::before, *::after { font-family: 'JetBrains Mono', monospace !important; box-sizing: border-box; }
body, .gradio-container { background: #060809 !important; color: #c9d1d9 !important; }
.gradio-container { max-width: 1100px !important; padding: 0 24px !important; }
footer, .built-with { display: none !important; }
.gr-panel, .gr-box, .gr-form, .contain, .gap, .form { background: transparent !important; border: none !important; box-shadow: none !important; }
textarea, input[type="text"], input[type="number"] { background: #0d1117 !important; border: 1px solid #21262d !important; color: #c9d1d9 !important; border-radius: 6px !important; font-size: 13px !important; padding: 10px 12px !important; }
textarea:focus, input:focus { border-color: #00d4ff !important; outline: none !important; box-shadow: 0 0 0 3px #00d4ff15 !important; }
label > span, .gr-block label span { color: #6e7681 !important; font-size: 10px !important; letter-spacing: 0.12em !important; text-transform: uppercase !important; font-weight: 500 !important; }
input[type="range"] { accent-color: #00d4ff !important; }
input[type="checkbox"] { accent-color: #00d4ff !important; }
.wrap, .gr-check-radio { background: transparent !important; border: none !important; }
.gr-button-primary, button.primary { background: #00d4ff !important; color: #000 !important; border: none !important; border-radius: 6px !important; font-weight: 700 !important; font-size: 12px !important; letter-spacing: 0.12em !important; text-transform: uppercase !important; padding: 14px 28px !important; cursor: pointer !important; width: 100% !important; }
.gr-button-primary:hover, button.primary:hover { background: #33ddff !important; box-shadow: 0 0 24px #00d4ff40 !important; }
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #21262d; border-radius: 2px; }
hr { border-color: #21262d !important; margin: 24px 0 !important; }
.gr-row { gap: 20px !important; }
@keyframes slide-in { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }
.tab-nav button { background: transparent !important; color: #6e7681 !important; border: none !important; border-bottom: 2px solid transparent !important; font-size: 11px !important; letter-spacing: 0.12em !important; text-transform: uppercase !important; padding: 10px 20px !important; }
.tab-nav button.selected { color: #00d4ff !important; border-bottom-color: #00d4ff !important; }
"""

HEADER_HTML = """
<div style="padding:40px 0 32px 0; border-bottom:1px solid #21262d; margin-bottom:28px;">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <polygon points="8,1 15,5 15,11 8,15 1,11 1,5" stroke="#00d4ff" stroke-width="1" fill="none"/>
      <polygon points="8,4 12,6.5 12,9.5 8,12 4,9.5 4,6.5" fill="#00d4ff22" stroke="#00d4ff44" stroke-width="0.5"/>
    </svg>
    <span style="font-size:10px;letter-spacing:0.25em;color:#00d4ff;font-weight:700;text-transform:uppercase;">INFERENCE ENGINE</span>
    <div style="height:1px;flex:1;background:linear-gradient(90deg,#21262d 0%,transparent 100%);"></div>
    <span style="font-size:10px;color:#30363d;letter-spacing:0.1em;">GPT-2 · 124M · CPU/CUDA</span>
  </div>
  <div style="font-size:38px;font-weight:700;color:#e6edf3;line-height:1.1;margin-bottom:14px;letter-spacing:-0.02em;">
    LLM optimization,<br/><span style="color:#00d4ff;">measured live.</span>
  </div>
  <div style="font-size:13px;color:#6e7681;line-height:1.7;max-width:520px;margin-bottom:20px;">
    KV caching. Flash Attention. Grouped-Query Attention. Each optimization built from raw PyTorch — no shortcuts.
    Toggle them and <em style="color:#8b949e;">watch the numbers move</em>.
  </div>
  <div style="display:flex;gap:8px;flex-wrap:wrap;">
    <span style="background:#0d1117;border:1px solid #21262d;color:#6e7681;font-size:10px;padding:4px 10px;border-radius:4px;">no model.generate()</span>
    <span style="background:#0d1117;border:1px solid #21262d;color:#6e7681;font-size:10px;padding:4px 10px;border-radius:4px;">raw matmul attention</span>
    <span style="background:#0d1117;border:1px solid #21262d;color:#6e7681;font-size:10px;padding:4px 10px;border-radius:4px;">tiled flash attention</span>
    <span style="background:#0d1117;border:1px solid #21262d;color:#6e7681;font-size:10px;padding:4px 10px;border-radius:4px;">pre-allocated kv cache</span>
  </div>
</div>
"""

EMPTY_HTML = """
<div style="border:1px dashed #21262d;border-radius:8px;padding:48px 24px;text-align:center;color:#30363d;font-size:11px;letter-spacing:0.12em;line-height:2;">
  ◇ RESULTS APPEAR HERE<br/>
  <span style="font-size:10px;color:#21262d;">configure → run benchmark</span>
</div>
"""

SECTION = "font-size:9px;color:#30363d;letter-spacing:0.18em;text-transform:uppercase;margin:0 0 12px 0;"


# ── HTML helpers ──────────────────────────────────────────────────────────────

def metric_card(label, config_desc, tok_s, ttft, step_ms, mem_mb,
                accent="#ff6b35", icon="◇", speedup=None):
    bar_pct = min(100, tok_s / 150 * 100)
    speedup_badge = ""
    if speedup is not None and speedup > 0:
        color = "#00d4ff" if speedup >= 1 else "#ff6b35"
        speedup_badge = f'<span style="background:{color}15;border:1px solid {color}40;color:{color};font-size:10px;font-weight:700;padding:3px 8px;border-radius:3px;margin-left:10px;letter-spacing:0.08em;">↑ {speedup:.1f}× faster</span>'
    return f"""
    <div style="background:#0d1117;border:1px solid #21262d;border-left:3px solid {accent};border-radius:8px;padding:22px 24px;margin-bottom:14px;animation:slide-in 0.3s ease;">
      <div style="display:flex;align-items:center;margin-bottom:6px;">
        <span style="font-size:10px;color:{accent};letter-spacing:0.18em;font-weight:700;text-transform:uppercase;">{icon} {label}</span>
        {speedup_badge}
      </div>
      <div style="font-size:11px;color:#30363d;margin-bottom:18px;">{config_desc}</div>
      <div style="font-size:44px;font-weight:700;color:{accent};line-height:1;letter-spacing:-0.02em;">{tok_s:.1f}</div>
      <div style="font-size:10px;color:#30363d;letter-spacing:0.18em;text-transform:uppercase;margin:4px 0 16px 0;">tokens / second</div>
      <div style="height:2px;background:#161b22;border-radius:1px;margin-bottom:22px;overflow:hidden;">
        <div style="height:100%;width:{bar_pct:.1f}%;background:linear-gradient(90deg,{accent}66,{accent});border-radius:1px;"></div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;">
        <div><div style="font-size:9px;color:#30363d;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:5px;">TTFT</div>
          <div style="font-size:20px;font-weight:500;color:#8b949e;">{ttft:.0f}<span style="font-size:11px;color:#30363d;margin-left:2px;">ms</span></div></div>
        <div><div style="font-size:9px;color:#30363d;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:5px;">AVG STEP</div>
          <div style="font-size:20px;font-weight:500;color:#8b949e;">{step_ms:.1f}<span style="font-size:11px;color:#30363d;margin-left:2px;">ms</span></div></div>
        <div><div style="font-size:9px;color:#30363d;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:5px;">KV MEM</div>
          <div style="font-size:20px;font-weight:500;color:#8b949e;">{mem_mb:.1f}<span style="font-size:11px;color:#30363d;margin-left:2px;">MB</span></div></div>
      </div>
    </div>"""


def loading_card(label, msg, accent="#00d4ff"):
    return f"""
    <div style="background:#0d1117;border:1px solid #21262d;border-left:3px solid {accent};border-radius:8px;padding:22px 24px;margin-bottom:14px;">
      <div style="font-size:10px;color:{accent};letter-spacing:0.18em;font-weight:700;text-transform:uppercase;margin-bottom:12px;">◆ {label}</div>
      <div style="font-size:12px;color:#6e7681;animation:blink 1.2s ease infinite;">{msg}</div>
    </div>"""


def profiler_html(events, total_ms):
    """Render profiler results as a styled HTML bar chart."""
    rows = sorted(
        [(e.key, e.cpu_time_total / 1000, e.count) for e in events if e.cpu_time_total > 0],
        key=lambda x: -x[1]
    )[:15]
    if not rows or total_ms == 0:
        return "<p style='color:#30363d'>No profiler data.</p>"

    # Categorise for insight
    def cat_ms(keywords):
        return sum(ms for op, ms, _ in rows if any(k in op for k in keywords))

    matmul_ms  = cat_ms(["aten::mm", "aten::bmm", "aten::addmm", "aten::matmul"])
    softmax_ms = cat_ms(["softmax"])
    norm_ms    = cat_ms(["layer_norm"])
    cat_t_ms   = cat_ms(["aten::cat"])
    linear_ms  = cat_ms(["aten::linear"])

    bars = ""
    for op, ms, calls in rows:
        pct = 100 * ms / total_ms
        short = op if len(op) <= 40 else op[:37] + "..."
        # colour by category
        if any(k in op for k in ["mm", "bmm", "addmm", "matmul"]):
            color = "#00d4ff"
        elif "softmax" in op:
            color = "#ff6b35"
        elif "layer_norm" in op:
            color = "#a371f7"
        elif "cat" in op:
            color = "#f78166"
        else:
            color = "#3d4449"
        bars += f"""
        <div style="margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;font-size:10px;color:#6e7681;margin-bottom:4px;">
            <span>{short}</span>
            <span style="color:#8b949e;">{ms:.1f}ms &nbsp; {pct:.1f}% &nbsp; ×{calls}</span>
          </div>
          <div style="height:4px;background:#161b22;border-radius:2px;">
            <div style="height:100%;width:{pct:.1f}%;background:{color};border-radius:2px;"></div>
          </div>
        </div>"""

    # Legend
    legend = """
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:20px;font-size:10px;color:#6e7681;">
      <span><span style="color:#00d4ff;">■</span> matmul</span>
      <span><span style="color:#ff6b35;">■</span> softmax</span>
      <span><span style="color:#a371f7;">■</span> layer_norm</span>
      <span><span style="color:#f78166;">■</span> cat (kv cache)</span>
      <span><span style="color:#3d4449;">■</span> other</span>
    </div>"""

    dominant = max([("matmul", matmul_ms), ("softmax", softmax_ms),
                    ("layer_norm", norm_ms), ("kv cat", cat_t_ms)], key=lambda x: x[1])
    insight = f"""
    <div style="margin-top:20px;padding:14px 16px;background:#0d1117;border:1px solid #21262d;border-left:3px solid #00d4ff;border-radius:6px;font-size:11px;color:#6e7681;line-height:1.8;">
      <div style="color:#00d4ff;font-size:9px;letter-spacing:0.18em;text-transform:uppercase;margin-bottom:8px;">◆ BOTTLENECK ANALYSIS</div>
      <b style="color:#c9d1d9;">{dominant[0]}</b> dominates at {100*dominant[1]/total_ms:.1f}% of total time.<br/>
      matmul: <b style="color:#00d4ff;">{100*matmul_ms/total_ms:.1f}%</b> &nbsp;·&nbsp;
      softmax: <b style="color:#ff6b35;">{100*softmax_ms/total_ms:.1f}%</b> &nbsp;·&nbsp;
      layer_norm (×24): <b style="color:#a371f7;">{100*norm_ms/total_ms:.1f}%</b> &nbsp;·&nbsp;
      kv cat: <b style="color:#f78166;">{100*cat_t_ms/total_ms:.1f}%</b><br/><br/>
      {'<span style="color:#f78166;">KV cache concat is significant — pre-allocated cache eliminates O(n) copy per step.</span><br/>' if cat_t_ms > 0.05 * total_ms else ''}
      {'<span style="color:#a371f7;">LayerNorm is costly at 24 calls/step — fused kernels (apex/triton) help here.</span>' if norm_ms > 0.15 * total_ms else ''}
    </div>"""

    return f"""
    <div style="font-size:9px;color:#30363d;letter-spacing:0.18em;text-transform:uppercase;margin-bottom:16px;">
      ◆ OP BREAKDOWN — {total_ms:.0f}ms total (CPU time)
    </div>
    {legend}{bars}{insight}"""


# ── Streaming generation ──────────────────────────────────────────────────────

@torch.no_grad()
def _stream_tokens(model, prompt, config):
    """
    Generator: yields (decoded_text, live_metrics_dict) after each new token.

    Uses the same prefill → decode structure as base_generator.py but yields
    control back to Gradio after every token so the UI updates in real time.
    """
    input_ids = torch.tensor(
        [TOKENIZER.encode(prompt)], dtype=torch.long, device=DEVICE
    )
    generated = input_ids.clone()
    start = time.perf_counter()
    ttft = None
    step_times = []

    if config.use_kv_cache:
        # Prefill
        pf_start = time.perf_counter()
        logits, kv_caches = model(input_ids, use_cache=True)
        ttft = (time.perf_counter() - pf_start) * 1000

        next_token = sample_token(logits[:, -1, :], config.temperature, config.top_k, config.top_p)
        generated = torch.cat([generated, next_token], dim=1)
        n_new = 1
        elapsed = time.perf_counter() - start

        yield TOKENIZER.decode(generated[0].tolist()), {
            "ttft": ttft, "tok_s": n_new / elapsed, "n": n_new, "step_ms": ttft, "mem_mb": 0.0
        }

        for _ in range(config.max_new_tokens - 1):
            step_start = time.perf_counter()
            logits, kv_caches = model(next_token, kv_caches=kv_caches, use_cache=True)
            next_token = sample_token(logits[:, -1, :], config.temperature, config.top_k, config.top_p)
            generated = torch.cat([generated, next_token], dim=1)

            step_ms = (time.perf_counter() - step_start) * 1000
            step_times.append(step_ms)
            n_new = generated.shape[1] - input_ids.shape[1]
            elapsed = time.perf_counter() - start

            # Cache memory
            k, v = kv_caches[0]
            mem_mb = (k.nelement() * k.element_size() * 2 * len(kv_caches)) / (1024 * 1024)

            yield TOKENIZER.decode(generated[0].tolist()), {
                "ttft": ttft,
                "tok_s": n_new / elapsed,
                "n": n_new,
                "step_ms": sum(step_times) / len(step_times),
                "mem_mb": mem_mb,
            }
            if next_token.item() == TOKENIZER.eot_token:
                break

    else:
        # Naive: full recompute each step
        for i in range(config.max_new_tokens):
            step_start = time.perf_counter()
            logits, _ = model(generated, use_cache=False)
            next_token = sample_token(logits[:, -1, :], config.temperature, config.top_k, config.top_p)
            generated = torch.cat([generated, next_token], dim=1)

            step_ms = (time.perf_counter() - step_start) * 1000
            if ttft is None:
                ttft = step_ms
            step_times.append(step_ms)
            n_new = generated.shape[1] - input_ids.shape[1]
            elapsed = time.perf_counter() - start

            yield TOKENIZER.decode(generated[0].tolist()), {
                "ttft": ttft,
                "tok_s": n_new / elapsed,
                "n": n_new,
                "step_ms": sum(step_times) / len(step_times),
                "mem_mb": 0.0,
            }
            if next_token.item() == TOKENIZER.eot_token:
                break


def generate_streaming(prompt, max_tokens, temperature, use_kv_cache, use_gqa, use_flash_attn):
    """
    Gradio generator: yields (text, metrics_html) pairs.
    Runs baseline first (blocking), then streams optimized generation live.
    """
    if not prompt.strip():
        yield "", EMPTY_HTML
        return

    # Step 1: show loading
    yield "", loading_card("BASELINE", "Running baseline (naive, no cache)...", "#ff6b35")

    # Step 2: Run baseline (fast, capped at 30 tokens)
    baseline_config = InferenceConfig(
        use_kv_cache=False,
        max_new_tokens=min(int(max_tokens), 30),
        temperature=float(temperature),
        top_k=50, top_p=0.95, device=DEVICE,
    )
    baseline_gen = Generator(MODEL_MHA, baseline_config)
    _, bm = baseline_gen.generate(prompt)

    b_card = metric_card(
        "BASELINE", f"naive · full recompute every step · {bm.tokens_generated} tokens",
        bm.tokens_per_sec, bm.time_to_first_token_ms, bm.avg_step_time_ms, bm.cache_memory_mb,
        accent="#ff6b35", icon="◇",
    )

    # Step 3: show baseline + start optimized
    parts = []
    if use_kv_cache:   parts.append("KV Cache")
    if use_flash_attn: parts.append("Flash Attn")
    if use_gqa:        parts.append("GQA 12→4")
    if not parts:      parts.append("Naive")
    opt_label = " + ".join(parts)

    yield "", b_card + loading_card("OPTIMIZED", f"Streaming {opt_label}...", "#00d4ff")

    # Step 4: stream optimized
    if use_flash_attn:
        model = MODEL_FLASH
    elif use_gqa:
        model = MODEL_GQA
    else:
        model = MODEL_MHA

    opt_config = InferenceConfig(
        use_kv_cache=use_kv_cache,
        use_flash_attn=use_flash_attn,
        use_gqa=use_gqa,
        max_new_tokens=int(max_tokens),
        temperature=float(temperature),
        top_k=50, top_p=0.95, device=DEVICE,
    )

    for text, lm in _stream_tokens(model, prompt, opt_config):
        speedup = lm["tok_s"] / max(bm.tokens_per_sec, 1e-6)
        live_card = metric_card(
            "OPTIMIZED",
            f"{opt_label} · {lm['n']} tokens so far · {DEVICE}",
            lm["tok_s"], lm["ttft"], lm["step_ms"], lm["mem_mb"],
            accent="#00d4ff", icon="◆", speedup=speedup,
        )
        yield text, b_card + live_card


# ── Profiler ──────────────────────────────────────────────────────────────────

def run_profiler(prompt, use_flash_attn, use_gqa):
    """Run torch.profiler for 20 tokens and return HTML breakdown."""
    if not prompt.strip():
        return "<p style='color:#30363d'>Enter a prompt first.</p>"

    model = MODEL_FLASH if use_flash_attn else (MODEL_GQA if use_gqa else MODEL_MHA)
    config = InferenceConfig(
        use_kv_cache=True,
        use_flash_attn=use_flash_attn,
        use_gqa=use_gqa,
        max_new_tokens=20,
        temperature=0.8, top_k=50, top_p=0.95, device=DEVICE,
    )

    # Warmup
    gen = Generator(model, config)
    gen.generate(prompt)

    # Profile
    activities = [torch.profiler.ProfilerActivity.CPU]
    if DEVICE == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=False) as prof:
        gen.generate(prompt)

    events = prof.key_averages()
    total_ms = sum(e.cpu_time_total / 1000 for e in events if e.cpu_time_total > 0)

    label = "Flash Attn" if use_flash_attn else ("GQA" if use_gqa else "Standard MHA")
    header = f'<div style="{SECTION}margin-bottom:20px;">◆ PROFILER — {label} · 20 tokens · {DEVICE}</div>'
    return header + profiler_html(events, total_ms)


# ── Build UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(css=CSS, title="Inference Engine") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Tabs():

        # ── Tab 1: Generate ───────────────────────────────────────────────────
        with gr.Tab("Generate"):
            with gr.Row():

                # Left: controls
                with gr.Column(scale=1):
                    gr.HTML(f'<div style="{SECTION}">◆ PROMPT</div>')
                    prompt = gr.Textbox(
                        label="", placeholder="Enter your prompt...",
                        value="The future of artificial intelligence is",
                        lines=4, show_label=False,
                    )

                    gr.HTML(f'<div style="{SECTION};margin-top:20px;">◆ PARAMETERS</div>')
                    max_tokens  = gr.Slider(10, 200, value=60, step=10, label="Max Tokens")
                    temperature = gr.Slider(0.0, 2.0, value=0.8, step=0.1, label="Temperature")

                    gr.HTML(f'<div style="{SECTION};margin-top:20px;">◆ OPTIMIZATIONS</div>')
                    use_kv_cache   = gr.Checkbox(value=True,  label="KV Cache  ·  O(1) decode vs O(n²) naive")
                    use_flash_attn = gr.Checkbox(value=False, label="Flash Attention  ·  tiled online-softmax, O(N) memory")
                    use_gqa        = gr.Checkbox(value=False, label="Grouped-Query Attention  ·  3× smaller KV cache")

                    gr.HTML('<div style="margin-top:20px;"></div>')
                    run_btn = gr.Button("▶  RUN BENCHMARK", variant="primary", size="lg")

                    gr.HTML("""
                    <div style="margin-top:20px;padding:14px 16px;background:#0d1117;border:1px solid #21262d;border-radius:6px;font-size:11px;color:#30363d;line-height:1.8;">
                      Runs <span style="color:#6e7681;">baseline</span> (naive) then streams
                      <span style="color:#00d4ff;">your config</span> live.<br/>
                      Speedup is real. Tokens appear as they're generated.
                    </div>""")

                # Right: outputs
                with gr.Column(scale=1):
                    metrics_html = gr.HTML(EMPTY_HTML)
                    gr.HTML(f'<div style="{SECTION};margin-top:20px;">◆ GENERATED TEXT</div>')
                    output_text = gr.Textbox(
                        label="", lines=8, show_label=False,
                        placeholder="tokens stream here as they generate...",
                    )

            run_btn.click(
                fn=generate_streaming,
                inputs=[prompt, max_tokens, temperature, use_kv_cache, use_gqa, use_flash_attn],
                outputs=[output_text, metrics_html],
            )

        # ── Tab 2: Profiler ───────────────────────────────────────────────────
        with gr.Tab("Profiler"):
            gr.HTML(f"""
            <div style="padding:24px 0 16px 0;border-bottom:1px solid #21262d;margin-bottom:24px;">
              <div style="{SECTION}">◆ PYTORCH PROFILER</div>
              <div style="font-size:13px;color:#6e7681;line-height:1.7;max-width:540px;margin-top:8px;">
                Wraps <code style="color:#00d4ff;background:#0d1117;padding:1px 6px;border-radius:3px;">torch.profiler.profile</code>
                around a 20-token generation run and shows where time actually goes.
                Identifies whether you're bottlenecked by matmul, softmax, layernorm, or KV cache ops.
              </div>
            </div>""")

            with gr.Row():
                with gr.Column(scale=1):
                    prof_prompt = gr.Textbox(
                        label="Prompt", value="The history of artificial intelligence began",
                        lines=3,
                    )
                    prof_flash = gr.Checkbox(value=False, label="Flash Attention")
                    prof_gqa   = gr.Checkbox(value=False, label="Grouped-Query Attention")
                    gr.HTML('<div style="margin-top:16px;"></div>')
                    prof_btn = gr.Button("▶  PROFILE", variant="primary", size="lg")
                    gr.HTML("""
                    <div style="margin-top:16px;padding:14px 16px;background:#0d1117;border:1px solid #21262d;border-radius:6px;font-size:11px;color:#30363d;line-height:1.8;">
                      Runs one warmup pass (not counted),<br/>
                      then profiles 20 tokens of cached generation.<br/>
                      CPU time. Chrome trace exported to<br/>
                      <span style="color:#6e7681;">benchmarks/results/trace.json</span>
                    </div>""")

                with gr.Column(scale=2):
                    prof_output = gr.HTML("""
                    <div style="border:1px dashed #21262d;border-radius:8px;padding:48px 24px;text-align:center;color:#30363d;font-size:11px;letter-spacing:0.12em;line-height:2;">
                      ◇ PROFILER RESULTS APPEAR HERE<br/>
                      <span style="font-size:10px;color:#21262d;">click profile to run</span>
                    </div>""")

            prof_btn.click(
                fn=run_profiler,
                inputs=[prof_prompt, prof_flash, prof_gqa],
                outputs=[prof_output],
            )

    # Footer
    gr.HTML("""
    <div style="border-top:1px solid #21262d;margin-top:36px;padding-top:20px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
      <div style="font-size:10px;color:#21262d;letter-spacing:0.06em;">raw pytorch · no model.generate() · no huggingface at runtime</div>
      <div style="font-size:10px;color:#21262d;">Jason Ling ·
        <a href="https://github.com" style="color:#30363d;text-decoration:none;">github ↗</a>
      </div>
    </div>""")


if __name__ == "__main__":
    demo.queue()   # required for streaming generators
    demo.launch(share=False)
