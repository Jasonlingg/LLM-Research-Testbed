"""
Weight loader: maps HuggingFace GPT-2 pretrained weights into our architecture.

This is the bridge that lets us use pretrained weights with our from-scratch code.
The key challenge is that HF uses a different naming convention:
    HF:  transformer.h.0.attn.c_attn.weight
    Ours: blocks.0.attn.c_attn.weight

We also handle the Conv1D → Linear difference: HF's GPT-2 uses Conv1D layers
(which are really just linear layers with transposed weights).
"""
import torch
from model.transformer import GPT2
from config import ModelConfig


def load_pretrained_weights(model: GPT2, model_name: str = "gpt2") -> GPT2:
    """
    Load HuggingFace pretrained GPT-2 weights into our model.
    
    This is the ONLY place we use the transformers library.
    At inference time, we depend only on PyTorch.
    
    Args:
        model: Our GPT2 model instance
        model_name: HuggingFace model identifier ("gpt2" for 124M)
    
    Returns:
        model with pretrained weights loaded
    """
    from transformers import GPT2LMHeadModel
    
    print(f"Loading pretrained weights from '{model_name}'...")
    hf_model = GPT2LMHeadModel.from_pretrained(model_name)
    hf_state = hf_model.state_dict()
    
    # Build mapping: HF key → our key
    mapping = {}
    
    # Embeddings
    mapping["transformer.wte.weight"] = "embedding.token_embedding.weight"
    mapping["transformer.wpe.weight"] = "embedding.position_embedding.weight"
    
    # Transformer blocks
    for i in range(model.config.n_layers):
        hf_prefix = f"transformer.h.{i}"
        our_prefix = f"blocks.{i}"
        
        # Layer norms
        mapping[f"{hf_prefix}.ln_1.weight"] = f"{our_prefix}.ln_1.weight"
        mapping[f"{hf_prefix}.ln_1.bias"] = f"{our_prefix}.ln_1.bias"
        mapping[f"{hf_prefix}.ln_2.weight"] = f"{our_prefix}.ln_2.weight"
        mapping[f"{hf_prefix}.ln_2.bias"] = f"{our_prefix}.ln_2.bias"
        
        # Attention: c_attn (combined QKV) and c_proj
        mapping[f"{hf_prefix}.attn.c_attn.weight"] = f"{our_prefix}.attn.c_attn.weight"
        mapping[f"{hf_prefix}.attn.c_attn.bias"] = f"{our_prefix}.attn.c_attn.bias"
        mapping[f"{hf_prefix}.attn.c_proj.weight"] = f"{our_prefix}.attn.c_proj.weight"
        mapping[f"{hf_prefix}.attn.c_proj.bias"] = f"{our_prefix}.attn.c_proj.bias"
        
        # MLP: c_fc and c_proj
        mapping[f"{hf_prefix}.mlp.c_fc.weight"] = f"{our_prefix}.mlp.c_fc.weight"
        mapping[f"{hf_prefix}.mlp.c_fc.bias"] = f"{our_prefix}.mlp.c_fc.bias"
        mapping[f"{hf_prefix}.mlp.c_proj.weight"] = f"{our_prefix}.mlp.c_proj.weight"
        mapping[f"{hf_prefix}.mlp.c_proj.bias"] = f"{our_prefix}.mlp.c_proj.bias"
    
    # Final layer norm
    mapping["transformer.ln_f.weight"] = "ln_f.weight"
    mapping["transformer.ln_f.bias"] = "ln_f.bias"
    
    # LM head is weight-tied with token embedding, so we skip it
    
    # Load weights with Conv1D transpose handling
    our_state = model.state_dict()
    loaded = 0
    
    for hf_key, our_key in mapping.items():
        if hf_key not in hf_state:
            print(f"  WARNING: {hf_key} not found in HF checkpoint")
            continue
        if our_key not in our_state:
            print(f"  WARNING: {our_key} not found in our model")
            continue
        
        hf_tensor = hf_state[hf_key]
        our_tensor = our_state[our_key]

        # HF's GPT-2 uses Conv1D for all linear projections in attention and MLP
        # Conv1D stores weights as [in_features, out_features]
        # Our nn.Linear expects [out_features, in_features]
        # We need to transpose ALL Conv1D weights, even square ones (attn.c_proj)
        is_conv1d_weight = (
            ".weight" in hf_key and
            len(hf_tensor.shape) == 2 and
            ("attn.c_attn" in hf_key or "attn.c_proj" in hf_key or
             "mlp.c_fc" in hf_key or "mlp.c_proj" in hf_key)
        )

        if is_conv1d_weight:
            hf_tensor = hf_tensor.T

        # Verify shapes match after transpose
        if hf_tensor.shape != our_tensor.shape:
            print(f"  Shape mismatch: {hf_key} {hf_tensor.shape} vs {our_key} {our_tensor.shape}")
            continue

        our_state[our_key] = hf_tensor
        loaded += 1
    
    # Weight tying: lm_head.weight and embedding.token_embedding.weight are the same
    # Parameter. our_state still has lm_head.weight pointing at the old (random) tensor.
    # If we load_state_dict as-is, that would overwrite the shared param with random.
    # Point lm_head.weight at the loaded embedding tensor so both slots get correct weights.
    if "lm_head.weight" in our_state and "embedding.token_embedding.weight" in our_state:
        our_state["lm_head.weight"] = our_state["embedding.token_embedding.weight"]
    
    model.load_state_dict(our_state)
    print(f"  Loaded {loaded}/{len(mapping)} weight tensors")
    
    return model


def create_model(model_name: str = "gpt2", device: str = "cpu", config=None) -> GPT2:
    """
    Convenience function: create model and load pretrained weights.

    Always loads into a standard MHA model first (weight_loader maps to c_attn
    keys that only exist on MultiHeadAttention), then converts to GQA if requested.

    Usage:
        model = create_model("gpt2", device="cuda")
        model = create_model("gpt2", config=InferenceConfig(use_gqa=True))
    """
    from config import InferenceConfig
    if config is None:
        config = InferenceConfig()

    # Always build with standard MHA first so weight_loader can populate c_attn/c_proj.
    # FlashAttention shares identical attribute names (c_attn, c_proj) so we can
    # build with use_flash_attn directly — no separate conversion step needed.
    model_config = ModelConfig(use_flash_attn=config.use_flash_attn)
    model = GPT2(model_config)
    model = load_pretrained_weights(model, model_name)

    if config.use_flash_attn:
        print(f"  FlashAttn: tiled online-softmax (block_size={config.flash_block_size})")

    if config.use_gqa:
        from model.gqa_converter import convert_model_to_gqa
        model = convert_model_to_gqa(model, config.gqa_num_kv_groups)
        reduction = model_config.n_heads // config.gqa_num_kv_groups
        print(f"  GQA: {model_config.n_heads} heads → {config.gqa_num_kv_groups} KV groups ({reduction}x cache reduction)")

    model = model.to(device)
    model.eval()

    print(f"  Model ready: {model.num_parameters / 1e6:.1f}M parameters on {device}")
    return model
