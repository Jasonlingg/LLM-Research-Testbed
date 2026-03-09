"""
GQA converter: replaces MHA attention blocks in a loaded GPT-2 model with GQA.

Strategy:
1. Load pretrained weights into a standard MHA model (weight_loader handles this)
2. Call convert_model_to_gqa() to swap each block's attention module
3. from_pretrained_mha() averages K,V heads within groups for weight conversion

Why load as MHA first: weight_loader maps HF keys to c_attn/c_proj attribute names,
which only exist on MultiHeadAttention. GroupedQueryAttention uses W_q/W_k/W_v instead.
Loading into MHA first, then converting, avoids needing a separate GQA-aware loader.
"""
from model.transformer import GPT2
from optimizations.grouped_query_attention import GroupedQueryAttention


def convert_model_to_gqa(model: GPT2, n_kv_groups: int) -> GPT2:
    """
    Convert all MHA attention layers to GQA in-place.

    Must be called AFTER load_pretrained_weights() so MHA weights are populated.
    Replaces each block.attn (MultiHeadAttention) with GroupedQueryAttention
    using from_pretrained_mha() to perform the weight conversion.

    Weight shape note:
        weight_loader transposes Conv1D → Linear, so after loading:
            block.attn.c_attn.weight  is [3*d_model, d_model]  (Linear format)
            block.attn.c_proj.weight  is [d_model, d_model]    (Linear format)
        from_pretrained_mha() expects the original HF Conv1D format:
            c_attn_weight: [d_model, 3*d_model]
            c_proj_weight: [d_model, d_model]
        So we transpose back before passing.

    Args:
        model: GPT2 model with pretrained MHA weights loaded
        n_kv_groups: Number of KV head groups (e.g. 4 means 12 Q heads → 4 KV heads)

    Returns:
        Same model instance with all attention blocks replaced by GQA
    """
    d_model = model.config.d_model
    n_heads = model.config.n_heads

    for block in model.blocks:
        gqa = GroupedQueryAttention.from_pretrained_mha(
            c_attn_weight=block.attn.c_attn.weight.data.T,  # [d_model, 3*d_model]
            c_attn_bias=block.attn.c_attn.bias.data,         # [3*d_model]
            c_proj_weight=block.attn.c_proj.weight.data.T,   # [d_model, d_model]
            c_proj_bias=block.attn.c_proj.bias.data,          # [d_model]
            d_model=d_model,
            n_heads=n_heads,
            n_kv_groups=n_kv_groups,
        )
        block.attn = gqa

    model.config.use_gqa = True
    model.config.gqa_num_kv_groups = n_kv_groups

    return model
