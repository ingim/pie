#!/usr/bin/env python3
"""
Layer-by-layer divergence finder with profiling: HuggingFace vs MPS Backend

Manually processes each layer for both HF (using native ops) and MPS (using our ops),
comparing intermediate outputs to find exactly where they diverge.

NOW WITH HIERARCHICAL PROFILING!
"""

import sys
import os
from pathlib import Path

# Add backend-python to path (we're in tests/metal_kernels_debug/)
workspace_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(workspace_dir / "backend" / "backend-python"))

# DO NOT set PIE_METAL_PYTORCH_MODE - we want to test actual Metal kernels

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import profiler (unified profiler)
from profiler import start_profile, initialize_memory_tracker, get_memory_tracker

cache_dir = os.path.expanduser("~/.cache/pie")

# Initialize unified profiler
initialize_memory_tracker(output_dir=".", enabled=True, enable_timing=True)

print("=" * 80)
print("LAYER-BY-LAYER DIVERGENCE FINDER: HF vs MPS")
print("=" * 80)
print("Processing each layer manually to find where HF and MPS diverge")

# Load models
print("\nLoading models...")
model_hf = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    cache_dir=cache_dir,
    attn_implementation="eager",  # Required for output_attentions=True
)
model_hf = model_hf.to("mps").to(torch.float32)
model_hf.eval()

from server import build_config
from model_loader import load_model as load_model_common, load_model_info
from model_factory import create_model_and_fusion_map

config = build_config(
    model="llama-3.2-1b-instruct",
    host="localhost",
    port=62105,
    controller_host="127.0.0.1",
    controller_port=8080,
    auth_token=None,
    cache_dir=cache_dir,
    kv_page_size=16,
    max_dist_size=32,
    max_num_kv_pages=10240,
    max_num_embeds=128,
    max_num_adapters=32,
    max_adapter_rank=8,
    device="mps",
    dtype="float32",
)

model_info = load_model_info(config)
model_mps = load_model_common(config, model_info, create_model_and_fusion_map)
model_mps = model_mps.to(dtype=torch.float32)
model_mps.eval()

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct", cache_dir=cache_dir
)

print("✅ Models loaded")

# Test prompt
test_prompt = "The capital of France is"
test_tokens = tokenizer.encode(test_prompt, add_special_tokens=False)
num_tokens = len(test_tokens)

print(f"\nPrompt: {repr(test_prompt)}")
print(f"Tokens: {test_tokens} ({num_tokens} tokens)")

# Get embeddings
input_ids = torch.tensor(test_tokens, dtype=torch.long, device="mps")

with torch.no_grad():
    hidden_hf = model_hf.model.embed_tokens(input_ids)
    hidden_mps = model_mps.model.embed_tokens(input_ids)

    embed_diff = (hidden_hf - hidden_mps).abs().max().item()
    print(f"\nEmbeddings diff: {embed_diff:.6e}")

    num_heads = 32
    num_kv_heads = 8
    head_dim = 64

    # Setup for MPS
    position_ids_mps = torch.arange(num_tokens, dtype=torch.long, device="mps")

    # Allocate KV cache for MPS
    kv_cache_mps = []
    for _ in range(16):
        cache_layer = torch.zeros(
            1, 2, num_tokens, num_kv_heads, head_dim, dtype=torch.float32, device="mps"
        )
        kv_cache_mps.append(cache_layer)

    print("\n" + "=" * 80)
    print("LAYER-BY-LAYER COMPARISON")
    print("=" * 80)

    # Get HF full output with hidden states and attention outputs
    print("\nRunning HF full forward pass...")
    with torch.no_grad():
        hf_outputs = model_hf(
            input_ids.unsqueeze(0), output_hidden_states=True, output_attentions=True
        )
        hf_hidden_states = hf_outputs.hidden_states  # Tuple of (num_layers+1) tensors
        hf_attentions = hf_outputs.attentions  # Tuple of (num_layers) tensors

    print(f"Got {len(hf_hidden_states)} hidden state tensors from HF")
    print(
        f"Got {len(hf_attentions)} attention tensors from HF (shape: {hf_attentions[0].shape})"
    )

    # Start profiling the full MPS inference
    with start_profile("mps_inference"):
        for layer_idx in range(16):
            with start_profile(f"layer_{layer_idx}"):
                print(f"\n--- Layer {layer_idx} ---")

                layer_mps = model_mps.model.layers[layer_idx]

                # Get HF's INPUT to this layer
                hidden_hf_input = hf_hidden_states[layer_idx].squeeze(0)
                # Get HF's OUTPUT from this layer
                hidden_hf_output = hf_hidden_states[layer_idx + 1].squeeze(0)

                # Check input to this layer
                input_diff = (hidden_hf_input - hidden_mps).abs().max().item()
                print(f"  Input diff: {input_diff:.6e}")

                # MPS: Manual processing with our operations
                residual_mps = hidden_mps

                with start_profile("input_norm"):
                    hidden_mps_norm = layer_mps.input_layernorm(hidden_mps)

                # QKV projection (MPS uses fused)
                with start_profile("qkv_projection"):
                    qkv_mps = layer_mps.self_attn.qkv_proj(hidden_mps_norm)
                    q_mps, k_mps, v_mps = torch.split(
                        qkv_mps,
                        [
                            num_heads * head_dim,
                            num_kv_heads * head_dim,
                            num_kv_heads * head_dim,
                        ],
                        dim=-1,
                    )

                # Reshape
                with start_profile("qkv_reshape"):
                    q_mps = q_mps.view(num_tokens, num_heads, head_dim)
                    k_mps = k_mps.view(num_tokens, num_kv_heads, head_dim)
                    v_mps = v_mps.view(num_tokens, num_kv_heads, head_dim)

                if layer_idx == 0:
                    # Compare with HF's QKV for layer 0
                    with torch.no_grad():
                        layer_hf_0 = model_hf.model.layers[0]
                        hidden_hf_norm = layer_hf_0.input_layernorm(hidden_hf_input)
                        q_hf_check = layer_hf_0.self_attn.q_proj(
                            hidden_hf_norm.squeeze(0)
                            if hidden_hf_norm.dim() > 2
                            else hidden_hf_norm
                        )
                        k_hf_check = layer_hf_0.self_attn.k_proj(
                            hidden_hf_norm.squeeze(0)
                            if hidden_hf_norm.dim() > 2
                            else hidden_hf_norm
                        )
                        v_hf_check = layer_hf_0.self_attn.v_proj(
                            hidden_hf_norm.squeeze(0)
                            if hidden_hf_norm.dim() > 2
                            else hidden_hf_norm
                        )

                        q_hf_check = q_hf_check.view(num_tokens, num_heads, head_dim)
                        k_hf_check = k_hf_check.view(num_tokens, num_kv_heads, head_dim)
                        v_hf_check = v_hf_check.view(num_tokens, num_kv_heads, head_dim)

                        q_diff_before_rope = (q_hf_check - q_mps).abs().max().item()
                        k_diff_before_rope = (k_hf_check - k_mps).abs().max().item()
                        v_diff_before_rope = (v_hf_check - v_mps).abs().max().item()

                        print(f"\n  🔍 Layer 0 QKV before RoPE:")
                        print(f"     Q diff: {q_diff_before_rope:.6e}")
                        print(f"     K diff: {k_diff_before_rope:.6e}")
                        print(f"     V diff: {v_diff_before_rope:.6e}")

                        # Save a copy of Q before RoPE for debugging
                        q_mps_before_rope = q_mps.clone()
                        k_mps_before_rope = k_mps.clone()

                # Apply RoPE using MPS Metal kernels
                from metal_kernels.ops import apply_llama31_rope_pos_ids_inplace

                if layer_idx == 0:
                    print(f"\n  🔧 Before MPS RoPE:")
                    print(f"     q_mps[2,12,0]: {q_mps[2,12,0].item():.6f}")
                    print(f"     q_mps[2,12,32]: {q_mps[2,12,32].item():.6f}")
                    print(f"     q_mps.is_contiguous(): {q_mps.is_contiguous()}")
                    print(f"     q_mps.shape: {q_mps.shape}")
                    print(f"     q_mps.stride(): {q_mps.stride()}")
                    print(f"     q_mps.dtype: {q_mps.dtype}")

                # PROFILED: RoPE operation
                with start_profile("rope"):
                    apply_llama31_rope_pos_ids_inplace(
                        q_mps,
                        k_mps,
                        position_ids_mps,
                        rope_theta=500000.0,
                        rope_scale=32.0,
                        interleave=False,
                    )
                    # Force MPS synchronization
                    torch.mps.synchronize()

        if layer_idx == 0:
            print(f"\n  🔧 After MPS RoPE:")
            print(f"     q_mps[2,12,0]: {q_mps[2,12,0].item():.6f}")
            print(f"     q_mps[2,12,32]: {q_mps[2,12,32].item():.6f}")
            print(
                f"     Changed? {(q_mps[2,12,0] != q_mps_before_rope[2,12,0]).item()}"
            )

        if layer_idx == 0:
            # Compare Q, K after RoPE with HF
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

            inv_freq_hf, _ = ROPE_INIT_FUNCTIONS["llama3"](
                model_hf.config, "mps", seq_len=None
            )
            t = torch.arange(num_tokens, dtype=inv_freq_hf.dtype, device="mps")
            freqs = torch.einsum("n,d->nd", t, inv_freq_hf)
            cos_base = torch.cos(freqs)
            sin_base = torch.sin(freqs)
            cos = torch.cat([cos_base, cos_base], dim=-1)[None, None]
            sin = torch.cat([sin_base, sin_base], dim=-1)[None, None]

            q_hf_check_for_rope = q_hf_check.unsqueeze(0).transpose(1, 2)
            k_hf_check_for_rope = k_hf_check.unsqueeze(0).transpose(1, 2)
            pos_ids = position_ids_mps.unsqueeze(0)

            q_hf_after_rope, k_hf_after_rope = apply_rotary_pos_emb(
                q_hf_check_for_rope, k_hf_check_for_rope, cos, sin, position_ids=pos_ids
            )

            # Squeeze extra dims
            while q_hf_after_rope.dim() > 3:
                q_hf_after_rope = q_hf_after_rope.squeeze(0)
            while k_hf_after_rope.dim() > 3:
                k_hf_after_rope = k_hf_after_rope.squeeze(0)

            q_hf_after_rope = q_hf_after_rope.transpose(0, 1)  # [seq, heads, dim]
            k_hf_after_rope = k_hf_after_rope.transpose(0, 1)

            print(f"\n  DEBUG shapes:")
            print(f"     q_hf_after_rope.shape: {q_hf_after_rope.shape}")
            print(f"     q_mps.shape: {q_mps.shape}")
            print(f"     k_hf_after_rope.shape: {k_hf_after_rope.shape}")
            print(f"     k_mps.shape: {k_mps.shape}")

            print(f"\n  DEBUG values (token 0, head 0, first 4 dims):")
            print(f"     q_hf_after_rope[0,0,:4]: {q_hf_after_rope[0,0,:4]}")
            print(f"     q_mps[0,0,:4]: {q_mps[0,0,:4]}")

            # Calculate diff with detailed debugging
            q_diff_tensor = (q_hf_after_rope - q_mps).abs()
            k_diff_tensor = (k_hf_after_rope - k_mps).abs()

            print(f"\n  DEBUG diff tensor stats:")
            print(f"     q_diff max: {q_diff_tensor.max().item():.6e}")
            print(f"     q_diff mean: {q_diff_tensor.mean().item():.6e}")
            print(f"     q_diff[0,0,:4]: {q_diff_tensor[0,0,:4]}")

            # Find location of max diff
            max_idx = q_diff_tensor.argmax()
            max_token = max_idx // (32 * 64)
            max_head = (max_idx % (32 * 64)) // 64
            max_dim = max_idx % 64
            print(f"     Max diff at token={max_token}, head={max_head}, dim={max_dim}")
            print(
                f"     HF value (after RoPE): {q_hf_after_rope[max_token, max_head, max_dim].item():.6f}"
            )
            print(
                f"     MPS value (after RoPE): {q_mps[max_token, max_head, max_dim].item():.6f}"
            )
            print(
                f"     HF value (before RoPE): {q_hf_check[max_token, max_head, max_dim].item():.6f}"
            )
            print(
                f"     MPS value (before RoPE): {q_mps_before_rope[max_token, max_head, max_dim].item():.6f}"
            )

            q_diff_after_rope = q_diff_tensor.max().item()
            k_diff_after_rope = k_diff_tensor.max().item()

            print(f"\n  🔍 Layer 0 QK after RoPE:")
            print(f"     Q diff: {q_diff_after_rope:.6e}")
            print(f"     K diff: {k_diff_after_rope:.6e}")

        # Update KV cache (manual update for this test, not using append_paged_kv_cache)
        with start_profile("update_kv_cache"):
            for i in range(num_tokens):
                kv_cache_mps[layer_idx][0, 0, i, :, :] = k_mps[i, :, :]
                kv_cache_mps[layer_idx][0, 1, i, :, :] = v_mps[i, :, :]

        # Attention using MPS Metal kernels
        from metal_kernels.ops import BatchPrefillWithPagedKVCacheWrapper

        with start_profile("attention"):
            with start_profile("attention_wrapper_setup"):
                wrapper = BatchPrefillWithPagedKVCacheWrapper(
                    workspace_buffer=torch.empty(0, device="mps"), kv_layout="NHD"
                )
                wrapper.plan(
                    qo_indptr=torch.tensor(
                        [0, num_tokens], dtype=torch.int32, device="mps"
                    ),
                    paged_kv_indptr=torch.tensor(
                        [0, 1], dtype=torch.int32, device="mps"
                    ),
                    paged_kv_indices=torch.tensor([0], dtype=torch.int32, device="mps"),
                    paged_kv_last_page_len=torch.tensor(
                        [num_tokens], dtype=torch.int32, device="mps"
                    ),
                    num_qo_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim_qk=head_dim,
                    page_size=num_tokens,
                )

            with start_profile("attention_kernel_execution"):
                attn_out_mps = wrapper.run(
                    query=q_mps, kv_cache=kv_cache_mps[layer_idx]
                )
                torch.mps.synchronize()

            # Compare attention output with HF (before O projection)
            if layer_idx == 0:
                hf_attn_weights = hf_attentions[layer_idx].squeeze(0)
                print(f"\n  📊 HF attention weights shape: {hf_attn_weights.shape}")

                # Compute HF attention output manually
                v_expanded = v_mps.repeat_interleave(num_heads // num_kv_heads, dim=1)
                v_expanded = v_expanded.transpose(0, 1)

                hf_attn_output = torch.bmm(hf_attn_weights, v_expanded)
                hf_attn_output = hf_attn_output.transpose(0, 1)
                hf_attn_output_flat = hf_attn_output.reshape(
                    num_tokens, num_heads * head_dim
                )

                attn_diff = (hf_attn_output_flat - attn_out_mps).abs().max().item()
                print(f"  📊 Attention output diff (before O proj): {attn_diff:.6e}")

            # O projection and residual
            with start_profile("o_projection"):
                o_out_mps = layer_mps.self_attn.o_proj(attn_out_mps)
                hidden_mps_after_attn = residual_mps + o_out_mps

            # Post-attention norm and MLP
            residual_mps_2 = hidden_mps_after_attn

            with start_profile("post_attn_norm"):
                hidden_mps_norm2 = layer_mps.post_attention_layernorm(
                    hidden_mps_after_attn
                )

            with start_profile("mlp"):
                mlp_out_mps = layer_mps.mlp(hidden_mps_norm2)

            hidden_mps = residual_mps_2 + mlp_out_mps

            # Compare outputs
            if layer_idx == 15:
                print(f"  NOTE: Layer 15 output from HF has final norm applied")
                hidden_mps_for_comparison = model_mps.model.norm(hidden_mps)
                print(f"  Comparing norm(hidden_mps) vs hidden_hf_output")
            else:
                hidden_mps_for_comparison = hidden_mps

            output_diff = (
                (hidden_hf_output - hidden_mps_for_comparison).abs().max().item()
            )
            mean_diff = (
                (hidden_hf_output - hidden_mps_for_comparison).abs().mean().item()
            )
            print(f"  Output diff (max): {output_diff:.6e}")
            print(f"  Output diff (mean): {mean_diff:.6e}")

            # Check per-token differences
            token_diffs = (
                (hidden_hf_output - hidden_mps_for_comparison).abs().max(dim=-1).values
            )
            token_0_diff = token_diffs[0].item()
            token_last_diff = token_diffs[-1].item()
            print(f"  Token 0 max diff: {token_0_diff:.6e}")
            print(f"  Token {num_tokens-1} max diff: {token_last_diff:.6e}")

            if output_diff > 1e-5:
                print(f"  ⚠️  Significant divergence detected in layer {layer_idx}!")
                print(f"  HF output sample [0,:5]: {hidden_hf_output[0,:5]}")
                print(f"  MPS output sample [0,:5]: {hidden_mps_for_comparison[0,:5]}")

    # Final norm
    print(f"\n--- Final Norm ---")
    hidden_hf_final = hf_hidden_states[-1].squeeze(0)
    hidden_mps_final = model_mps.model.norm(hidden_mps)

    norm_diff = (hidden_hf_final - hidden_mps_final).abs().max().item()
    print(f"  Diff: {norm_diff:.6e}")

    print(f"\n--- LM Head ---")
    logits_hf = hf_outputs.logits.squeeze(0)
    logits_mps = model_mps.lm_head(hidden_mps_final)

    logits_diff = (logits_hf - logits_mps).abs().max().item()
    print(f"  Diff: {logits_diff:.6e}")

    # Final analysis
    print("\n" + "=" * 80)
    print("FINAL ANALYSIS")
    print("=" * 80)

    print(f"\nFinal logits diff: {logits_diff:.6e}")

    if logits_diff < 0.01:
        print("✅ Error accumulation is TOLERABLE (< 0.01)")
    elif logits_diff < 0.1:
        print("⚠️  Error accumulation is BORDERLINE (0.01 - 0.1)")
    else:
        print("❌ Error accumulation is NOT TOLERABLE (> 0.1)")

# Print profiling results
print("\n" + "=" * 80)
print("PROFILING RESULTS")
print("=" * 80)

# Save profiling data using unified profiler
tracker = get_memory_tracker()
profile_file = tracker.save_snapshot()
print(f"Profiling data saved to: {profile_file}")
print("Use the profile visualizer to view detailed timing breakdown.")
