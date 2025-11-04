#!/usr/bin/env python3
"""
Layer-by-layer divergence test with MULTIPLE PAGE CACHE scenarios.

Tests various paging configurations:
1. Single page (9 tokens in page_size=16)
2. Exact page boundary (16 tokens in page_size=16)
3. Two pages (17 tokens in page_size=16)
4. Multiple pages (50 tokens in page_size=16)
"""

import sys
import os
from pathlib import Path

# Add backend-python to path (we're in tests/metal_kernels_debug/)
workspace_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(workspace_dir / "backend" / "backend-python"))

os.environ["PIE_METAL_PYTORCH_MODE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

cache_dir = os.path.expanduser("~/.cache/pie")


def test_scenario(
    num_tokens, page_size=16, model_hf=None, model_pt=None, tokenizer=None
):
    """Test a specific number of tokens with given page size."""

    print("\n" + "=" * 80)
    print(f"SCENARIO: {num_tokens} tokens, page_size={page_size}")
    print("=" * 80)

    num_pages = (num_tokens + page_size - 1) // page_size
    last_page_len = num_tokens % page_size if num_tokens % page_size != 0 else page_size

    print(f"  Total tokens: {num_tokens}")
    print(f"  Page size: {page_size}")
    print(f"  Num pages: {num_pages}")
    print(f"  Last page len: {last_page_len}")

    # Generate token sequence
    # Use a simple repeating pattern for testing
    test_tokens = [791, 6864, 315, 9822, 374] * (num_tokens // 5 + 1)
    test_tokens = test_tokens[:num_tokens]

    print(f"  Token sequence: {test_tokens[:10]}{'...' if num_tokens > 10 else ''}")

    # Get embeddings
    input_ids = torch.tensor(test_tokens, dtype=torch.long)

    with torch.no_grad():
        hidden_hf = model_hf.model.embed_tokens(input_ids)
        hidden_pt = model_pt.model.embed_tokens(input_ids)

        embed_diff = (hidden_hf - hidden_pt).abs().max().item()
        print(f"  Embeddings diff: {embed_diff:.6e}")

        # Process each layer
        from metal_kernels._internal.pytorch_reference import (
            rope_reference,
            attention_reference,
        )

        num_heads = 32
        num_kv_heads = 8
        head_dim = 64

        # Setup for PT with PAGED cache
        position_ids_pt = torch.arange(num_tokens, dtype=torch.long)
        kv_cache_pt = []
        for _ in range(16):
            # PAGED KV cache: [num_pages, 2, page_size, num_kv_heads, head_dim]
            cache_layer = torch.zeros(
                num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float32
            )
            kv_cache_pt.append(cache_layer)

        # KV paging parameters
        kv_page_indices = torch.arange(num_pages, dtype=torch.int32)
        kv_page_indptr = torch.tensor([0, num_pages], dtype=torch.int32)
        kv_last_page_lens = torch.tensor([last_page_len], dtype=torch.int32)
        qo_indptr = torch.tensor([0, num_tokens], dtype=torch.int32)

        if num_tokens in [17, 25]:  # Debug failing cases
            print(f"\n  📍 Debug paging params:")
            print(f"     kv_page_indices: {kv_page_indices.tolist()}")
            print(f"     kv_page_indptr: {kv_page_indptr.tolist()}")
            print(f"     kv_last_page_lens: {kv_last_page_lens.tolist()}")
            print(f"     qo_indptr: {qo_indptr.tolist()}")

        # Get HF full output
        hf_outputs = model_hf(
            input_ids.unsqueeze(0),
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
        )
        hf_hidden_states = hf_outputs.hidden_states
        hf_attentions = hf_outputs.attentions

        # Track divergence
        max_layer_diff = 0.0
        first_diverge_layer = None

        for layer_idx in range(16):
            layer_pt = model_pt.model.layers[layer_idx]

            # Get HF's INPUT and OUTPUT
            hidden_hf_input = hf_hidden_states[layer_idx].squeeze(0)
            hidden_hf_output = hf_hidden_states[layer_idx + 1].squeeze(0)

            # Check input
            input_diff = (hidden_hf_input - hidden_pt).abs().max().item()

            # PT: Manual processing
            residual_pt = hidden_pt
            hidden_pt_norm = layer_pt.input_layernorm(hidden_pt)

            # QKV projection
            qkv_pt = layer_pt.self_attn.qkv_proj(hidden_pt_norm)
            q_pt, k_pt, v_pt = torch.split(qkv_pt, [32 * 64, 8 * 64, 8 * 64], dim=-1)

            # Reshape
            q_pt = q_pt.view(num_tokens, num_heads, head_dim)
            k_pt = k_pt.view(num_tokens, num_kv_heads, head_dim)
            v_pt = v_pt.view(num_tokens, num_kv_heads, head_dim)

            # Apply RoPE
            rope_theta = 500000.0
            q_pt = rope_reference(
                q_pt,
                position_ids_pt,
                rope_theta=rope_theta,
                rope_factor=32.0,
                interleaved=False,
                inplace=False,
                low_freq_factor=1.0,
                high_freq_factor=4.0,
                old_context_len=8192,
            )
            k_pt = rope_reference(
                k_pt,
                position_ids_pt,
                rope_theta=rope_theta,
                rope_factor=32.0,
                interleaved=False,
                inplace=False,
                low_freq_factor=1.0,
                high_freq_factor=4.0,
                old_context_len=8192,
            )

            # Manually populate KV cache across pages
            # The cache has shape: [num_pages, 2, page_size, num_kv_heads, head_dim]
            # We need to write tokens across pages correctly
            for token_idx in range(num_tokens):
                page_idx = token_idx // page_size
                slot_idx = token_idx % page_size
                kv_cache_pt[layer_idx][page_idx, 0, slot_idx, :, :] = k_pt[
                    token_idx, :, :
                ]
                kv_cache_pt[layer_idx][page_idx, 1, slot_idx, :, :] = v_pt[
                    token_idx, :, :
                ]

            # Debug: Check cache population for failing cases
            if layer_idx == 0 and num_tokens in [17, 25]:
                print(f"\n  📍 Layer 0 cache check:")
                print(
                    f"     Page 0, slot 15 (last of page 1): k norm = {kv_cache_pt[0][0, 0, 15, :, :].norm().item():.4f}"
                )
                print(
                    f"     Page 1, slot 0 (first of page 2): k norm = {kv_cache_pt[0][1, 0, 0, :, :].norm().item():.4f}"
                )
                if num_tokens >= 25:
                    print(
                        f"     Page 1, slot 8 (9th token in page 2): k norm = {kv_cache_pt[0][1, 0, 8, :, :].norm().item():.4f}"
                    )

            # Attention
            attn_out_pt = attention_reference(
                q_pt,
                kv_cache_pt[layer_idx],
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_lens,
                qo_indptr,
            )

            # O projection
            o_out_pt = layer_pt.self_attn.o_proj(attn_out_pt)

            # Residual
            hidden_pt = residual_pt + o_out_pt

            # MLP
            residual_pt = hidden_pt
            hidden_pt_norm = layer_pt.post_attention_layernorm(hidden_pt)
            mlp_out_pt = layer_pt.mlp(hidden_pt_norm)
            hidden_pt = residual_pt + mlp_out_pt

            # Compare output
            # IMPORTANT: For layer 15, hidden_hf_output has norm applied, but hidden_pt doesn't!
            if layer_idx == 15:
                hidden_pt_for_comparison = model_pt.model.norm(hidden_pt)
            else:
                hidden_pt_for_comparison = hidden_pt

            output_diff = (
                (hidden_hf_output - hidden_pt_for_comparison).abs().max().item()
            )

            if output_diff > max_layer_diff:
                max_layer_diff = output_diff

            if output_diff > 1e-5 and first_diverge_layer is None:
                first_diverge_layer = layer_idx

            if layer_idx < 3 or output_diff > 1e-5:
                print(
                    f"  Layer {layer_idx:2d}: input_diff={input_diff:.6e}, output_diff={output_diff:.6e}"
                )

        # Final norm and logits
        # NOTE: hf_hidden_states[-1] already has norm applied!
        hidden_pt_final = model_pt.model.norm(hidden_pt)
        hidden_hf_final = hf_hidden_states[-1].squeeze(0)  # Already has norm applied

        norm_diff = (hidden_hf_final - hidden_pt_final).abs().max().item()

        # IMPORTANT: Use HF's forward pass logits, not manually computed
        # HF computes logits during forward(), not from manual lm_head call
        logits_pt = model_pt.lm_head(hidden_pt_final)
        logits_hf = hf_outputs.logits.squeeze(0)  # Use HF's forward pass logits

        logits_diff = (logits_hf - logits_pt).abs().max().item()

        print(f"\n  Final norm diff: {norm_diff:.6e}")
        print(f"  Logits diff: {logits_diff:.6e}")

        # Summary
        # Accept if either all layers match OR logits match (with tolerance for float32)
        logits_threshold = 1e-4  # Relaxed threshold for float32 precision

        if max_layer_diff < 1e-5:
            print(f"\n  ✅ PASS: All layers match (max diff: {max_layer_diff:.6e})")
            return True
        elif logits_diff < logits_threshold:
            print(
                f"\n  ⚠️  Layer divergence: {max_layer_diff:.6e} (first at layer {first_diverge_layer if first_diverge_layer else 'N/A'})"
            )
            print(
                f"      ✅ But logits match within tolerance! (diff: {logits_diff:.6e} < {logits_threshold})"
            )
            return True
        else:
            print(f"\n  ❌ DIVERGENCE: Max layer diff: {max_layer_diff:.6e}")
            if first_diverge_layer is not None:
                print(f"      First divergence at layer {first_diverge_layer}")
            print(
                f"      Logits differ by {logits_diff:.6e} (threshold: {logits_threshold})"
            )
            return False


def main():
    print("=" * 80)
    print("MULTI-PAGE LAYER-BY-LAYER TEST")
    print("=" * 80)

    # Load models once
    print("\nLoading models...")
    model_hf = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        cache_dir=cache_dir,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model_hf = model_hf.to("cpu").to(torch.float32)
    model_hf.eval()

    from common import load_model as load_model_common, build_config
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
        device="cpu",
        dtype="float32",
    )

    model_pt, _ = load_model_common(config, create_model_and_fusion_map)
    model_pt = model_pt.to(dtype=torch.float32)
    model_pt.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", cache_dir=cache_dir
    )

    print("✅ Models loaded")

    # Test scenarios
    scenarios = [
        (9, 16, "Single page - 9 tokens in 16-slot page"),
        (16, 16, "Exact boundary - 16 tokens fills one page"),
        (17, 16, "Two pages - 17 tokens needs 2 pages"),
        (25, 16, "Two pages - 25 tokens (page 1 full, page 2 partial)"),
        (32, 16, "Exact 2 pages - 32 tokens fills 2 pages"),
        (50, 16, "Multiple pages - 50 tokens needs 4 pages"),
    ]

    results = []
    for num_tokens, page_size, description in scenarios:
        print(f"\n{'='*80}")
        print(f"TEST: {description}")
        passed = test_scenario(num_tokens, page_size, model_hf, model_pt, tokenizer)
        results.append((description, passed))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for desc, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {desc}")

    all_passed = all(p for _, p in results)
    if all_passed:
        print("\n🎉 All scenarios passed!")
    else:
        print("\n⚠️  Some scenarios failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
