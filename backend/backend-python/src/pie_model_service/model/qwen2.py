from __future__ import annotations

import dataclasses
from typing import Optional

import torch
import torch.nn.functional as fun

from src.pie_model_service.adapter import AdapterSubpass
from src.pie_model_service.utils import is_apple_silicon

if is_apple_silicon():
    import flashinfer_metal as ops  # type: ignore[import-not-found]
else:
    import flashinfer as ops  # type: ignore[import-not-found,no-redef]


@dataclasses.dataclass
class Config:
    type: str
    num_layers: int
    num_query_heads: int
    num_key_value_heads: int
    head_size: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    use_qkv_bias: bool
    rms_norm_eps: float
    rope_theta: float

    @staticmethod
    def from_dict(config: dict) -> Config:
        return Config(
            type=config["type"],
            num_layers=int(config["num_layers"]),
            num_query_heads=int(config["num_query_heads"]),
            num_key_value_heads=int(config["num_key_value_heads"]),
            head_size=int(config["head_size"]),
            hidden_size=int(config["hidden_size"]),
            intermediate_size=int(config["intermediate_size"]),
            vocab_size=int(config["vocab_size"]),
            rms_norm_eps=float(config["rms_norm_eps"]),
            rope_theta=float(config["rope"]["theta"]),
        )


def prepare_params(
    params: dict[str, torch.Tensor],
):
    """
    Modifies the weights dictionary in-place to fuse tensors.

    This function applies the fusion rules derived from the v1 model structure:
    1. Fuses Q, K, V weights and biases into a single 'qkv_proj' tensor.
    2. Fuses 'gate_proj' and 'up_proj' weights into 'gate_up_proj'.
    """
    # Find all layer indices present in the weights
    layer_indices = set()
    for key in params.keys():
        parts = key.split(".")
        # Check for 'model.layers.{idx}.*'
        if len(parts) > 3 and parts[0] == "model" and parts[1] == "layers":
            layer_indices.add(int(parts[2]))

    for layer_idx in sorted(list(layer_indices)):
        prefix = f"model.layers.{layer_idx}"

        # Fuse QKV weights
        q = params.pop(f"{prefix}.self_attn.q_proj.weight")
        k = params.pop(f"{prefix}.self_attn.k_proj.weight")
        v = params.pop(f"{prefix}.self_attn.v_proj.weight")
        params[f"{prefix}.self_attn.qkv_proj.weight"] = torch.cat([q, k, v], dim=0)

        # Qwen2 uses QKV bias, so fuse them
        q_bias = params.pop(f"{prefix}.self_attn.q_proj.bias")
        k_bias = params.pop(f"{prefix}.self_attn.k_proj.bias")
        v_bias = params.pop(f"{prefix}.self_attn.v_proj.bias")
        params[f"{prefix}.self_attn.qkv_proj.bias"] = torch.cat(
            [q_bias, k_bias, v_bias], dim=0
        )

        # Fuse MLP weights
        gate = params.pop(f"{prefix}.mlp.gate_proj.weight")
        up = params.pop(f"{prefix}.mlp.up_proj.weight")
        params[f"{prefix}.mlp.gate_up_proj.weight"] = torch.cat([gate, up], dim=0)

        # Qwen2 MLP does not use bias, so no bias fusion is needed here.


class ForwardPass:

    workspace_buffer: torch.Tensor
    wrapper_decode: ops.BatchDecodeWithPagedKVCacheWrapper
    wrapper_append: ops.BatchPrefillWithPagedKVCacheWrapper

    def __init__(self, device: torch.device):
        self.workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        self.wrapper_decode = ops.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )
        self.wrapper_append = ops.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )

    @torch.inference_mode()
    def execute(
        self,
        config: Config,
        params: dict[str, torch.Tensor],
        # inputs
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        qo_indptr: torch.Tensor,
        # kv cache
        kv_cache_at_layer: list[torch.Tensor],
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        # mask
        custom_mask: torch.Tensor | None,
        single_token_inference_mode: bool,
        # subpasses
        adapter_subpass: Optional[AdapterSubpass],
    ):
        hidden_states = input_embeds
        n, _ = hidden_states.size()

        page_size = int(kv_cache_at_layer[0].shape[2])

        seq_lens = ops.get_seq_lens(
            kv_page_indptr,
            kv_last_page_lens,
            page_size,
        )

        batch_indices, batch_positions = ops.get_batch_indices_positions(
            append_indptr=qo_indptr,
            seq_lens=seq_lens,
            nnz=n,
        )
        del seq_lens  # No longer needed

        # Per the original Qwen2 implementation, FlashInfer's decode wrapper
        # had issues with certain Q/KV head ratios.
        # We preserve the logic of always using the append wrapper.
        _ = single_token_inference_mode
        wrapper = self.wrapper_append
        wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_page_indptr,
            paged_kv_indices=kv_page_indices,
            paged_kv_last_page_len=kv_last_page_lens,
            num_qo_heads=config.num_query_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim_qk=config.head_size,
            page_size=page_size,
            custom_mask=custom_mask,
            q_data_type=config.dtype,
        )

        for layer_idx in range(config.num_layers):

            # Save input for the first residual connection
            residual = hidden_states

            # 1. Input RMSNorm
            normed_input = fun.rms_norm(
                hidden_states,
                normalized_shape=[config.hidden_size],
                weight=params[f"model.layers.{layer_idx}.input_layernorm.weight"],
                eps=config.rms_norm_eps,
            )

            ### BEGIN SELF_ATTN
            # 2. QKV Projection
            qkv_proj = fun.linear(
                normed_input,
                weight=params[f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"],
                bias=params.get(f"model.layers.{layer_idx}.self_attn.qkv_proj.bias"),
            )
            q, k, v = torch.split(
                qkv_proj,
                [
                    config.num_query_heads * config.head_size,
                    config.num_key_value_heads * config.head_size,
                    config.num_key_value_heads * config.head_size,
                ],
                dim=-1,
            )
            del qkv_proj  # Free fused tensor immediately after split

            # 3. Adapter (if any)
            if adapter_subpass is not None:
                adapter_subpass.execute(
                    layer_idx,
                    normed_input,  # Adapter needs the normed input
                    q_state=q,
                    k_state=k,
                    v_state=v,
                )
            # We're done with the normed input for this layer
            del normed_input

            # 4. Reshape QKV
            q = q.view(n, config.num_query_heads, config.head_size)
            k = k.view(n, config.num_key_value_heads, config.head_size)
            v = v.view(n, config.num_key_value_heads, config.head_size)

            # 5. Apply RoPE (in-place)
            ops.apply_rope_pos_ids_inplace(
                q=q,
                k=k,
                pos_ids=position_ids,
                rope_theta=config.rope_theta,
            )

            # 6. Append K, V to cache
            ops.append_paged_kv_cache(
                append_key=k,
                append_value=v,
                batch_indices=batch_indices,
                positions=batch_positions,
                paged_kv_cache=kv_cache_at_layer[layer_idx],
                kv_indices=kv_page_indices,
                kv_indptr=kv_page_indptr,
                kv_last_page_len=kv_last_page_lens,
                kv_layout="NHD",
            )
            # K and V are now in the cache, no longer needed in flight
            del k, v

            # 7. Compute Attention
            attn_output = wrapper.run(q, kv_cache_at_layer[layer_idx])
            del q  # Q is no longer needed

            attn_output = attn_output.reshape(n, -1)

            # 8. Output Projection
            attn_proj = fun.linear(
                attn_output,
                weight=params[f"model.layers.{layer_idx}.self_attn.o_proj.weight"],
                bias=params.get(f"model.layers.{layer_idx}.self_attn.o_proj.bias"),
            )
            del attn_output
            ### END SELF_ATTN

            # 9. First Residual Connection
            hidden_states = residual + attn_proj
            # Free the original input and the attention projection
            del residual, attn_proj

            # Save result of first add for the second residual connection
            residual = hidden_states

            # 10. Post-Attention RMSNorm
            normed_mlp_input = fun.rms_norm(
                hidden_states,
                normalized_shape=[config.hidden_size],
                weight=params[
                    f"model.layers.{layer_idx}.post_attention_layernorm.weight"
                ],
                eps=config.rms_norm_eps,
            )

            ### BEGIN MLP
            # 11. Gate/Up Projection
            gate_up = fun.linear(
                normed_mlp_input,
                weight=params[f"model.layers.{layer_idx}.mlp.gate_up_proj.weight"],
                bias=params.get(f"model.layers.{layer_idx}.mlp.gate_up_proj.bias"),
            )
            # Done with normed input for MLP
            del normed_mlp_input

            gate, up = gate_up.chunk(2, dim=-1)
            del gate_up  # Free fused tensor

            # 12. SiLU Activation & Gating
            interim_state = fun.silu(gate) * up
            del gate, up

            # 13. Down Projection
            mlp_output = fun.linear(
                interim_state,
                weight=params[f"model.layers.{layer_idx}.mlp.down_proj.weight"],
                bias=params.get(f"model.layers.{layer_idx}.mlp.down_proj.bias"),
            )
            del interim_state
            ### END MLP

            # 14. Second Residual Connection
            hidden_states = residual + mlp_output
            # Free the mid-block tensor and the mlp output
            del residual, mlp_output

        # Final RMSNorm after all layers
        hidden_states = fun.rms_norm(
            hidden_states,
            normalized_shape=[config.hidden_size],
            weight=params["model.norm.weight"],
            eps=config.rms_norm_eps,
        )

        return hidden_states
