from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Any

import torch
import torch.nn.functional as fun
import torch.distributed as dist  # Import torch.distributed

from . import Spec as SpecBase, Param as ParamBase, Buffer as BufferBase, Config

from ..adapter import AdapterSubpass
from ..utils import is_apple_silicon, get_available_memory

if is_apple_silicon():
    import flashinfer_metal as ops  # type: ignore[import-not-found]
else:
    import flashinfer as ops  # type: ignore[import-not-found,no-redef]


def _shard_column(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Shards a tensor for column-parallel linear layer (splits dim 0)."""
    # Ensure tensor is contiguous before chunking
    return torch.chunk(tensor.contiguous(), world_size, dim=0)[rank]


def _shard_row(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Shards a tensor for row-parallel linear layer (splits dim 1)."""
    # Ensure tensor is contiguous before chunking
    return torch.chunk(tensor.contiguous(), world_size, dim=1)[rank]


def _eval_max_num_kv_pages(
    spec: Spec,
    config: Config,
) -> int:

    available_bytes = get_available_memory(
        devices=config.devices,
        rank=config.rank,
    )
    usable_bytes = available_bytes * config.mem_utilization
    element_size_bytes = torch.empty((), dtype=config.dtype).element_size()
    total_bytes_per_page = (
        element_size_bytes
        * 2
        * config.kv_page_size
        * spec.num_kv_heads
        * spec.dim_head
        * spec.num_layers
    )

    max_num_pages = int(usable_bytes // total_bytes_per_page)
    return max_num_pages


@dataclass
class Spec(SpecBase):

    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    num_vocabs: int

    dim_head: int
    dim_hidden: int
    dim_mlp: int

    rms_norm_eps: float

    rope_factor: float
    rope_high_frequency_factor: float
    rope_low_frequency_factor: float
    rope_theta: float

    @staticmethod
    def from_dict(spec: dict) -> Spec:
        return Spec(
            num_layers=int(spec["num_layers"]),
            num_q_heads=int(spec["num_query_heads"]),
            num_kv_heads=int(spec["num_key_value_heads"]),
            dim_head=int(spec["head_size"]),
            dim_hidden=int(spec["hidden_size"]),
            dim_mlp=int(spec["intermediate_size"]),
            num_vocabs=int(spec["vocab_size"]),
            rms_norm_eps=float(spec["rms_norm_eps"]),
            rope_factor=float(spec["rope"]["factor"]),
            rope_high_frequency_factor=float(spec["rope"]["high_frequency_factor"]),
            rope_low_frequency_factor=float(spec["rope"]["low_frequency_factor"]),
            rope_theta=float(spec["rope"]["theta"]),
        )


@dataclass
class Param(ParamBase):

    spec: Spec
    config: Config

    # linear weights
    proj_down: list[torch.Tensor]
    proj_gate_up: list[torch.Tensor]
    proj_qkv: list[torch.Tensor]
    proj_o: list[torch.Tensor]

    # embedding weight
    embed_token: torch.Tensor

    # norm weights
    norm_attn: list[torch.Tensor]
    norm_mlp: list[torch.Tensor]
    norm_last: torch.Tensor

    @staticmethod
    def from_reader(
        spec: Spec,
        config: Config,
        read: Callable[..., torch.Tensor],
    ) -> Param:

        shardable_dims = {
            "vocab_size": spec.num_vocabs,
            "num_query_heads": spec.num_q_heads,
            "num_key_value_heads": spec.num_kv_heads,
            "intermediate_size": spec.dim_mlp,
        }
        for name, dim in shardable_dims.items():
            if dim % config.world_size != 0:
                raise ValueError(
                    f"Config dimension {name} ({dim}) is not divisible "
                    f"by world_size ({config.world_size})."
                )

        norm_attn = []
        norm_mlp = []
        proj_down = []
        proj_gate_up = []
        proj_qkv = []
        proj_o = []

        embed_w = read(
            "model.embed_tokens.weight",
            expected_shape=(spec.num_vocabs, spec.dim_hidden),
        )
        embed_token = _shard_column(embed_w, config.rank, config.world_size).to(
            config.device
        )

        for i in range(spec.num_layers):
            prefix = f"model.layers.{i}"

            norm_attn.append(
                read(
                    f"{prefix}.input_layernorm.weight",
                    expected_shape=(spec.dim_hidden,),
                ).to(config.device)
            )
            norm_mlp.append(
                read(
                    f"{prefix}.post_attention_layernorm.weight",
                    expected_shape=(spec.dim_hidden,),
                ).to(config.device)
            )

            q_proj_w = read(
                f"{prefix}.self_attn.q_proj.weight",
                expected_shape=(spec.dim_hidden, spec.dim_hidden),
            )
            k_proj_w = read(
                f"{prefix}.self_attn.k_proj.weight",
                expected_shape=(
                    spec.num_kv_heads * spec.dim_head,
                    spec.dim_hidden,
                ),
            )
            v_proj_w = read(
                f"{prefix}.self_attn.v_proj.weight",
                expected_shape=(
                    spec.num_kv_heads * spec.dim_head,
                    spec.dim_hidden,
                ),
            )

            fused_qkv = torch.cat([q_proj_w, k_proj_w, v_proj_w], dim=0)
            proj_qkv.append(
                _shard_column(fused_qkv, config.rank, config.world_size).to(
                    config.device
                )
            )

            o_proj_w = read(
                f"{prefix}.self_attn.o_proj.weight",
                expected_shape=(spec.dim_hidden, spec.dim_hidden),
            )
            proj_o.append(
                _shard_row(o_proj_w, config.rank, config.world_size).to(config.device)
            )

            gate_proj_w = read(
                f"{prefix}.mlp.gate_proj.weight",
                expected_shape=(spec.dim_mlp, spec.dim_hidden),
            )
            up_proj_w = read(
                f"{prefix}.mlp.up_proj.weight",
                expected_shape=(spec.dim_mlp, spec.dim_hidden),
            )

            fused_gate_up = torch.cat([gate_proj_w, up_proj_w], dim=0)
            proj_gate_up.append(
                _shard_column(fused_gate_up, config.rank, config.world_size).to(
                    config.device
                )
            )

            down_proj_w = read(
                f"{prefix}.mlp.down_proj.weight",
                expected_shape=(spec.dim_hidden, spec.dim_mlp),
            )
            proj_down.append(
                _shard_row(down_proj_w, config.rank, config.world_size).to(
                    config.device
                )
            )

        final_norm_w = read("model.norm.weight", expected_shape=(spec.dim_hidden,))
        final_norm = final_norm_w.to(config.device)

        return Param(
            spec=spec,
            config=config,
            proj_down=proj_down,
            proj_gate_up=proj_gate_up,
            proj_qkv=proj_qkv,
            proj_o=proj_o,
            embed_token=embed_token,
            norm_attn=norm_attn,
            norm_mlp=norm_mlp,
            norm_last=final_norm,
        )


@dataclass
class Buffer(BufferBase):

    spec: Spec
    config: Config

    kv_cache: list[torch.Tensor]
    adapter: list[tuple[torch.Tensor, torch.Tensor]]

    decode_attn: ops.BatchDecodeWithPagedKVCacheWrapper
    append_attn: ops.BatchPrefillWithPagedKVCacheWrapper

    @staticmethod
    def from_config(spec: Spec, config: Config) -> Buffer:

        workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=config.device
        )
        decode_attn = ops.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD")
        append_attn = ops.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")

        adapter = [
            (
                torch.zeros(
                    (
                        config.max_num_adapters,
                        config.max_adapter_rank * 3,
                        spec.dim_hidden,
                    ),
                    dtype=config.dtype,
                    device=config.device,
                ),
                torch.zeros(
                    (
                        config.max_num_adapters,
                        spec.dim_head * (spec.num_q_heads + spec.num_kv_heads * 2),
                        config.max_adapter_rank,
                    ),
                    dtype=config.dtype,
                    device=config.device,
                ),
            )
            for _ in range(spec.num_layers)
        ]

        # update config.max_num_kv_pages
        config.max_num_kv_pages = _eval_max_num_kv_pages(spec, config)

        kv_cache = [
            torch.zeros(
                (
                    config.max_num_kv_pages,
                    2,
                    config.kv_page_size,
                    spec.num_kv_heads,
                    spec.dim_head,
                ),
                dtype=config.dtype,
                device=config.device,
            )
            for _ in range(spec.num_layers)
        ]

        return Buffer(
            spec=spec,
            config=config,
            adapter=adapter,
            kv_cache=kv_cache,
            decode_attn=decode_attn,
            append_attn=append_attn,
        )


class ForwardPass:

    device: torch.device

    workspace_buffer: torch.Tensor
    wrapper_decode: ops.BatchDecodeWithPagedKVCacheWrapper
    wrapper_append: ops.BatchPrefillWithPagedKVCacheWrapper

    def __init__(self, device: torch.device):

        self.device = device

        self.workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        self.wrapper_decode = ops.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )
        self.wrapper_append = ops.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )

    def embed_tokens(
        self,
        param: Param,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:

        local_vocab_size = param.spec.num_vocabs // param.world_size
        vocab_start_index = param.rank * local_vocab_size
        vocab_end_index = (param.rank + 1) * local_vocab_size

        # Map global token_ids to local_ids for this rank's embedding table
        local_ids = token_ids - vocab_start_index
        mask = (token_ids >= vocab_start_index) & (token_ids < vocab_end_index)

        # Zero out IDs that are not in this rank's shard
        # (padding_idx=0 in fun.embedding will handle index 0)
        local_ids[~mask] = 0

        local_embeds = fun.embedding(
            input=local_ids,
            weight=param.embed_token,
            padding_idx=0,
        )

        # Zero out embeddings from tokens that didn't belong to this shard
        local_embeds[~mask.unsqueeze(-1)] = 0.0
        dist.all_reduce(local_embeds)

        return local_embeds

    def sample(
        self,
        param: Param,
        hidden_states: torch.Tensor,
        indices_for_logits: torch.Tensor,
        temperatures: torch.Tensor,
        sampler_groups: dict,
        sampler_params: dict,
    ) -> torch.Tensor:
        """
        Performs the final linear projection (LM head) and gathers
        logits from all ranks to restore the full distribution.

        Input hidden_states are replicated.
        Weight is sharded [local_vocab_size, hidden_size].
        Output local_logits are sharded [batch_size, local_vocab_size].
        This function gathers them into [batch_size, vocab_size].
        """

        hidden_states = hidden_states[indices_for_logits]

        # Final RMSNorm after all layers (replicated)
        hidden_states = fun.rms_norm(
            hidden_states,
            normalized_shape=[param.spec.dim_hidden],
            weight=param.norm_last,
            eps=param.spec.rms_norm_eps,
        )

        # --- 1. Calculate Local Logits ---
        # This is a [batch_size, local_vocab_size] tensor because
        # param.token_embed is sharded along the vocab dimension.
        #
        local_logits = fun.linear(
            input=hidden_states,
            weight=param.embed_token,
        )

        # --- 2. Prepare for All-Gather ---
        # If we are not in a distributed setting, we can just return.
        if param.world_size == 1:
            logits = local_logits
        else:
            # Get shape info for the full tensor
            batch_size, _ = local_logits.shape
            full_vocab_size = param.spec.num_vocabs

            # Create the full output tensor on each rank.
            # This tensor will receive the gathered data.
            logits = torch.empty(
                batch_size,
                full_vocab_size,
                dtype=local_logits.dtype,
                device=local_logits.device,
            )

            # --- 3. Perform All-Gather ---
            # This operation gathers all `local_logits` tensors (one from each rank)
            # and concatenates them along dim=1 into `full_logits`.
            # The `full_logits` tensor will be identical on all ranks after this call.
            dist.all_gather_into_tensor(
                output_tensor=logits,
                input_tensor=local_logits.contiguous(),  # .contiguous() is good practice
            )

        scaled_logits = logits / torch.clamp(temperatures, min=1e-6)
        probs = torch.softmax(scaled_logits, dim=-1)

        if not torch.isfinite(probs).all():
            raise RuntimeError("Non-finite probabilities produced by LM head")

        num_logit_requests = len(indices_for_logits)

        final_dists: list[tuple[list[int], list[float]] | None] = [
            None
        ] * num_logit_requests

        final_tokens_tensor = torch.empty(
            num_logit_requests, dtype=torch.long, device=probs.device
        )

        for sampler_idx, indices in sampler_groups.items():
            if not indices:
                continue

            indices_tensor = torch.tensor(
                indices, device=probs.device, dtype=torch.long
            )
            group_probs = probs.index_select(0, indices_tensor)

            # Handle distributions (sampler_idx=0)
            if sampler_idx == 0:
                group_top_k = [sampler_params[i]["top_k"] for i in indices]
                max_k = max(group_top_k) if group_top_k else 0
                if max_k > 0:
                    topk_vals, topk_inds = torch.topk(group_probs, k=max_k, sorted=True)
                    for i, original_idx in enumerate(indices):
                        k = sampler_params[original_idx]["top_k"]
                        ids = topk_inds[i, :k].tolist()
                        vals = topk_vals[i, :k].tolist()
                        final_dists[original_idx] = (ids, vals)

            # Handle sampling operations (sampler_idx > 0)
            else:
                if sampler_idx == 1:  # Old 0: sampling_from_probs
                    sampled = ops.sampling.sampling_from_probs(group_probs)
                elif sampler_idx == 2:  # Old 1: top_p_sampling_from_probs
                    top_p_vals = torch.tensor(
                        [sampler_params[i]["top_p"] for i in indices],
                        device=probs.device,
                        dtype=probs.dtype,
                    )
                    sampled = ops.sampling.top_p_sampling_from_probs(
                        group_probs, top_p=top_p_vals
                    )
                elif sampler_idx == 3:  # Old 2: top_k_sampling_from_probs
                    top_k_vals = torch.tensor(
                        [sampler_params[i]["top_k"] for i in indices],
                        device=probs.device,
                        dtype=torch.long,
                    )
                    sampled = ops.sampling.top_k_sampling_from_probs(
                        group_probs, top_k=top_k_vals
                    )
                elif sampler_idx == 4:  # Old 3: min_p_sampling_from_probs
                    min_p_vals = torch.tensor(
                        [sampler_params[i]["min_p"] for i in indices],
                        device=probs.device,
                        dtype=probs.dtype,
                    )
                    sampled = ops.sampling.min_p_sampling_from_probs(
                        group_probs, min_p=min_p_vals
                    )
                elif sampler_idx == 5:  # Old 4: top_k_top_p_sampling_from_probs
                    top_k_vals = torch.tensor(
                        [sampler_params[i]["top_k"] for i in indices],
                        device=probs.device,
                        dtype=torch.long,
                    )
                    top_p_vals = torch.tensor(
                        [sampler_params[i]["top_p"] for i in indices],
                        device=probs.device,
                        dtype=probs.dtype,
                    )
                    sampled = ops.sampling.top_k_top_p_sampling_from_probs(
                        group_probs, top_k=top_k_vals, top_p=top_p_vals
                    )
                else:
                    raise ValueError(f"Unknown sampler index: {sampler_idx}")

                final_tokens_tensor.scatter_(0, indices_tensor, sampled)

        return final_tokens_tensor, final_dists

    def attention(
        self,
        param: Param,
        hidden_states: torch.Tensor,
        layer_idx: int,
        position_ids: torch.Tensor,
        kv_cache_layer: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_positions: torch.Tensor,
        adapter_subpass: Optional[AdapterSubpass],
        wrapper: Any,
    ) -> torch.Tensor:
        """
        Executes the attention block for a single layer, including the
        pre-norm and residual connection.
        """
        # --- Calculate local TP sizes ---
        local_num_query_heads = param.spec.num_q_heads // param.world_size
        local_num_key_value_heads = param.spec.num_kv_heads // param.world_size
        local_q_size = local_num_query_heads * param.spec.dim_head
        local_kv_size = local_num_key_value_heads * param.spec.dim_head

        n = hidden_states.size(0)

        # Save input for the first residual connection (replicated)
        residual = hidden_states

        # 1. Input RMSNorm (replicated input -> replicated output)  ------------------> input gather
        normed_input = fun.rms_norm(
            hidden_states,
            normalized_shape=[param.spec.dim_hidden],
            weight=param.norm_attn[layer_idx],
            eps=param.spec.rms_norm_eps,
        )

        # 2. QKV Projection (Column Parallel)
        # Input is replicated, weight is sharded -> output is sharded
        qkv_proj = fun.linear(
            normed_input,
            weight=param.proj_qkv[layer_idx],
            bias=None,
        )

        # q, k, v are all LOCAL shards
        q, k, v = torch.split(
            qkv_proj,
            [
                local_q_size,
                local_kv_size,
                local_kv_size,
            ],
            dim=-1,
        )

        # ------------------------------- cut here -------------------------------

        # 3. Adapter (if any)
        if adapter_subpass is not None:
            adapter_subpass.execute(
                layer_idx,
                normed_input,  # Adapter needs the (replicated) normed input
                q_state=q,
                k_state=k,
                v_state=v,
            )
        del normed_input
        # ------------------------------- cut here -------------------------------

        # 4. Reshape QKV (local shapes) ------------------> input scatter
        q = q.view(n, local_num_query_heads, param.spec.dim_head)
        k = k.view(n, local_num_key_value_heads, param.spec.dim_head)
        v = v.view(n, local_num_key_value_heads, param.spec.dim_head)

        # 5. Apply RoPE (in-place on local shards)
        ops.apply_llama31_rope_pos_ids_inplace(
            q=q,
            k=k,
            pos_ids=position_ids,
            rope_scale=param.spec.rope_factor,
            rope_theta=param.spec.rope_theta,
            low_freq_factor=param.spec.rope_low_frequency_factor,
            high_freq_factor=param.spec.rope_high_frequency_factor,
        )

        # gather where?

        # 6. Append K, V to cache (local shards to local cache)
        # kv_cache_layer is the LOCAL shard of the cache for this layer
        ops.append_paged_kv_cache(
            append_key=k,
            append_value=v,
            batch_indices=batch_indices,
            positions=batch_positions,
            paged_kv_cache=kv_cache_layer,
            kv_indices=kv_page_indices,
            kv_indptr=kv_page_indptr,
            kv_last_page_len=kv_last_page_lens,
            kv_layout="NHD",
        )

        # 7. Compute Attention (on local shards)
        # wrapper was planned with local head counts
        attn_output = wrapper.run(q, kv_cache_layer)
        del q, k, v
        del qkv_proj

        # ------------------------------- cut here -------------------------------

        # attn_output is a local shard
        attn_output = attn_output.reshape(n, -1)

        # 8. Output Projection (Row Parallel)
        # Input is sharded, weight is sharded -> output is partial
        attn_proj = fun.linear(
            attn_output,
            weight=param.proj_o[layer_idx],
            bias=None,
        )
        del attn_output

        # ALL-REDUCE: Sum partial outputs from all ranks
        dist.all_reduce(attn_proj)

        # 9. First Residual Connection
        # residual (replicated) + attn_proj (now replicated)
        return residual + attn_proj

    def mlp(
        self,
        hidden_states: torch.Tensor,
        param: Param,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Executes the MLP block for a single layer, including the
        pre-norm and residual connection.
        """
        # Save result for the second residual connection (replicated)
        residual = hidden_states

        # 10. Post-Attention RMSNorm (replicated)  <----------- can be used as gather
        normed_mlp_input = fun.rms_norm(
            hidden_states,
            normalized_shape=[param.spec.dim_hidden],
            weight=param.norm_mlp[layer_idx],
            eps=param.spec.rms_norm_eps,
        )

        # 11. Gate/Up Projection (Column Parallel)
        # Input is replicated, weight is sharded -> output is sharded
        gate_up = fun.linear(
            normed_mlp_input,
            weight=param.proj_gate_up[layer_idx],
            bias=None,
        )
        del normed_mlp_input

        # gate and up are local shards
        gate, up = gate_up.chunk(2, dim=-1)

        # 12. SiLU Activation & Gating (on local shards)
        interim_state = fun.silu(gate) * up
        del gate, up
        del gate_up

        # 13. Down Projection (Row Parallel)
        # Input is sharded, weight is sharded -> output is partial
        mlp_output = fun.linear(
            interim_state,
            weight=param.proj_down[layer_idx],
            bias=None,
        )
        del interim_state

        # ALL-REDUCE: Sum partial outputs from all ranks
        dist.all_reduce(mlp_output)

        # 14. Second Residual Connection
        # residual (replicated) + mlp_output (now replicated)
        return residual + mlp_output  # <---------------- can be used as scatter

    def transform(
        self,
        param: Param,
        # inputs
        input_embeds: torch.Tensor,  # Replicated [n, hidden_size]
        position_ids: torch.Tensor,
        qo_indptr: torch.Tensor,
        # kv cache
        kv_cache_at_layer: list[torch.Tensor],  # Each tensor is a LOCAL shard
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        # mask
        custom_mask: torch.Tensor | None,
        single_token_inference_mode: bool,
        # subpasses
        adapter_subpass: Optional[AdapterSubpass],
    ) -> torch.Tensor:

        # --- Calculate local TP sizes ---
        # <-- These are still needed here for planning the wrapper
        local_num_query_heads = param.spec.num_q_heads // param.world_size
        local_num_key_value_heads = param.spec.num_kv_heads // param.world_size

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

        if single_token_inference_mode:
            wrapper = self.wrapper_decode
            wrapper.plan(
                indptr=kv_page_indptr,
                indices=kv_page_indices,
                last_page_len=kv_last_page_lens,
                num_qo_heads=local_num_query_heads,  # Use local head count
                num_kv_heads=local_num_key_value_heads,  # Use local head count
                head_dim=param.spec.dim_head,
                page_size=page_size,
                pos_encoding_mode="NONE",
                q_data_type=input_embeds.dtype,
            )
        else:
            wrapper = self.wrapper_append
            wrapper.plan(
                qo_indptr=qo_indptr,
                paged_kv_indptr=kv_page_indptr,
                paged_kv_indices=kv_page_indices,
                paged_kv_last_page_len=kv_last_page_lens,
                num_qo_heads=local_num_query_heads,  # Use local head count
                num_kv_heads=local_num_key_value_heads,  # Use local head count
                head_dim_qk=param.spec.dim_head,
                page_size=page_size,
                custom_mask=custom_mask,
                q_dta_type=input_embeds.dtype,
            )

        for layer_idx in range(param.spec.num_layers):
            # 1. Attention Block (includes pre-norm and residual)
            hidden_states = self.attention(
                param=param,
                hidden_states=hidden_states,
                layer_idx=layer_idx,
                position_ids=position_ids,
                kv_cache_layer=kv_cache_at_layer[layer_idx],
                kv_page_indices=kv_page_indices,
                kv_page_indptr=kv_page_indptr,
                kv_last_page_lens=kv_last_page_lens,
                batch_indices=batch_indices,
                batch_positions=batch_positions,
                adapter_subpass=adapter_subpass,
                wrapper=wrapper,
            )

            # 2. MLP Block (includes pre-norm and residual)
            hidden_states = self.mlp(
                param=param,
                hidden_states=hidden_states,
                layer_idx=layer_idx,
            )

        # Returns replicated hidden_states
        return hidden_states
