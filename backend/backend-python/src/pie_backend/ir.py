from __future__ import annotations

from dataclasses import dataclass
import enum

import torch


class Operation(enum.Enum):

    ALLOC = -1
    FREE = -2

    EMBED_IMAGE = 0
    EMBED_TOKEN = 1
    PROJ_QKV = 2
    LORA_QKV = 3
    APPLY_ROPE = 4
    UPDATE_KVCACHE = 5
    ATTN_SPARSE = 6
    ATTN_ONE = 7
    ATTN_MANY = 8
    PROJ_O = 93
    PROJ_MLP = 9
    PROJ_MLP_SMALL = 90  # unused
    PROJ_MLP_MEDIUM = 91  # unused
    PROJ_HEAD = 10
    SPECULATE = 11  # unused
    SAMPLE_TOP_K = 11
    SAMPLE_POLY = 12
    SAMPLE_TOP_P = 13
    SAMPLE_TOP_P_TOP_K = 14
    SAMPLE_MIN_P = 15


"""
INPUT w_qkv_i, ...
INPUT kv_indices, kv_page_indptr, ...
ALLOC x
EMBED_TOKEN token_ids, x (virtual register)
REPEAT i in range(0, 16):
    ALLOC qkv
    PROJ_QKV w_qkv_i, x, qkv
    LORA_QKV w_lora_i, x, qkv
    APPLY_ROPE pos_ids, qkv
    ALLOC y
    ATTN_MANY i, qkv, y, kv_indices, kv_page_indptr, kv_last_page_lens
    FREE qkv
    PROJ_O w_o, x, y
    PROJ_MLP w_mlp, y, x
    FREE y
ALLOC d, out
PROJ_HEAD x, d
FREE x
SAMPLE_POLY d, out
FREE d
OUTPUT out
"""


class PagedBuffer:

    hidden_states: dict[BatchGroup, torch.Tensor]

    def __init__(self, num_max_concurrent_req: int):

        self.is_contig = True

    def get_opportunity_cost(self, req_ids: list[int]): ...

    def get_buffer(self, name: str, req_ids: list[int]):

        # if req_ids is in batch group -> easy case

        # if not -> have to create one & migrate.

        #
        ...


@dataclass
class EmbedToken:
    token_ids: list[int]

    def execute(self):
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


@dataclass
class ProjQkv:

    def execute(self, buffer: PagedBuffer):

        hidden_states = buffer.gather("hidden", self.req_ids)

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

        buffer.scatter("qkv", qkv_proj, self.req_ids)


@dataclass
class LoraQkv:
    def execute(self, buffer: PagedBuffer):

        hidden_states = buffer.gather("hidden", self.req_ids)

        normed_input = fun.rms_norm(
            hidden_states,
            normalized_shape=[param.spec.dim_hidden],
            weight=param.norm_attn[layer_idx],
            eps=param.spec.rms_norm_eps,
        )

        qkv_proj = buffer.gather("qkv", self.req_ids)

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

        qkv_proj = adapter_subpass.execute(
            layer_idx,
            normed_input,  # Adapter needs the (replicated) normed input
            q_state=q,
            k_state=k,
            v_state=v,
        )

        buffer.scatter("qkv", qkv_proj, self.req_ids)


@dataclass
class ApplyRope:
    def execute(self, buffer: PagedBuffer):

        qkv_proj = buffer.writer("qkv", self.req_ids)
        position_ids = buffer.reader("position_ids", self.req_ids)

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

        ops.apply_llama31_rope_pos_ids_inplace(
            q=q,
            k=k,
            pos_ids=position_ids,
            rope_scale=param.spec.rope_factor,
            rope_theta=param.spec.rope_theta,
            low_freq_factor=param.spec.rope_low_frequency_factor,
            high_freq_factor=param.spec.rope_high_frequency_factor,
        )


@dataclass
class AttnOne:
    def execute(self, buffer: PagedBuffer):

        qkv_proj = buffer.reader("qkv", self.req_ids)

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

        wrapper = buffer.get_attn_one_wrapper(self.req_ids)

        attn_output = wrapper.run(q, kv_cache_layer)

        # wrapper.plan(
        #     indptr=kv_page_indptr,
        #     indices=kv_page_indices,
        #     last_page_len=kv_last_page_lens,
        #     num_qo_heads=local_num_query_heads,  # Use local head count
        #     num_kv_heads=local_num_key_value_heads,  # Use local head count
        #     head_dim=param.config.head_size,
        #     page_size=page_size,
        #     pos_encoding_mode="NONE",
        #     q_data_type=input_embeds.dtype,
        # )

        ...


@dataclass
class AttnMany:
    def execute(self):
        attn_output = wrapper.run(q, kv_cache_layer)

        ...


@dataclass
class ProjO:
    def execute(self):

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


@dataclass
class ProjMlp:
    def execute(self):
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


@dataclass
class ProjHead:

    def execute(self):
        """
        Performs the final linear projection (LM head).

        Input hidden_states are replicated.
        Weight is sharded [local_vocab_size, hidden_size].
        Output logits are sharded [batch_size, local_vocab_size].
        """
        return fun.linear(
            input=hidden_states,
            # Weight is the sharded embedding weight
            weight=params["model.embed_tokens.weight"],
        )
