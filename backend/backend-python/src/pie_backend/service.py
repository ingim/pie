from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from dataclasses import dataclass, asdict

from . import message
from .adapter import AdapterSubpass
from .model import Model
from .utils import resolve_cache_dir, is_apple_silicon

# Direct import of backend operations based on platform
# metal_kernels now provides the same API structure as flashinfer
if is_apple_silicon():
    import flashinfer_metal as ops  # type: ignore[import-not-found]
else:
    import flashinfer as ops  # type: ignore[import-not-found]


@dataclass
class ServiceConfig:
    """
    Configuration for the server, validated upon creation.

    Use the `ServerConfig.from_args(...)` factory method to create
    a new instance from raw arguments.
    """

    # Required args
    model: str
    cache_dir: str
    # host: str = "localhost"
    # port: int = 10123
    # internal_auth_token: str | None = None
    kv_page_size: int = 16
    max_dist_size: int = 64
    max_num_embeds: int = 128
    max_batch_tokens: int = 10240
    max_num_adapters: int = 48
    max_adapter_rank: int = 8
    max_num_kv_pages: int | None = None
    gpu_mem_headroom: float | None = None
    device: torch.device | None = None
    dtype: torch.dtype = torch.bfloat16
    enable_profiling: bool = False

    @classmethod
    def from_args(
        cls,
        model: str,
        cache_dir: str | None = None,  # Input can be None
        kv_page_size: int = 16,
        max_dist_size: int = 64,
        max_num_embeds: int = 128,
        max_batch_tokens: int = 10240,
        max_num_adapters: int = 48,
        max_adapter_rank: int = 8,
        max_num_kv_pages: int | None = None,
        gpu_mem_headroom: float | None = None,
        device: str | None = None,
        dtype: str = "bfloat16",
        enable_profiling: bool = False,
    ) -> ServiceConfig:
        """
        Factory method to build a validated and resolved ServerConfig.
        This replaces the original `build_config` logic.
        """
        # 2. Resolution (from build_config)
        resolved_cache_dir = resolve_cache_dir(cache_dir)

        # 3. Create the immutable config instance
        return cls(
            model=model,
            cache_dir=resolved_cache_dir,  # Use the resolved value
            kv_page_size=kv_page_size,
            max_dist_size=max_dist_size,
            max_num_embeds=max_num_embeds,
            max_batch_tokens=max_batch_tokens,
            max_num_adapters=max_num_adapters,
            max_adapter_rank=max_adapter_rank,
            max_num_kv_pages=max_num_kv_pages,
            gpu_mem_headroom=gpu_mem_headroom,
            device=torch.device(device),
            dtype=torch.dtype(dtype),
            enable_profiling=enable_profiling,
        )

    def print(self) -> None:
        """
        Utility to print configuration in a consistent format.
        This replaces the original `print_config` function.
        """
        print("--- Configuration ---")
        # asdict() conveniently converts the dataclass to a dict
        for key, value in asdict(self).items():
            print(f"{key}: {value}")
        print("----------------------")


class Service:
    """Python backend handler using platform-appropriate operations."""

    # max_num_kv_pages: int
    # max_num_embeds: int
    # max_num_adapters: int
    # max_adapter_rank: int

    def __init__(
        self,
        config: ServiceConfig,
    ):
        """Initialize handler with platform-appropriate backend operations."""
        self.adapters = {}
        self.config = config  # Store config for later use

        self.model = Model(
            path=Path(config.cache_dir),
            name=config.model,
            device=config.device,
        )

        self.embeds = torch.empty(
            (config.max_num_embeds, self.model.config.hidden_size),
            device=config.device,
            dtype=config.dtype,
        )

        self.adapter_at_layer = [
            (
                torch.zeros(
                    (
                        config.max_num_adapters,
                        config.max_adapter_rank * 3,
                        self.model.config.hidden_size,
                    ),
                    dtype=config.dtype,
                    device=config.device,
                ),
                torch.zeros(
                    (
                        config.max_num_adapters,
                        self.model.config.head_size
                        * (
                            self.model.config.num_query_heads
                            + self.model.config.num_key_value_heads * 2
                        ),
                        config.max_adapter_rank,
                    ),
                    dtype=config.dtype,
                    device=config.device,
                ),
            )
            for _ in range(self.model.config.num_layers)
        ]

        # Automatically decide max_num_kv_pages based on the current memory availability
        if config.gpu_mem_headroom is not None:
            # Calculate the available GPU memory for the KV cache after accounting for
            # the reserved GPU memory specified through `gpu_mem_headroom`.
            free_gpu_mem_bytes, total_gpu_mem_bytes = torch.cuda.mem_get_info(
                config.device
            )
            used_gpu_mem_bytes = total_gpu_mem_bytes - free_gpu_mem_bytes
            reserved_gpu_mem_percentage = config.gpu_mem_headroom
            usable_gpu_mem_bytes = total_gpu_mem_bytes * (
                1 - (reserved_gpu_mem_percentage / 100)
            )
            available_kv_cache_bytes = usable_gpu_mem_bytes - used_gpu_mem_bytes

            if available_kv_cache_bytes <= 0:
                raise ValueError(
                    "Not enough GPU memory available to allocate the KV cache. "
                    "Please decrease 'gpu_mem_headroom'."
                )

            # Calculate the number of KV pages based on the available GPU memory.
            max_num_kv_pages = int(
                available_kv_cache_bytes
                / (
                    config.kv_page_size
                    * 2
                    * self.model.config.num_key_value_heads
                    * self.model.config.head_size
                    * self.model.config.num_layers
                    * config.dtype.itemsize
                )
            )

            # If the user also specified "max_num_kv_pages", then we will use the
            # larger of the two values.
            if config.max_num_kv_pages is not None:
                if max_num_kv_pages > config.max_num_kv_pages:
                    print(
                        f"'max_num_kv_pages' is increased to {max_num_kv_pages} "
                        "to respect 'gpu_mem_headroom'."
                    )
                    config.max_num_kv_pages = max_num_kv_pages
            else:
                config.max_num_kv_pages = max_num_kv_pages

        self.kv_cache_at_layer = [
            torch.zeros(
                (
                    config.max_num_kv_pages,
                    2,
                    config.kv_page_size,
                    self.model.config.num_key_value_heads,
                    self.model.config.head_size,
                ),
                dtype=config.dtype,
                device=config.device,
            )
            for _ in range(self.model.config.num_layers)
        ]

    def handshake(
        self, reqs: list[message.HandshakeRequest]
    ) -> list[message.HandshakeResponse]:
        """Handle handshake requests."""
        resps = []
        for _ in reqs:
            # Request details not currently used

            metadata = self.model.get_metadata()
            tokenizer = self.model.get_tokenizer()
            chat_template = self.model.get_chat_template()

            resp = message.HandshakeResponse(
                version=metadata["version"],
                model_name=metadata["name"],
                model_traits=["todo"],
                model_description=metadata["description"],
                prompt_template=chat_template["template_content"],
                prompt_template_type=chat_template["template_type"],
                prompt_stop_tokens=chat_template["stop_tokens"],
                kv_page_size=self.config.kv_page_size,
                max_batch_tokens=self.config.max_batch_tokens,
                resources={
                    0: self.config.max_num_kv_pages,
                    1: self.config.max_num_embeds,
                    2: self.config.max_num_adapters,
                },
                tokenizer_num_vocab=tokenizer["num_vocab"],
                tokenizer_merge_table=tokenizer["merge_table"],
                tokenizer_special_tokens=tokenizer["special_tokens"],
                tokenizer_split_regex=tokenizer["split_regex"],
                tokenizer_escape_non_printable=tokenizer["escape_non_printable"],
            )
            resps.append(resp)
        return resps

    def query(self, reqs: list[message.QueryRequest]) -> list[message.QueryResponse]:
        """Handle query requests."""
        resps = []
        for req in reqs:
            value = "unknown query"
            match req.query:
                case "ping":
                    value = "pong"
            resp = message.QueryResponse(value=value)
            resps.append(resp)
        return resps

    def embed_image(self, reqs: list[message.EmbedImageRequest]):
        """
        Embeds images into the specified embed pointers.
        """
        raise NotImplementedError

    @torch.inference_mode()
    def initialize_adapter(self, reqs: list[message.InitializeAdapterRequest]):
        raise NotImplementedError

    @torch.inference_mode()
    def update_adapter(self, reqs: list[message.UpdateAdapterRequest]):
        raise NotImplementedError

    @torch.inference_mode()
    def forward_pass(self, reqs: list[message.ForwardPassRequest]):
        reqs = sorted(reqs, key=lambda o: (o.adapter is None, o.adapter))

        batch = ForwardPassBatch(self)
        for req in reqs:
            batch.add_request(req)

        model_inputs = batch.finalize()

        output_embeds = self.model.forward(
            kv_cache_at_layer=self.kv_cache_at_layer, **model_inputs
        )

        responses = batch.package_responses(output_embeds)

        return responses

    def heartbeat(
        self, reqs: list[message.HeartbeatRequest]
    ) -> list[message.HeartbeatResponse]:
        """Handle heartbeat requests to keep the connection alive."""
        resps = []
        for _ in reqs:
            resps.append(message.HeartbeatResponse())
        return resps

    def upload_adapter(self, reqs: list[message.UploadAdapterRequest]):
        raise NotImplementedError

    def download_adapter(
        self, reqs: list[message.DownloadAdapterRequest]
    ) -> list[message.DownloadAdapterResponse]:
        raise NotImplementedError


def _decode_brle(brle_buffer: list[int]) -> np.ndarray:
    """
    Decodes a Binary Run-Length Encoded buffer into a boolean numpy array.
    The format assumes alternating runs of False and True, starting with False.
    """
    if not brle_buffer:
        return np.array([], dtype=bool)

    total_size = sum(brle_buffer)
    if total_size == 0:
        return np.array([], dtype=bool)

    decoded_array = np.empty(total_size, dtype=bool)
    current_pos = 0
    value = True  # In attention masking, True means attend.
    for run_len in brle_buffer:
        if run_len > 0:
            decoded_array[current_pos : current_pos + run_len] = value
        current_pos += run_len
        value = not value  # Flip value for the next run
    return decoded_array


class ForwardPassBatch:
    """Consolidates and processes a batch of forward pass requests."""

    # Static constant for the maximum top_k value for distributions.
    TOP_K_MAX_BOUND = 1024

    # important
    service: Service
    requests: list[message.ForwardPassRequest]

    # inputs for the model
    adapter_indices: list[int]
    seeds: list[int]
    kv_page_indices: list[int]
    kv_page_indptr: list[int]
    kv_last_page_lengths: list[int]
    qo_indptr: list[int]
    attention_masks: list[np.ndarray]
    batch_token_ids: list[int]
    batch_position_ids: list[int]

    # tracking states
    total_tokens_in_batch: int
    single_token_inference_mode: bool
    adapter_subpass_needed: bool

    # Output mapping for all logit-based operations (dists and sampling)
    indices_for_logits: list[int]
    indices_for_embed_storage: list[int]
    embed_storage_pointers: list[int]

    # sampler type and consolidated parameters
    sampler_type: list[int]
    sampler_params: list[dict]

    def __init__(self, service: Service):
        """Initializes the batch processor."""
        self.service = service
        self.requests = []

        # Inputs for the model
        self.adapter_indices = []
        self.seeds = []
        self.kv_page_indices = []
        self.kv_page_indptr = [0]
        self.kv_last_page_lengths = []
        self.qo_indptr = [0]
        self.attention_masks = []
        self.batch_token_ids = []
        self.batch_position_ids = []

        # Tracking state
        self.total_tokens_in_batch = 0
        self.single_token_inference_mode = True
        self.adapter_subpass_needed = False

        # Output mapping for all logit-based operations (dists and sampling)
        self.indices_for_logits = []
        self.indices_for_embed_storage = []
        self.embed_storage_pointers = []

        # Sampler type and consolidated parameters
        self.sampler_type = []
        self.sampler_params = []

    def add_request(self, req: message.ForwardPassRequest):
        """Processes and adds a single request to the batch."""
        self.requests.append(req)

        # Handle adapter information
        if req.adapter is not None and req.adapter in self.service.adapters:
            seed = req.adapter_seed if req.adapter_seed is not None else 0
            self.seeds.extend([seed] * len(req.input_tokens))
            self.adapter_indices.append(req.adapter)
            self.adapter_subpass_needed = True

        # Handle KV cache pages
        kv_page_ptrs = req.kv_page_ptrs or []
        self.kv_page_indices.extend(kv_page_ptrs)
        self.kv_page_indptr.append(len(self.kv_page_indices))
        self.kv_last_page_lengths.append(req.kv_page_last_len or 0)

        # Handle output mappings for embeddings that need to be stored
        output_embed_indices = req.output_embed_indices or []
        output_embed_ptrs = req.output_embed_ptrs or []
        if len(output_embed_indices) != len(output_embed_ptrs):
            raise ValueError(
                f"Mismatch between output_embed_indices length ({len(output_embed_indices)}) "
                f"and output_embed_ptrs length ({len(output_embed_ptrs)})"
            )
        for token_idx, storage_ptr in zip(output_embed_indices, output_embed_ptrs):
            self.indices_for_embed_storage.append(
                token_idx + self.total_tokens_in_batch
            )
            self.embed_storage_pointers.append(storage_ptr)

        # Handle output mappings for tokens requiring logits.
        output_token_indices = req.output_token_indices or []
        for token_idx in output_token_indices:
            self.indices_for_logits.append(token_idx + self.total_tokens_in_batch)

        # Extract sampler configurations.
        # sampler_idx=0 is for distributions, existing samplers are shifted by +1.
        output_token_samplers = req.output_token_samplers or []
        for sampler_config in output_token_samplers:
            params = {}
            sampler_idx = sampler_config["sampler"]
            self.sampler_type.append(sampler_idx)

            if sampler_idx == 0:
                params["top_k"] = min(
                    sampler_config.get("top_k", self.service.config.max_dist_size),
                    self.service.config.max_dist_size,
                )
            else:
                params["top_k"] = sampler_config.get("top_k", 0)
                params["top_p"] = sampler_config.get("top_p", 1.0)
                params["min_p"] = sampler_config.get("min_p", 0.0)

            params["temperature"] = sampler_config.get("temperature", 1.0)
            self.sampler_params.append(params)

        # Handle input tokens and positions
        self.batch_token_ids.extend(req.input_tokens)
        self.batch_position_ids.extend(req.input_token_positions)
        self.total_tokens_in_batch += len(req.input_tokens)
        self.qo_indptr.append(self.total_tokens_in_batch)

        if len(req.input_tokens) > 1:
            self.single_token_inference_mode = False

        attention_mask = self._generate_mask_for_request(req)
        self.attention_masks.append(attention_mask)

    def _generate_mask_for_request(self, req: message.ForwardPassRequest) -> np.ndarray:
        """Generates the custom attention mask for a single request."""
        if len(req.mask) != len(req.input_tokens):
            raise ValueError(
                f"Mismatch between number of masks ({len(req.mask)}) and "
                f"input tokens ({len(req.input_tokens)})."
            )

        kv_page_ptrs = req.kv_page_ptrs or []
        kv_page_last_len = req.kv_page_last_len or 0

        # Ensure we have at least one page for proper computation
        if len(kv_page_ptrs) >= 1:
            sequence_length = (
                self.service.config.kv_page_size * (len(kv_page_ptrs) - 1)
                + kv_page_last_len
            )
        else:
            sequence_length = kv_page_last_len

        # Validate sequence_length is sufficient for input tokens
        input_token_count = len(req.input_tokens)
        if sequence_length < input_token_count:
            raise ValueError(
                f"Insufficient sequence length ({sequence_length}) for input tokens "
                f"({input_token_count}). Sequence length must be at least equal to "
                f"the number of input tokens."
            )

        context_length = sequence_length - input_token_count

        request_attention_mask = np.zeros(
            (len(req.input_tokens), sequence_length), dtype=np.bool_
        )
        for i, brle_buffer in enumerate(req.mask):
            decoded_mask = _decode_brle(brle_buffer)
            expected_len = context_length + i + 1
            if len(decoded_mask) != expected_len:
                raise ValueError(
                    f"Decoded mask for token {i} has length {len(decoded_mask)}, "
                    f"but expected {expected_len}"
                )
            request_attention_mask[i, :expected_len] = decoded_mask

        return request_attention_mask.flatten()

    def finalize(self) -> dict:
        """Finalizes batch preparation, creating tensors and the adapter subpass."""
        device = self.service.config.device

        adapter_subpass = None
        if self.adapter_subpass_needed:
            seeds_tensor = torch.as_tensor(self.seeds, device=device, dtype=torch.long)
            adapter_subpass = AdapterSubpass(
                adapter_at_layer=self.service.adapter_at_layer,
                adapter_indices=self.adapter_indices,
                adapter_extras=self.service.adapters,
                rand_seeds=seeds_tensor,
                qo_indptr=self.qo_indptr,
            )

        batched_attention_mask = (
            np.concatenate(self.attention_masks)
            if self.attention_masks
            else np.array([], dtype=np.bool_)
        )
        token_ids_tensor = torch.as_tensor(
            self.batch_token_ids, device=device, dtype=torch.int32
        )

        input_embeds = self.service.model.embed_tokens(token_ids_tensor)  # type: ignore[operator]

        result = {
            "input_embeds": input_embeds,
            "position_ids": torch.as_tensor(
                self.batch_position_ids, device=device, dtype=torch.int32
            ),
            "qo_indptr": torch.as_tensor(
                self.qo_indptr, device=device, dtype=torch.int32
            ),
            "kv_page_indices": torch.as_tensor(
                self.kv_page_indices, device=device, dtype=torch.int32
            ),
            "kv_page_indptr": torch.as_tensor(
                self.kv_page_indptr, device=device, dtype=torch.int32
            ),
            "kv_last_page_lens": torch.as_tensor(
                self.kv_last_page_lengths, device=device, dtype=torch.int32
            ),
            "custom_mask": torch.as_tensor(
                batched_attention_mask, device=device, dtype=torch.bool
            ),
            "single_token_inference_mode": self.single_token_inference_mode,
            "adapter_subpass": adapter_subpass,
        }

        return result

    def package_responses(
        self, output_embeds: torch.Tensor
    ) -> list[message.ForwardPassResponse]:
        """Packages the model outputs into responses for each original request."""
        # Handle storing specified embeddings

        device = self.service.config.device

        if self.indices_for_embed_storage:
            embeddings_to_store = output_embeds[self.indices_for_embed_storage]
            for i, ptr in enumerate(self.embed_storage_pointers):
                self.service.embeds[ptr].copy_(
                    embeddings_to_store[i], non_blocking=True
                )

        if not self.indices_for_logits:
            return [
                message.ForwardPassResponse(dists=[], tokens=[]) for _ in self.requests
            ]

        # Calculate logits for all required tokens (both dists and samples)
        logits_input = output_embeds[self.indices_for_logits]

        logits = self.service.lm.lm_head(logits_input)  # type: ignore[attr-defined, operator]

        # Apply temperature scaling to all logits
        temperatures = torch.tensor(
            [p["temperature"] for p in self.sampler_params],
            device=device,
            dtype=logits.dtype,
        ).unsqueeze(1)
        scaled_logits = logits / torch.clamp(temperatures, min=1e-6)

        # We compute probabilities for the entire batch of logit requests
        probs = torch.softmax(scaled_logits, dim=-1)

        if not torch.isfinite(probs).all():
            raise RuntimeError("Non-finite probabilities produced by LM head")

        # Group requests by sampler type for efficient batch processing
        sampler_groups = {}
        for i, sampler_idx in enumerate(self.sampler_type):
            if sampler_idx not in sampler_groups:
                sampler_groups[sampler_idx] = []
            sampler_groups[sampler_idx].append(i)

        num_logit_requests = len(self.indices_for_logits)
        # Initialize result containers. Using lists of Nones helps place results correctly.
        final_dists: list[tuple[list[int], list[float]] | None] = [
            None
        ] * num_logit_requests
        final_tokens_tensor = torch.empty(
            num_logit_requests, dtype=torch.long, device=device
        )

        for sampler_idx, indices in sampler_groups.items():
            if not indices:
                continue

            indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)
            group_probs = probs.index_select(0, indices_tensor)

            # Handle distributions (sampler_idx=0)
            if sampler_idx == 0:
                group_top_k = [self.sampler_params[i]["top_k"] for i in indices]
                max_k = max(group_top_k) if group_top_k else 0
                if max_k > 0:
                    topk_vals, topk_inds = torch.topk(group_probs, k=max_k, sorted=True)
                    for i, original_idx in enumerate(indices):
                        k = self.sampler_params[original_idx]["top_k"]
                        ids = topk_inds[i, :k].tolist()
                        vals = topk_vals[i, :k].tolist()
                        final_dists[original_idx] = (ids, vals)

            # Handle sampling operations (sampler_idx > 0)
            else:
                sampled = None
                if sampler_idx == 1:  # Old 0: sampling_from_probs
                    ops_sampling = self.service.ops.sampling  # type: ignore
                    sampled = ops_sampling.sampling_from_probs(group_probs)
                elif sampler_idx == 2:  # Old 1: top_p_sampling_from_probs
                    top_p_vals = torch.tensor(
                        [self.sampler_params[i]["top_p"] for i in indices],
                        device=device,
                        dtype=self.service.config.dtype,
                    )
                    ops_sampling = self.service.ops.sampling  # type: ignore
                    sampled = ops_sampling.top_p_sampling_from_probs(
                        group_probs, top_p=top_p_vals
                    )
                elif sampler_idx == 3:  # Old 2: top_k_sampling_from_probs
                    top_k_vals = torch.tensor(
                        [self.sampler_params[i]["top_k"] for i in indices],
                        device=device,
                        dtype=torch.long,
                    )
                    ops_sampling = self.service.ops.sampling  # type: ignore
                    sampled = ops_sampling.top_k_sampling_from_probs(
                        group_probs, top_k=top_k_vals
                    )
                elif sampler_idx == 4:  # Old 3: min_p_sampling_from_probs
                    min_p_vals = torch.tensor(
                        [self.sampler_params[i]["min_p"] for i in indices],
                        device=device,
                        dtype=self.service.config.dtype,
                    )
                    ops_sampling = self.service.ops.sampling  # type: ignore
                    sampled = ops_sampling.min_p_sampling_from_probs(
                        group_probs, min_p=min_p_vals
                    )
                elif sampler_idx == 5:  # Old 4: top_k_top_p_sampling_from_probs
                    top_k_vals = torch.tensor(
                        [self.sampler_params[i]["top_k"] for i in indices],
                        device=device,
                        dtype=torch.long,
                    )
                    top_p_vals = torch.tensor(
                        [self.sampler_params[i]["top_p"] for i in indices],
                        device=device,
                        dtype=self.service.config.dtype,
                    )
                    ops_sampling = self.service.ops.sampling  # type: ignore
                    fn = ops_sampling.top_k_top_p_sampling_from_probs
                    sampled = fn(group_probs, top_k=top_k_vals, top_p=top_p_vals)
                else:
                    raise ValueError(f"Unknown sampler index: {sampler_idx}")

                # Place sampled tokens into the main tensor at their original batch positions
                # Ensure sampled tokens have the correct dtype (torch.long for token indices)
                if sampled.dtype != torch.long:
                    sampled = sampled.to(torch.long)
                final_tokens_tensor.scatter_(0, indices_tensor, sampled)

        # Distribute batched results back to individual responses
        responses = []
        cursor = 0
        for req in self.requests:
            output_token_indices = req.output_token_indices or []
            num_outputs = len(output_token_indices)
            request_dists = []
            request_tokens = []

            # Iterate through the slice of results belonging to this request
            for i in range(cursor, cursor + num_outputs):
                if self.sampler_type[i] == 0:  # This was a distribution request
                    if final_dists[i] is not None:
                        request_dists.append(final_dists[i])
                else:  # This was a sampling request
                    request_tokens.append(final_tokens_tensor[i].item())

            responses.append(
                message.ForwardPassResponse(dists=request_dists, tokens=request_tokens)
            )
            cursor += num_outputs

        return responses
