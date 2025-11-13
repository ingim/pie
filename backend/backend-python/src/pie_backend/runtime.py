from __future__ import annotations
from dataclasses import dataclass, asdict, field

import base64
import tomllib
from contextlib import ExitStack, closing
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm
import torch
import ztensor
import safetensors

from .model import llama3
from .model import qwen2
from .model import qwen3
from .model import llama3_tp
from .adapter import AdapterSubpass
from .utils import resolve_cache_dir


@dataclass
class ModelConfig:
    model: str
    cache_dir: str
    kv_page_size: int
    max_dist_size: int
    max_num_embeds: int
    max_batch_tokens: int | None
    max_num_adapters: int
    max_adapter_rank: int
    max_num_kv_pages: int | None
    mem_utilization: float

    device: list[torch.device]
    rank: int
    dtype: torch.dtype

    @classmethod
    def from_args(
        cls,
        model: str,
        cache_dir: str,
        kv_page_size: int,
        max_dist_size: int,
        max_num_embeds: int,
        max_num_adapters: int,
        max_adapter_rank: int,
        mem_utilization: float,
        device: list[str],
        rank: int,
        dtype: str = "bfloat16",
    ) -> ModelConfig:
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
            max_batch_tokens=None,
            max_num_adapters=max_num_adapters,
            max_adapter_rank=max_adapter_rank,
            max_num_kv_pages=None,
            mem_utilization=mem_utilization,
            device=[torch.device(d) for d in device],
            rank=rank,
            dtype=torch.dtype(dtype),
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


class LoadedModel:

    config: ModelConfig
    param: object
    buffer: object
    forward_pass: object
    batch: SyncBatch


class Runtime:

    config: ModelConfig
    rank: int

    model_config: llama3.Config | qwen2.Config | qwen3.Config
    model_param: llama3.Param
    model_pass: llama3.ForwardPass

    adapter_at_layer: list[tuple[torch.Tensor, torch.Tensor]]
    kv_cache_at_layer: list[tuple[torch.Tensor, torch.Tensor]]

    batch: SyncBatch
    adapters: dict

    def __init__(self, config: ModelConfig):

        self.config = config

        model_info_path = Path(config.cache_dir) / f"{config.model}.toml"

        if not model_info_path.exists():
            raise ValueError(
                f'Metadata file for model "{config.model}" not found at: {config.cache_dir}'
            )

        with open(model_info_path, "rb") as f:
            self.info = tomllib.load(f)
            self.type = self.info["architecture"]["type"]

        # get architecture-specific handlers
        match self.type:
            case "l4ma":
                self.prepare_params = llama3.prepare_params
                self.model_config = llama3.Config.from_dict(self.info["architecture"])
                self.forward_pass = llama3.ForwardPass(self.device)
                self.token_embed_pass = llama3.TokenEmbedPass()
                self.pred_pass = llama3.PredictionPass()
            case "qwen2":
                self.prepare_params = qwen2.prepare_params
                self.model_config = qwen2.Config.from_dict(self.info["architecture"])
                self.forward_pass = qwen2.ForwardPass(self.device)
                self.token_embed_pass = qwen2.TokenEmbedPass()
                self.pred_pass = qwen2.PredictionPass()
            case "qwen3":
                self.prepare_params = qwen3.prepare_params
                self.model_config = qwen3.Config.from_dict(self.info["architecture"])
                self.forward_pass = qwen3.ForwardPass(self.device)
                self.token_embed_pass = qwen3.TokenEmbedPass()
                self.pred_pass = qwen3.PredictionPass()

        self._load_params()
        self._init_states()

    def _load_model(self, config: ModelConfig):
        path_param = Path(config.cache_dir) / f"{config.model}"
        path_info = Path(config.cache_dir) / f"{config.model}.toml"

        if not path_info.exists():
            raise ValueError(
                f'Metadata file for model "{config.model}" not found at: {config.cache_dir}'
            )

        with open(path_info, "rb") as f:
            model_info = tomllib.load(f)
            model_type = model_info["architecture"]["type"]

        with ExitStack() as stack:
            readers: dict[str, object] = {}
            for param_file in tqdm(
                model_info["parameters"], desc="Scanning tensor files", unit="files"
            ):
                param_path = path_param / param_file

                if param_path.suffix == ".zt":
                    f = stack.enter_context(
                        ztensor.Reader(str(param_path))
                    )  # closed on exit
                    names = f.get_tensor_names()

                elif param_path.suffix == ".safetensors":
                    f = stack.enter_context(
                        safetensors.safe_open(
                            str(param_path), framework="pt", device="cpu"
                        )
                    )
                    names = list(f.keys())

                else:
                    continue

                for n in names:
                    readers[n] = f

            def reader(
                name: str, *, expected_shape: tuple[int, ...] | None = None
            ) -> torch.Tensor:

                f = readers.get(name)
                if f is None:
                    raise KeyError(f"Tensor '{name}' not found")

                # ztensor vs safetensors
                t = (
                    f.read_tensor(name, to="torch")  # ztensor
                    if hasattr(f, "read_tensor")
                    else f.get_tensor(name)  # safetensors
                )

                if expected_shape is not None and tuple(t.shape) != tuple(
                    expected_shape
                ):
                    raise ValueError(
                        f"{name} has shape {tuple(t.shape)}, expected {tuple(expected_shape)}"
                    )
                return t

            match model_type:
                case "l4ma":

                    model_spec = llama3_tp.Spec.from_dict(model_info["architecture"])
                    model_param = llama3_tp.Param.from_reader(
                        model_spec,
                        reader,
                        device=self.config.device[self.config.rank],
                        rank=self.config.rank,
                        world_size=len(self.config.device),
                    )

                    self.prepare_params = llama3.prepare_params
                    self.model_config = llama3.Config.from_dict(
                        self.info["architecture"]
                    )
                    self.forward_pass = llama3.ForwardPass(self.device)
                    self.token_embed_pass = llama3.TokenEmbedPass()
                    self.pred_pass = llama3.PredictionPass()
                case "qwen2":
                    self.prepare_params = qwen2.prepare_params
                    self.model_config = qwen2.Config.from_dict(
                        self.info["architecture"]
                    )
                    self.forward_pass = qwen2.ForwardPass(self.device)
                    self.token_embed_pass = qwen2.TokenEmbedPass()
                    self.pred_pass = qwen2.PredictionPass()
                case "qwen3":
                    self.prepare_params = qwen3.prepare_params
                    self.model_config = qwen3.Config.from_dict(
                        self.info["architecture"]
                    )
                    self.forward_pass = qwen3.ForwardPass(self.device)
                    self.token_embed_pass = qwen3.TokenEmbedPass()
                    self.pred_pass = qwen3.PredictionPass()
        ...

    def _load_params(self, param_cls: llama3.Param) -> None: ...

    def _init_states(self):
        self.adapter_at_layer = [
            (
                torch.zeros(
                    (
                        self.config.max_num_adapters,
                        self.config.max_adapter_rank * 3,
                        self.model_config.hidden_size,
                    ),
                    dtype=self.config.dtype,
                    device=self.config.device[self.rank],
                ),
                torch.zeros(
                    (
                        self.config.max_num_adapters,
                        self.model_config.head_size
                        * (
                            self.model_config.num_query_heads
                            + self.model_config.num_key_value_heads * 2
                        ),
                        self.config.max_adapter_rank,
                    ),
                    dtype=self.config.dtype,
                    device=self.config.device[self.rank],
                ),
            )
            for _ in range(self.model_config.num_layers)
        ]

        self.kv_cache_at_layer = [
            torch.zeros(
                (
                    self.config.max_num_kv_pages,
                    2,
                    self.config.kv_page_size,
                    self.model_config.num_key_value_heads,
                    self.model_config.head_size,
                ),
                dtype=self.config.dtype,
                device=self.config.device[self.rank],
            )
            for _ in range(self.model_config.num_layers)
        ]

    def get_metadata(self) -> dict:
        return {
            "name": self.info["name"],
            "description": self.info["description"],
            "version": self.info["version"],
        }

    def get_chat_template(self) -> dict:
        return {
            "template_type": self.info["template"]["type"],
            "template_content": self.info["template"]["content"],
            "stop_tokens": self.info["template"]["stop_tokens"],
        }

    def get_tokenizer(self) -> dict:
        vocab_file_path = (
            self.path / self.name / self.info["tokenizer"]["vocabulary_file"]
        )
        merge_rules: dict[int, bytes] = {}

        with open(vocab_file_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                # Skip empty or blank lines
                if not line:
                    continue

                # Expect two parts: base64-encoded token and rank
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(
                        f"Error on line {line_number}: expected 2 parts, "
                        f"but found {len(parts)} (line: '{line}')"
                    )

                b64_token, rank_str = parts

                # 1. Decode base64 token
                try:
                    decoded_token = base64.b64decode(b64_token)
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Error on line {line_number}: failed to decode base64 token."
                    ) from e

                # 2. Parse rank into an integer
                try:
                    rank = int(rank_str)
                except ValueError as e:
                    raise ValueError(
                        f"Error on line {line_number}: failed to parse "
                        f"rank '{rank_str}' as an integer."
                    ) from e

                merge_rules[rank] = decoded_token

        return {
            "type": self.info["tokenizer"]["type"],
            "num_vocab": len(self.info["architecture"]["vocab_size"]),
            "merge_table": merge_rules,
            "split_regex": self.info["tokenizer"]["split_regex"],
            "special_tokens": self.info["tokenizer"]["special_tokens"],
            "escape_non_printable": self.info["tokenizer"]["escape_non_printable"],
        }

    @torch.inference_mode()
    def initialize_adapter(
        self,
        adapter_ptr: int,
        rank: int,
        alpha: float,
        population_size: int,
        mu_fraction: float,
        initial_sigma: float,
    ):
        raise NotImplementedError

    @torch.inference_mode()
    def update_adapter(
        self,
        adapter_ptr: int,
        scores: list[float],
        seeds: list[int],
        max_sigma: float,
    ):
        raise NotImplementedError

    @torch.inference_mode()
    def forward_pass(
        self,
        input_tokens: list[int],
        input_token_positions: list[int],
        adapter: int | None,
        adapter_seed: int | None,
        mask: list[list[int]],
        kv_page_ptrs: list[int],
        kv_page_last_len: int,
        output_token_indices: list[int],
        output_token_samplers: list[dict],
        output_embed_ptrs: list[int],
        output_embed_indices: list[int],
    ):

        # init batch if not present
        if self.batch is None:
            self.batch = SyncBatch()
        batch = self.batch

        # Handle adapter information
        if adapter is not None and adapter in self.adapters:
            seed = adapter_seed if adapter_seed is not None else 0
            batch.seeds.extend([seed] * len(input_tokens))
            batch.adapter_indices.append(adapter)
            batch.adapter_subpass_needed = True

        # Handle KV cache pages
        batch.kv_page_indices.extend(kv_page_ptrs)
        batch.kv_page_indptr.append(len(batch.kv_page_indices))
        batch.kv_last_page_lengths.append(kv_page_last_len or 0)

        # Handle output mappings for embeddings that need to be stored
        if len(output_embed_indices) != len(output_embed_ptrs):
            raise ValueError(
                f"Mismatch between output_embed_indices length ({len(output_embed_indices)}) "
                f"and output_embed_ptrs length ({len(output_embed_ptrs)})"
            )
        for token_idx, storage_ptr in zip(output_embed_indices, output_embed_ptrs):
            batch.indices_for_embed_storage.append(
                token_idx + batch.total_tokens_in_batch
            )
            batch.embed_storage_pointers.append(storage_ptr)

        # Handle output mappings for tokens requiring logits.
        for token_idx in output_token_indices:
            batch.indices_for_logits.append(token_idx + batch.total_tokens_in_batch)

        # Extract sampler configurations.
        # sampler_idx=0 is for distributions, existing samplers are shifted by +1.
        for sampler_config in output_token_samplers:
            params = {}
            sampler_idx = sampler_config["sampler"]
            batch.sampler_type.append(sampler_idx)

            if sampler_idx == 0:
                params["top_k"] = min(
                    sampler_config.get("top_k", self.config.max_dist_size),
                    self.config.max_dist_size,
                )
            else:
                params["top_k"] = sampler_config.get("top_k", 0)
                params["top_p"] = sampler_config.get("top_p", 1.0)
                params["min_p"] = sampler_config.get("min_p", 0.0)

            params["temperature"] = sampler_config.get("temperature", 1.0)
            batch.sampler_params.append(params)

        # Handle input tokens and positions
        batch.batch_token_ids.extend(input_tokens)
        batch.batch_position_ids.extend(input_token_positions)
        batch.total_tokens_in_batch += len(input_tokens)
        batch.qo_indptr.append(batch.total_tokens_in_batch)

        if len(input_tokens) > 1:
            batch.single_token_inference_mode = False

        attention_mask = _generate_mask_for_request(
            input_tokens,
            mask,
            kv_page_ptrs,
            kv_page_last_len,
            self.config.kv_page_size,
        )
        batch.attention_masks.append(attention_mask)

    @torch.inference_mode()
    def block_on_forward_pass(self):
        """Finalizes batch preparation, creating tensors and the adapter subpass."""
        batch = self.batch
        device = self.config.device[self.rank]

        adapter_subpass = None
        if batch.adapter_subpass_needed:
            seeds_tensor = torch.as_tensor(batch.seeds, device=device, dtype=torch.long)
            adapter_subpass = AdapterSubpass(
                adapter_at_layer=self.adapter_at_layer,
                adapter_indices=batch.adapter_indices,
                adapter_extras=self.adapters,
                rand_seeds=seeds_tensor,
                qo_indptr=batch.qo_indptr,
            )

        batched_attention_mask = (
            np.concatenate(batch.attention_masks)
            if batch.attention_masks
            else np.array([], dtype=np.bool_)
        )
        token_ids_tensor = torch.as_tensor(
            batch.batch_token_ids, device=device, dtype=torch.int32
        )

        input_embeds = self.model_pass.embed_tokens(
            param=self.model_param,
            token_ids=token_ids_tensor,
        )

        hidden_states = self.model_pass.transform(
            param=self.model_param,
            input_embeds=input_embeds,
            position_ids=torch.as_tensor(
                batch.batch_position_ids, device=device, dtype=torch.int32
            ),
            qo_indptr=torch.as_tensor(
                batch.qo_indptr, device=device, dtype=torch.int32
            ),
            kv_cache_at_layer=self.kv_cache_at_layer,
            kv_page_indices=torch.as_tensor(
                batch.kv_page_indptr, device=device, dtype=torch.int32
            ),
            kv_page_indptr=torch.as_tensor(
                batch.kv_page_indptr, device=device, dtype=torch.int32
            ),
            kv_last_page_lens=torch.as_tensor(
                batch.kv_last_page_lengths, device=device, dtype=torch.int32
            ),
            custom_mask=torch.as_tensor(
                batched_attention_mask, device=device, dtype=torch.bool
            ),
            single_token_inference_mode=batch.single_token_inference_mode,
            adapter_subpass=adapter_subpass,
        )

        # --------------
        # Apply temperature scaling to all logits
        temperatures = torch.tensor(
            [p["temperature"] for p in batch.sampler_params],
            device=device,
            dtype=logits.dtype,
        ).unsqueeze(1)

        # Group requests by sampler type for efficient batch processing
        sampler_groups = {}
        for i, sampler_idx in enumerate(batch.sampler_type):
            if sampler_idx not in sampler_groups:
                sampler_groups[sampler_idx] = []
            sampler_groups[sampler_idx].append(i)

        sample_outputs, final_dists = self.model_pass.sample(
            param=self.model_param,
            hidden_states=hidden_states,
            indices_for_logits=batch.indices_for_logits,
            temperatures=temperatures,
            sampler_groups=sampler_groups,
            sampler_params=batch.sampler_params,
        )

        # --samling

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

    @torch.inference_mode()
    def upload_adapter(
        self,
        adapter_ptr: int,
        scores: list[float],
        seeds: list[int],
        max_sigma: float,
    ):
        raise NotImplementedError

    @torch.inference_mode()
    def download_adapter(
        self, adapter_data: bytes
    ) -> list[message.DownloadAdapterResponse]:
        raise NotImplementedError


@dataclass
class SyncBatch:
    adapter_indices: list[int] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    kv_page_indices: list[int] = field(default_factory=list)
    kv_page_indptr: list[int] = field(default_factory=list)
    kv_last_page_lengths: list[int] = field(default_factory=list)
    qo_indptr: list[int] = field(default_factory=list)
    attention_masks: list[np.ndarray] = field(default_factory=list)
    batch_token_ids: list[int] = field(default_factory=list)
    batch_position_ids: list[int] = field(default_factory=list)

    # tracking states
    total_tokens_in_batch: int = 0
    single_token_inference_mode: bool = True
    adapter_subpass_needed: bool = False

    # Output mapping for all logit-based operations (dists and sampling)
    indices_for_logits: list[int] = field(default_factory=list)
    indices_for_embed_storage: list[int] = field(default_factory=list)
    embed_storage_pointers: list[int] = field(default_factory=list)

    # sampler type and consolidated parameters
    sampler_type: list[int] = field(default_factory=list)
    sampler_params: list[dict] = field(default_factory=list)


def _generate_mask_for_request(
    input_tokens: list[int],
    mask: list[list[int]],
    kv_page_ptrs: list[int],
    kv_page_last_len: int,
    kv_page_size: int,
) -> np.ndarray:
    """Generates the custom attention mask for a single request."""
    if len(mask) != len(input_tokens):
        raise ValueError(
            f"Mismatch between number of masks ({len(mask)}) and "
            f"input tokens ({len(input_tokens)})."
        )

    # Ensure we have at least one page for proper computation
    if len(kv_page_ptrs) >= 1:
        sequence_length = kv_page_size * (len(kv_page_ptrs) - 1) + kv_page_last_len
    else:
        sequence_length = kv_page_last_len

    # Validate sequence_length is sufficient for input tokens
    input_token_count = len(input_tokens)
    if sequence_length < input_token_count:
        raise ValueError(
            f"Insufficient sequence length ({sequence_length}) for input tokens "
            f"({input_token_count}). Sequence length must be at least equal to "
            f"the number of input tokens."
        )

    context_length = sequence_length - input_token_count

    request_attention_mask = np.zeros(
        (len(input_tokens), sequence_length), dtype=np.bool_
    )
    for i, brle_buffer in enumerate(mask):
        decoded_mask = _decode_brle(brle_buffer)
        expected_len = context_length + i + 1
        if len(decoded_mask) != expected_len:
            raise ValueError(
                f"Decoded mask for token {i} has length {len(decoded_mask)}, "
                f"but expected {expected_len}"
            )
        request_attention_mask[i, :expected_len] = decoded_mask

    return request_attention_mask.flatten()


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
