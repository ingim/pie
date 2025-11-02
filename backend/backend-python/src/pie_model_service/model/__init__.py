from __future__ import annotations

import base64
import tomllib
from pathlib import Path
from typing import Callable
import torch
import ztensor
from tqdm import tqdm
import llama3
import qwen2
import qwen3
import safetensors

from src.pie_model_service.adapter import AdapterSubpass


class Model:

    path: Path
    info: dict
    type: str
    params: dict
    device: torch.device

    prepare_params: Callable
    config: llama3.Config | qwen2.Config | qwen3.Config
    forward_pass: llama3.ForwardPass | qwen2.ForwardPass | qwen3.ForwardPass

    def __init__(
        self,
        path: Path,
        name: str,
        device: torch.device,
    ):
        self.path = path
        self.name = name
        self.device = device
        model_info_path = path / f"{name}.toml"

        if not model_info_path.exists():
            raise ValueError(f'Metadata file for model "{name}" not found at: {path}')

        with open(model_info_path, "rb") as f:
            self.info = tomllib.load(f)
            self.type = self.info["architecture"]["type"]

        # get architecture-specific handlers
        match self.type:
            case "l4ma":
                self.prepare_params = llama3.prepare_params
                self.config = llama3.Config.from_dict(self.info["architecture"])
                self.forward_pass = llama3.ForwardPass(self.device)
            case "qwen2":
                self.prepare_params = qwen2.prepare_params
                self.config = qwen2.Config.from_dict(self.info["architecture"])
                self.forward_pass = qwen2.ForwardPass(self.device)
            case "qwen3":
                self.prepare_params = qwen3.prepare_params
                self.config = qwen3.Config.from_dict(self.info["architecture"])
                self.forward_pass = qwen3.ForwardPass(self.device)

        self._init_on_device(self.device)

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

    def _init_on_device(self, device: torch.device):

        params = {}

        for param_file in tqdm(
            self.info["parameters"], desc="Scanning tensor files", unit="files"
        ):

            param_path = self.path / self.name / param_file

            match param_path.suffix:
                case "zt":
                    with ztensor.Reader(str(param_path)) as f:
                        for n in tqdm(
                            f.get_tensor_names(),
                            desc="Loading tensor parameters",
                            unit="tensors",
                        ):
                            params[n] = f.read_tensor(n, to="torch").to(device)

                case "safetensors":
                    with safetensors.safe_open(
                        str(param_path), framework="pt", device=device
                    ) as f:
                        for n in tqdm(
                            f.keys(),
                            desc="Loading tensor parameters",
                            unit="tensors",
                        ):
                            params[n] = f.get_tensor(n)

        # prepare params (fuse, dequantize, etc)
        self.prepare_params(params)
        self.params = params

    def forward_pass(
        self,
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
        adapter_subpass: AdapterSubpass | None,
    ):

        self.forward_pass.execute(
            config=self.config,
            params=self.params,
            input_embeds=input_embeds,
            position_ids=position_ids,
            qo_indptr=qo_indptr,
            kv_cache_at_layer=kv_cache_at_layer,
            kv_page_indices=kv_page_indices,
            kv_page_indptr=kv_page_indptr,
            kv_last_page_lens=kv_last_page_lens,
            custom_mask=custom_mask,
            single_token_inference_mode=single_token_inference_mode,
            adapter_subpass=adapter_subpass,
        )
