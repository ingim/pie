from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, Any

import torch
from torchao.quantization import (
    Int4WeightOnlyConfig,
    Int8WeightOnlyConfig,
    Float8WeightOnlyConfig,
)


@dataclass
class Config:
    devices: list[torch.device]
    rank: int
    dtype: torch.dtype
    quantization: (
        Int4WeightOnlyConfig | Int8WeightOnlyConfig | Float8WeightOnlyConfig | None
    )

    kv_page_size: int
    max_dist_size: int
    max_num_embeds: int
    max_batch_tokens: int | None
    max_num_adapters: int
    max_adapter_rank: int
    max_num_kv_pages: int | None
    mem_utilization: float

    @property
    def device(self) -> torch.device:
        return self.devices[self.rank]

    @property
    def world_size(self) -> int:
        return len(self.devices)

    @staticmethod
    def _parse_dtype(dtype: str) -> torch.dtype:
        if not hasattr(torch, dtype):
            raise ValueError(f"Invalid torch.dtype string: '{dtype}'")
        return getattr(torch, dtype)

    @staticmethod
    def _parse_devices(devices: list[str]) -> list[torch.device]:
        return [torch.device(d) for d in devices]

    @staticmethod
    def _parse_quantization(q: dict[str, Any] | None):
        if q is None:
            return None

        qtype = q.get("type")
        if qtype is None:
            raise ValueError("Quantization config must contain a 'type' field.")

        match qtype.lower():
            case "int4":
                valid_group_sizes = {256, 128, 64, 32}
                valid_algorithms = {"tinygemm", "hqq"}

                group_size = q.get("group_size")
                algorithm = q.get("algorithm")

                if group_size is not None:
                    if group_size not in valid_group_sizes:
                        raise ValueError(
                            f"Invalid group_size={group_size} for int4 "
                            f"(allowed: {sorted(valid_group_sizes)})"
                        )

                if algorithm is not None:
                    if algorithm not in valid_algorithms:
                        raise ValueError(
                            "Invalid algorithm '{algorithm}' for int4. "
                            f"(allowed: {sorted(valid_algorithms)})"
                        )

                return Int4WeightOnlyConfig(
                    group_size=group_size,
                    int4_choose_qparams_algorithm=algorithm,
                )

            case "int8":
                return Int8WeightOnlyConfig(
                    group_size=q.get("group_size"),
                )
            case "float8":
                return Float8WeightOnlyConfig()
            case "none" | None:
                return None
            case _:
                raise ValueError(
                    f"Unknown quantization type '{qtype}'. "
                    f"Expected one of: 'int4', 'int8', 'float8', 'none'"
                )

    @staticmethod
    def from_dict(config: dict) -> Config:

        dtype = Config._parse_dtype(config.get("dtype"))
        devices = Config._parse_devices(config.get("devices"))
        quantization = Config._parse_quantization(config.get("quantization"))

        return Config(
            devices=devices,
            rank=config["rank"],
            dtype=dtype,
            quantization=quantization,
            kv_page_size=config["kv_page_size"],
            max_dist_size=config["max_dist_size"],
            max_num_embeds=config["max_num_embeds"],
            max_batch_tokens=config.get("max_batch_tokens"),
            max_num_adapters=config["max_num_adapters"],
            max_adapter_rank=config["max_adapter_rank"],
            max_num_kv_pages=config.get("max_num_kv_pages"),
            mem_utilization=config["mem_utilization"],
        )


# Model architecture specification
@dataclass
class Spec(ABC):

    @staticmethod
    @abstractmethod
    def from_dict(spec: dict) -> Spec:
        """Construct a Spec object from a configuration dictionary."""
        pass


@dataclass
class Param(ABC):
    spec: Spec
    config: Config

    @staticmethod
    @abstractmethod
    def from_reader(
        spec: Spec,
        config: Config,
        read: Callable[..., torch.Tensor],
    ) -> Param:
        """Create a Param-like object from a tensor reader and spec."""
        pass


@dataclass
class Buffer(ABC):
    spec: Spec
    config: Config

    @staticmethod
    @abstractmethod
    def from_config(
        spec: Spec,
        config: Config,
    ) -> Buffer:
        """Create a Param-like object from a tensor reader and spec."""
        pass
