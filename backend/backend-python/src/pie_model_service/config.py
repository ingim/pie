import sys
from dataclasses import dataclass, asdict
from typing import Any
from .utils import resolve_cache_dir, terminate


@dataclass
class ServerConfig:
    """
    Configuration for the server, validated upon creation.

    Use the `ServerConfig.from_args(...)` factory method to create
    a new instance from raw arguments.
    """

    # Required args
    model: str
    cache_dir: str
    host: str = "localhost"
    port: int = 10123
    internal_auth_token: str | None = None
    kv_page_size: int = 16
    max_dist_size: int = 64
    max_num_embeds: int = 128
    max_batch_tokens: int = 10240
    max_num_adapters: int = 48
    max_adapter_rank: int = 8
    max_num_kv_pages: int | None = None
    gpu_mem_headroom: float | None = None
    device: str | None = None
    dtype: str = "bfloat16"
    enable_profiling: bool = False

    @classmethod
    def from_args(
        cls,
        model: str,
        host: str = "localhost",
        port: int = 9123,
        internal_auth_token: str | None = None,
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
    ) -> "ServerConfig":
        """
        Factory method to build a validated and resolved ServerConfig.
        This replaces the original `build_config` logic.
        """

        # 1. Validation (from build_config)
        if max_num_kv_pages is None and gpu_mem_headroom is None:
            terminate(
                "Config must contain either 'max_num_kv_pages' or 'gpu_mem_headroom'."
            )

        # 2. Resolution (from build_config)
        resolved_cache_dir = resolve_cache_dir(cache_dir)

        # 3. Create the immutable config instance
        return cls(
            model=model,
            host=host,
            port=port,
            internal_auth_token=internal_auth_token,
            cache_dir=resolved_cache_dir,  # Use the resolved value
            kv_page_size=kv_page_size,
            max_dist_size=max_dist_size,
            max_num_embeds=max_num_embeds,
            max_batch_tokens=max_batch_tokens,
            max_num_adapters=max_num_adapters,
            max_adapter_rank=max_adapter_rank,
            max_num_kv_pages=max_num_kv_pages,
            gpu_mem_headroom=gpu_mem_headroom,
            device=device,
            dtype=dtype,
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


# --- Example Usage ---

if __name__ == "__main__":

    print("--- 1. Creating a valid config ---")
    try:
        # The main() logic is now replaced by calling the factory method
        config = ServerConfig.from_args(
            model="my-cool-model",
            max_num_kv_pages=1024,  # Satisfies the validation rule
            device="cuda:0",
            internal_auth_token="my-secret-token",
        )

        # The print_config() function is now a method
        config.print()

        # You can access properties directly
        print(f"\nModel is: {config.model}")
        print(f"Cache dir is: {config.cache_dir}")

    except ValueError as e:
        print(f"Configuration failed: {e}")

    print("\n--- 2. Creating an invalid config (to test validation) ---")
    try:
        # This will fail the validation check
        invalid_config = ServerConfig.from_args(
            model="my-bad-model"
            # Missing max_num_kv_pages AND gpu_mem_headroom
        )
    except ValueError as e:
        print(f"Configuration correctly failed as expected.")
