import fire
from .server import start_service
from .utils import terminate
from .config import ServiceConfig


# Main entry point for the server
def main(
    model: str,
    host: str = "localhost",
    port: int = 9123,
    internal_auth_token: str | None = None,
    cache_dir: str | None = None,
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
):
    """
    Runs the application with configuration provided as command-line arguments.

    Args:
        model: Name of the model to load (required).
        host: Hostname for the ZMQ service to bind to.
        port: Port for the ZMQ service to bind to.
        controller_host: Hostname of the controller to register with.
        controller_port: Port of the controller to register with.
        internal_auth_token: Internal authentication token for connecting to the controller.
        cache_dir: Directory for model cache. Defaults to PIE_HOME env var,
                   then the platform-specific user cache dir.
        kv_page_size: The size of each page in the key-value cache.
        max_dist_size: Maximum distance for embeddings.
        max_num_kv_pages: Maximum number of pages in the key-value cache.
        gpu_mem_headroom: TODO
        max_num_embeds: Maximum number of embeddings to store.
        max_batch_tokens: Maximum number of tokens in a batch.
        max_num_adapters: Maximum number of adapters that can be loaded.
        max_adapter_rank: Maximum rank for any loaded adapter.
        device: The device to run the model on (e.g., 'mps', 'cuda:0', 'cpu').
        dtype: The data type for model weights (e.g., 'bfloat16', 'float16').
        enable_profiling: Enable unified profiler (timing + tensor tracking) (default: False).
    """

    config = ServiceConfig.from_args(
        model=model,
        host=host,
        port=port,
        internal_auth_token=internal_auth_token,
        cache_dir=cache_dir,
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

    config.print()

    try:
        start_service(config)
    except ValueError as e:
        terminate(f"Error: {e}")


if __name__ == "__main__":
    fire.Fire(main)
