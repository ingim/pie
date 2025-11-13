import os
import platform
import sys
import traceback
from pathlib import Path
import psutil

import torch


def is_apple_silicon() -> bool:
    """Check if running on macOS with Apple Silicon (M1/M2/M3/M4).

    Returns:
        True if running on Apple Silicon, False otherwise
    """
    return platform.system() == "Darwin" and platform.processor() == "arm"


def resolve_cache_dir(cache_dir: str | None) -> str:
    """Resolve the cache directory using CLI arg > env var > default.

    - Windows: Uses %LOCALAPPDATA%/pie
    - Unix (Linux, macOS, etc.): Uses ~/.cache/pie for Docker compatibility
    """
    if cache_dir:
        return cache_dir

    if "PIE_HOME" in os.environ:
        return os.environ["PIE_HOME"]

    # Platform-specific cache directory (matches C++ backend in utils.hpp)
    if sys.platform == "win32":
        # Windows: Use LOCALAPPDATA for cache (standard on Windows)
        local_appdata = os.environ.get("LOCALAPPDATA")
        if not local_appdata:
            raise RuntimeError(
                "Could not determine cache directory. "
                "Please set %LOCALAPPDATA% or specify --cache-dir"
            )
        return str(Path(local_appdata) / "pie")
    else:
        # Unix (Linux, macOS): Use ~/.cache for Docker volume mount compatibility
        home = Path.home()
        return str(home / ".cache" / "pie")


def terminate(msg: str) -> None:
    """Terminate the program with a message."""
    print(f"\n[!!!] {msg} Terminating.", file=sys.stderr)
    traceback.print_exc()
    os._exit(1)


def get_device_sm():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return major * 10 + minor
    return 0


def get_available_memory(devices: list[torch.device], rank: int = 0) -> int:
    device = devices[rank]

    is_cuda = device.type == "cuda"
    is_cpu = device.type in ("cpu", "mps")

    # Clear cache on all CUDA devices to get a better measurement
    if is_cuda and torch.cuda.is_available():
        with torch.cuda.device(device.index):
            torch.cuda.empty_cache()

    total_free_bytes = []

    if is_cuda:
        if get_device_sm() in (87, 110, 121):  # Orin, Thor, Spark
            total_free_bytes = psutil.virtual_memory().available
        else:
            total_free_bytes, _ = torch.cuda.mem_get_info(device)

    if is_cpu:
        total_free_bytes = psutil.virtual_memory().available

    if not total_free_bytes:
        raise RuntimeError("No supported devices found in the devices list")

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and len(devices) > 1
    ):
        tensor = torch.tensor(total_free_bytes, dtype=torch.uint64)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
        total_free_bytes = int(tensor.item())

    return total_free_bytes
