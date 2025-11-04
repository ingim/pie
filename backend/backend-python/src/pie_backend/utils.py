import os
import platform
import sys
import traceback
from pathlib import Path


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
