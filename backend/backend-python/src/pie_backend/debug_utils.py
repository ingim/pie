"""Debug utilities for Metal and Python backends."""

import os


def checkpoint_validation(*_args, **_kwargs):
    """No-op decorator for checkpoint validation.

    Accepts any arguments including checkpoint_name, capture_tensors,
    include_metadata, etc. but ignores them and returns the original function.
    Used as a placeholder since debug framework is not available.
    """

    def decorator(func):
        return func

    return decorator


def is_tensor_debug_enabled() -> bool:
    """Check if MetalTensorDebug logging is enabled via environment variable."""
    return os.environ.get("METAL_DEBUG_TENSOR", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def is_capture_debug_enabled() -> bool:
    """Check if MetalCaptureDebug logging is enabled via environment variable."""
    return os.environ.get("METAL_DEBUG_CAPTURE", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
