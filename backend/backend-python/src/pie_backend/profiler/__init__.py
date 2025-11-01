"""
Unified profiling utilities for PyTorch models.

This package combines timing profiling, memory tracking, and tensor flow analysis.
Exports APIs from both the legacy profiler and the unified tracker.
"""

import time
from contextlib import contextmanager

# Import from unified tracker (main profiler)
from .tracker import (
    MemoryTracker,
    get_memory_summary,
    get_memory_tracker,
    initialize_memory_tracker,
    memory_checkpoint,
    save_memory_profile,
    stop_memory_tracker,
)

# Import from legacy profiler (for backward compatibility)
from .legacy_profiler import (
    set_profiling_enabled,
    profile_with_tensors,
)

from .calculate_metrics import calculate_attention_metrics


@contextmanager
def start_profile(name: str):
    """
    Profile an operation with timing and optional tensor tracking (unified profiler).

    This is a convenience wrapper that delegates to the global memory tracker's
    start_profile method. Compatible with the old profiler.start_profile() API.

    Usage:
        from profiler import start_profile

        with start_profile("operation_name"):
            # code to profile

    Args:
        name: Name of the operation (e.g., "forward_pass", "attn_metal_kernel")
    """
    tracker = get_memory_tracker()
    with tracker.start_profile(name):
        yield


@contextmanager
def profile_attention(layer_idx: int, query_states, kv_cache):
    """
    Profile attention operation with automatic metrics calculation.

    This is a specialized profiler for attention operations that automatically
    calculates FLOPs and data volume for bottleneck analysis.

    Usage:
        from profiler import profile_attention

        with profile_attention(layer_idx, query_states, kv_cache):
            output = runtime.run_attention(...)

    Args:
        layer_idx: Layer index for naming
        query_states: Query tensor [batch, num_heads, head_dim]
        kv_cache: KV cache tensor [kv_len, 2, num_kv_heads, head_dim]
    """
    tracker = get_memory_tracker()

    # If profiling disabled, just yield without overhead
    if not tracker or not tracker.enabled:
        yield
        return

    # Calculate metrics before timing
    batch_size, num_heads, head_dim = query_states.shape
    kv_seq_len = kv_cache.shape[0]
    metrics = calculate_attention_metrics(
        batch_size=batch_size,
        seq_len=1,
        num_heads=num_heads,
        head_dim=head_dim,
        kv_seq_len=kv_seq_len,
        dtype=str(query_states.dtype).rsplit(".", maxsplit=1)[-1],
    )

    # Time the operation with synchronization
    MemoryTracker.synchronize_device()
    start_time = time.perf_counter()

    try:
        yield
    finally:
        MemoryTracker.synchronize_device()
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Add to operation log for bottleneck analysis
        hook_tracker = tracker.get_hook_tracker()
        hook_tracker.log_custom_operation(
            name=f"attention_layer_{layer_idx}",
            module_type="Attention",
            duration_ms=duration_ms,
            flops=metrics["flops"],
            data_bytes=metrics["data_bytes"],
        )


__all__ = [
    # Unified tracker API
    "initialize_memory_tracker",
    "get_memory_tracker",
    "memory_checkpoint",
    "save_memory_profile",
    "get_memory_summary",
    "stop_memory_tracker",
    "start_profile",
    "profile_attention",
    # Legacy profiler API (backward compatibility)
    "set_profiling_enabled",
    "profile_with_tensors",
]
