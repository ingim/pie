"""
PyTorch profiler integration module.

This module integrates with PyTorch's built-in profiler to capture low-level operations.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch


class PyTorchProfiler:
    """
    Integrates with PyTorch's built-in profiler.

    Captures low-level PyTorch operations (gemm, matmul, add, etc.) and builds
    an operation graph showing: op_name → input_tensors → output_tensors
    """

    def __init__(self):
        """Initialize the PyTorch profiler integration."""
        self._pytorch_ops: list[dict[str, Any]] = []

    @contextmanager
    def track_pytorch_ops(self, scanner):
        """
        Context manager to track low-level PyTorch operations.

        This uses PyTorch's profiler to capture every operation and build a complete
        operation graph.

        Usage:
            with pytorch_profiler.track_pytorch_ops(scanner):
                output = model.forward(input)

        Args:
            scanner: TensorScanner instance for capturing snapshots
        """
        # Clear previous ops
        self._pytorch_ops.clear()

        # Capture memory snapshot before
        _ = scanner.capture_snapshot(
            "pytorch_ops_pre",
            None,  # No current operation
            lambda *args: None,  # No-op lifecycle recorder
        )

        # Use PyTorch profiler with memory tracking
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,  # Stack traces are expensive
            with_modules=False,
        ) as prof:
            yield prof

        # Capture memory snapshot after
        _ = scanner.capture_snapshot(
            "pytorch_ops_post",
            None,  # No current operation
            lambda *args: None,  # No-op lifecycle recorder
        )

        # Extract operation information from profiler
        self._extract_pytorch_ops_from_profiler(prof)

    def _extract_pytorch_ops_from_profiler(self, prof) -> None:
        """
        Extract operation information from PyTorch profiler.

        Args:
            prof: PyTorch profiler object
        """
        try:
            for evt in prof.key_averages():
                # Skip profiler overhead events
                if "profiler" in evt.key.lower():
                    continue

                op_info = {
                    "op_name": evt.key,
                    "count": evt.count,
                    "cpu_time_us": evt.cpu_time_total,
                    "cuda_time_us": (
                        evt.cuda_time_total if hasattr(evt, "cuda_time_total") else 0
                    ),
                    "cpu_memory_mb": (
                        evt.cpu_memory_usage / (1024 * 1024)
                        if hasattr(evt, "cpu_memory_usage")
                        else 0
                    ),
                    "cuda_memory_mb": (
                        evt.cuda_memory_usage / (1024 * 1024)
                        if hasattr(evt, "cuda_memory_usage")
                        else 0
                    ),
                    "input_shapes": (
                        evt.input_shapes if hasattr(evt, "input_shapes") else []
                    ),
                }
                self._pytorch_ops.append(op_info)
        except (AttributeError, RuntimeError) as e:
            print(f"⚠️  Failed to extract PyTorch ops: {e}")

    def get_ops(self) -> list[dict[str, Any]]:
        """Get the list of captured PyTorch operations."""
        return self._pytorch_ops
