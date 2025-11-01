"""
Timing profiler module for performance profiling.

This module handles hierarchical timing tree construction and GPU synchronization.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
import torch

from .types import TreeNode


class TimingProfiler:
    """
    Hierarchical timing profiler with GPU synchronization.

    Builds a timing tree showing execution times for nested operations,
    with proper GPU synchronization for accurate measurements.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize the timing profiler.

        Args:
            enabled: Whether timing profiling is enabled
        """
        self._enabled = enabled
        self._timing_tree: dict[str, TreeNode] = {}
        self._timing_stack: list[str] = []

    @contextmanager
    def start_profile(self, name: str, tensor_registry: dict | None = None):
        """
        Profile an operation with timing and optional tensor tracking.

        This builds a hierarchical timing tree and optionally captures tensor I/O.

        Usage:
            with profiler.start_profile("operation_name"):
                # code to profile

        Args:
            name: Name of the operation (e.g., "attn_metal_kernel", "forward_pass")
            tensor_registry: Optional tensor registry for tensor I/O tracking
        """
        # If profiling is disabled, yield without overhead
        if not self._enabled:
            yield
            return

        # Build hierarchical path based on current stack
        parent_path = self._timing_stack[-1] if self._timing_stack else None
        full_path = f"{parent_path}.{name}" if parent_path else name

        # Create or get tree node
        if full_path not in self._timing_tree:
            self._timing_tree[full_path] = TreeNode(
                name=name,
                full_path=full_path,
                parent=parent_path,
            )

        # Track parent-child relationship
        if parent_path and parent_path in self._timing_tree:
            if full_path not in self._timing_tree[parent_path].children:
                self._timing_tree[parent_path].children.append(full_path)

        node = self._timing_tree[full_path]

        # Capture pre-state for tensor tracking (if enabled)
        pre_tensors = set(tensor_registry.keys()) if tensor_registry else set()

        # Synchronize GPU before timing to ensure accurate measurements
        self.synchronize_device()
        start_time = time.perf_counter()

        # Push onto timing stack
        self._timing_stack.append(full_path)

        try:
            yield
        finally:
            # Synchronize GPU after operation to capture actual execution time
            self.synchronize_device()
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            # Record timing
            node.times.append(duration_ms)
            node.count += 1

            # Update statistics
            node.avg_ms = float(np.mean(node.times))
            node.min_ms = float(np.min(node.times))
            node.max_ms = float(np.max(node.times))
            node.std_dev_ms = float(np.std(node.times)) if len(node.times) > 1 else 0.0

            # Capture tensor I/O if tensor registry provided (only on first invocation)
            if tensor_registry and node.count == 1:
                self._capture_tensor_io(node, pre_tensors, tensor_registry)

            # Pop from timing stack
            if self._timing_stack and self._timing_stack[-1] == full_path:
                self._timing_stack.pop()

    def _capture_tensor_io(
        self,
        node: TreeNode,
        pre_tensors: set[int],
        tensor_registry: dict,
    ) -> None:
        """
        Capture typical input/output tensors for this operation.

        Args:
            node: Tree node to update
            pre_tensors: Set of tensor IDs before operation
            tensor_registry: Tensor registry for looking up tensor info
        """
        post_tensors = set(tensor_registry.keys())
        new_tensors = post_tensors - pre_tensors
        accessed_tensors = pre_tensors & post_tensors

        # Record inputs (tensors that existed before)
        node.typical_input_tensors = [
            {
                "id": tid,
                "shape": list(tensor_registry[tid].shape),
                "dtype": tensor_registry[tid].dtype,
                "device": tensor_registry[tid].device,
                "size_mb": round(tensor_registry[tid].size_mb, 3),
            }
            for tid in list(accessed_tensors)[:5]  # Limit to 5 typical inputs
            if tid in tensor_registry
        ]

        # Record outputs (new tensors created)
        node.typical_output_tensors = [
            {
                "id": tid,
                "shape": list(tensor_registry[tid].shape),
                "dtype": tensor_registry[tid].dtype,
                "device": tensor_registry[tid].device,
                "size_mb": round(tensor_registry[tid].size_mb, 3),
            }
            for tid in list(new_tensors)[:5]  # Limit to 5 typical outputs
            if tid in tensor_registry
        ]

    @staticmethod
    def synchronize_device() -> None:
        """
        Synchronize GPU operations to ensure accurate timing measurements.

        This is critical for profiling GPU operations because without synchronization,
        timing only measures how long it takes to queue operations, not execute them.
        """
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()

    def get_timing_tree(self) -> dict[str, TreeNode]:
        """Get the timing tree."""
        return self._timing_tree

    def is_enabled(self) -> bool:
        """Check if timing profiling is enabled."""
        return self._enabled
