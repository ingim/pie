"""
Hook-Based Operation Tracker

Uses PyTorch forward hooks to directly track tensor IDs flowing through operations.
This provides accurate operation → tensor mappings without relying on profiler memory data.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Any
import time

import torch
from torch import nn

from .calculate_metrics import calculate_operation_metrics


class HookBasedTracker:
    """
    Tracks operations using forward hooks to capture exact tensor IDs.

    This is more reliable than profiler-based tracking because it directly
    observes which tensor objects flow through each operation.
    """

    def __init__(self, base_tracker):
        """Initialize with reference to base memory tracker."""
        self.base_tracker = base_tracker
        self.enabled = base_tracker.enabled

        # Storage for hook data
        self._hook_data = []
        self._hook_handles = []
        self._operation_log = []  # Operation-centric log with timestamps
        self._operation_start_times = {}  # Track start time per module

    @contextmanager
    def track_model_with_hooks(self, model: nn.Module):
        """
        Track model forward pass using hooks.

        Args:
            model: PyTorch model to track

        Yields:
            None
        """
        if not self.enabled:
            yield
            return

        # Clear previous data
        self._hook_data.clear()
        self._hook_handles.clear()

        # Register hooks on all leaf modules
        for name, module in model.named_modules():
            # Only hook leaf modules (those with no children)
            if len(list(module.children())) == 0:
                # Register pre-hook for timing start
                pre_handle = module.register_forward_pre_hook(
                    self._create_pre_hook(name)
                )
                self._hook_handles.append(pre_handle)

                # Register post-hook for timing end and data capture
                post_handle = module.register_forward_hook(self._create_hook(name))
                self._hook_handles.append(post_handle)

        # Capture a snapshot BEFORE forward pass to register existing tensors
        # (weights, kv_cache, etc.) so hooks can update their allocated_by field
        self.base_tracker.capture_snapshot("pre_forward_hook")

        try:
            yield
        finally:
            # Remove hooks
            for handle in self._hook_handles:
                handle.remove()
            self._hook_handles.clear()

            # Capture post-forward snapshot to register NEW tensors created during forward
            self.base_tracker.capture_snapshot("post_forward_hook")

            # Process hook data
            self._process_hook_data()

    def _create_pre_hook(self, module_name: str):
        """Create a pre-forward hook function for timing."""

        def pre_hook(module, input_tensors):
            _ = module
            _ = input_tensors
            # Record start time for this module
            self._operation_start_times[module_name] = time.perf_counter()

        return pre_hook

    def _create_hook(self, module_name: str):
        """Create a forward hook function for a module."""

        def hook(module, input_tensors, output):
            # Capture timestamp when operation executes
            timestamp = datetime.now().isoformat()

            # Calculate duration
            duration_ms = 0.0
            if module_name in self._operation_start_times:
                duration_sec = (
                    time.perf_counter() - self._operation_start_times[module_name]
                )
                duration_ms = duration_sec * 1000  # Convert to milliseconds
                del self._operation_start_times[module_name]  # Clean up

            # Extract tensor IDs
            input_ids = []
            for inp in input_tensors:
                if isinstance(inp, torch.Tensor):
                    input_ids.append(id(inp))
                elif isinstance(inp, (tuple, list)):
                    for t in inp:
                        if isinstance(t, torch.Tensor):
                            input_ids.append(id(t))

            output_ids = []
            if isinstance(output, torch.Tensor):
                output_ids.append(id(output))
            elif isinstance(output, (tuple, list)):
                for t in output:
                    if isinstance(t, torch.Tensor):
                        output_ids.append(id(t))

            # Get shapes and dtype for validation
            input_shapes = []
            for inp in input_tensors:
                if isinstance(inp, torch.Tensor):
                    input_shapes.append(tuple(inp.shape))

            output_shape = None
            output_dtype = "float16"  # Default
            if isinstance(output, torch.Tensor):
                output_shape = tuple(output.shape)
                output_dtype = str(output.dtype).rsplit(".", maxsplit=1)[-1]

            # Store hook data with timestamp and duration
            self._hook_data.append(
                {
                    "timestamp": timestamp,
                    "duration_ms": duration_ms,
                    "module_name": module_name,
                    "module_type": module.__class__.__name__,
                    "input_tensor_ids": input_ids,
                    "output_tensor_ids": output_ids,
                    "input_shapes": input_shapes,
                    "output_shape": output_shape,
                    "dtype": output_dtype,
                }
            )

        return hook

    def _process_hook_data(self):
        """
        Build operation-centric log from captured hook data.

        This creates a timeline of operation executions with input/output tensors.
        """
        # Build operation log (operation-centric view)
        self._operation_log = []
        for idx, hook_info in enumerate(self._hook_data):
            op_name = f"{hook_info['module_type']}.{hook_info['module_name']}"

            # Calculate FLOPs and data volume metrics
            metrics = calculate_operation_metrics(
                hook_info["module_type"],
                hook_info["input_shapes"],
                hook_info["output_shape"],
                dtype=hook_info.get("dtype", "float16"),
            )

            operation_entry = {
                "operation_id": idx + 1,
                "timestamp": hook_info["timestamp"],
                "duration_ms": hook_info["duration_ms"],
                "name": op_name,
                "module_type": hook_info["module_type"],
                "module_name": hook_info["module_name"],
                "input_tensor_ids": hook_info["input_tensor_ids"],
                "output_tensor_ids": hook_info["output_tensor_ids"],
                "input_shapes": hook_info["input_shapes"],
                "output_shape": hook_info["output_shape"],
                "num_inputs": len(hook_info["input_tensor_ids"]),
                "num_outputs": len(hook_info["output_tensor_ids"]),
                # Add metrics for bottleneck analysis
                "flops": metrics["flops"],
                "data_bytes": metrics["total_data_bytes"],
                "arithmetic_intensity": metrics["arithmetic_intensity"],
            }
            self._operation_log.append(operation_entry)

        # Store in base tracker for export
        for entry in self._operation_log:
            self.base_tracker.add_operation_log_entry(entry)

        # Also update tensor metadata (for backward compatibility)
        for hook_info in self._hook_data:
            op_name = f"{hook_info['module_type']}.{hook_info['module_name']}"

            # Update output tensors - mark them as allocated by this operation
            for tensor_id in hook_info["output_tensor_ids"]:
                self.base_tracker.update_tensor_metadata(
                    tensor_id, allocated_by=op_name
                )

    def get_hook_data(self) -> list[dict[str, Any]]:
        """Get the captured hook data."""
        return self._hook_data

    def log_custom_operation(
        self,
        name: str,
        module_type: str,
        duration_ms: float,
        flops: int,
        data_bytes: int,
        input_shapes: list | None = None,
        output_shape: tuple | None = None,
    ):
        """
        Manually log a custom operation (e.g., attention kernel).

        Args:
            name: Operation name (e.g., "Attention.layer_0")
            module_type: Type of operation (e.g., "Attention")
            duration_ms: Duration in milliseconds
            flops: Total floating point operations
            data_bytes: Total data transferred in bytes
            input_shapes: Optional list of input tensor shapes
            output_shape: Optional output tensor shape
        """
        if not self.enabled:
            return

        # Create operation entry similar to hook-based entries
        operation_entry = {
            "operation_id": self.base_tracker.get_operation_log_count() + 1,
            "timestamp": datetime.now().isoformat(),
            "duration_ms": duration_ms,
            "name": name,
            "module_type": module_type,
            "module_name": name,  # Use full name as module_name
            "input_tensor_ids": [],  # Not tracked for custom ops
            "output_tensor_ids": [],
            "input_shapes": input_shapes or [],
            "output_shape": output_shape,
            "num_inputs": len(input_shapes) if input_shapes else 0,
            "num_outputs": 1 if output_shape else 0,
            "flops": flops,
            "data_bytes": data_bytes,
            "arithmetic_intensity": flops / max(data_bytes, 1),
        }

        self.base_tracker.add_operation_log_entry(operation_entry)


def create_hook_tracker(base_tracker):
    """Create a hook-based tracker from a base tracker."""
    return HookBasedTracker(base_tracker)
