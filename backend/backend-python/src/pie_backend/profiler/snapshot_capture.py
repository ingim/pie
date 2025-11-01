"""
Snapshot capture module for memory profiling.

This module handles GC-based tensor scanning and snapshot creation.
"""

from __future__ import annotations

import gc
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from typing import Any, Callable

import torch

from .analysis_utils import (
    classify_tensor_purpose,
    is_persistent_tensor,
    is_reusable_tensor,
)
from .types import MemorySnapshot, TensorInfo


class TensorScanner:
    """
    Scans memory for PyTorch tensors and creates memory snapshots.

    This scanner uses Python's garbage collector to find all live PyTorch tensors,
    classifies them, and tracks their lifecycle across snapshots.
    """

    def __init__(
        self,
        tensor_registry: dict[int, TensorInfo],
        tensor_last_seen: dict[int, str],
    ):
        """
        Initialize the tensor scanner.

        Args:
            tensor_registry: Shared registry mapping tensor ID to TensorInfo
            tensor_last_seen: Shared dict tracking when tensors were last seen
        """
        self._tensor_registry = tensor_registry
        self._tensor_last_seen = tensor_last_seen

    def capture_snapshot(
        self,
        checkpoint_name: str,
        current_operation: str | None,
        lifecycle_recorder: Callable,
    ) -> MemorySnapshot:
        """
        Capture current memory state across all devices.

        Args:
            checkpoint_name: Name for this checkpoint
            current_operation: Name of current operation (if any)
            lifecycle_recorder: Callback to record lifecycle events

        Returns:
            MemorySnapshot with current memory state
        """
        # Collect all tensors
        tensors: list[TensorInfo] = []
        device_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"total_bytes": 0, "total_mb": 0.0, "count": 0, "tensors": []}
        )

        current_tensor_ids = set()
        previous_tensor_ids = set(self._tensor_registry.keys())

        # Build lightweight tensor name hints from nn.Module parameters/buffers
        tensor_name_hints = self._collect_tensor_name_hints()

        # Scan all objects for tensors
        for obj in gc.get_objects():
            try:
                # Check object type by class name to avoid isinstance() and hasattr() issues
                obj_class = type(obj)
                class_name = obj_class.__name__
                module_name = getattr(obj_class, "__module__", "")

                # Check if it's a Tensor by class name
                if class_name == "Tensor" and "torch" in module_name:
                    tensor_id = self._process_tensor(
                        obj,
                        tensor_name_hints,
                        checkpoint_name,
                        current_operation,
                        tensors,
                        device_stats,
                        lifecycle_recorder,
                    )
                    current_tensor_ids.add(tensor_id)

            except (AttributeError, RuntimeError, TypeError):
                # Skip objects that can't be inspected
                pass

        # Detect deallocated tensors
        self._detect_deallocations(
            previous_tensor_ids,
            current_tensor_ids,
            current_operation,
            lifecycle_recorder,
        )

        # Build and return snapshot
        return self._build_snapshot_result(checkpoint_name, tensors, device_stats)

    def _collect_tensor_name_hints(self) -> dict[int, str]:
        """
        Collect tensor name hints from nn.Module parameters and buffers.

        Returns:
            Dictionary mapping tensor id to parameter/buffer name
        """
        tensor_name_hints: dict[int, str] = {}

        for obj in gc.get_objects():
            try:
                obj_class = type(obj)
                class_name = obj_class.__name__
                module_name = getattr(obj_class, "__module__", "")

                # Check if it's an nn.Module by class hierarchy
                if "Module" in class_name and "torch.nn" in module_name:
                    try:
                        # Check if methods exist by looking in __dict__ or using callable
                        if callable(getattr(obj, "named_parameters", None)):
                            for name, param in obj.named_parameters(recurse=False):
                                if param is not None:
                                    tensor_name_hints[id(param)] = name
                        if callable(getattr(obj, "named_buffers", None)):
                            for name, buffer in obj.named_buffers(recurse=False):
                                if buffer is not None:
                                    tensor_name_hints[id(buffer)] = name
                    except (AttributeError, RuntimeError, TypeError):
                        pass
            except (AttributeError, RuntimeError, TypeError):
                pass

        return tensor_name_hints

    def _process_tensor(
        self,
        tensor: torch.Tensor,
        tensor_name_hints: dict[int, str],
        checkpoint_name: str,
        current_operation: str | None,
        tensors: list[TensorInfo],
        device_stats: dict[str, dict[str, Any]],
        lifecycle_recorder: Callable,
    ) -> int:
        """
        Process a single tensor: classify, register/update, record lifecycle events.

        Args:
            tensor: The tensor to process
            tensor_name_hints: Mapping of tensor id to parameter/buffer names
            checkpoint_name: Name of the current checkpoint
            current_operation: Name of the current operation (if any)
            tensors: List to append tensor info to
            device_stats: Dictionary to update with device statistics
            lifecycle_recorder: Callback to record lifecycle events

        Returns:
            Tensor ID
        """
        tensor_id = id(tensor)

        size_bytes = tensor.element_size() * tensor.nelement()
        size_mb = size_bytes / (1024**2)
        device = str(tensor.device)
        shape = tuple(tensor.shape)

        # Get tensor name hint if available
        tensor_name = tensor_name_hints.get(tensor_id)

        # Classify tensor purpose
        purpose = classify_tensor_purpose(tensor, shape, tensor_name)
        is_persistent = is_persistent_tensor(tensor_id, size_mb, self._tensor_last_seen)
        is_reusable = is_reusable_tensor(tensor, purpose)

        # Check if this is a new tensor or update to registry
        if tensor_id not in self._tensor_registry:
            # New tensor - register it and record allocation
            tensor_info = TensorInfo(
                tensor_id=tensor_id,
                size_bytes=size_bytes,
                size_mb=size_mb,
                shape=shape,
                dtype=str(tensor.dtype),
                device=device,
                requires_grad=tensor.requires_grad,
                allocated_by=current_operation or "unknown",
                purpose=purpose,
                is_persistent=is_persistent,
                is_reusable=is_reusable,
            )
            self._tensor_registry[tensor_id] = tensor_info

            # Record allocation event
            lifecycle_recorder(
                tensor_id,
                "allocated",
                current_operation or "background",
                size_mb,
                shape,
            )
        else:
            # Existing tensor - update info
            tensor_info = self._tensor_registry[tensor_id]

            # Record access if we haven't seen it in this snapshot yet
            if self._tensor_last_seen.get(tensor_id) != checkpoint_name:
                lifecycle_recorder(
                    tensor_id,
                    "accessed_read",
                    current_operation or "background",
                    size_mb,
                    shape,
                )

        tensors.append(tensor_info)

        # Update device statistics
        device_stats[device]["total_bytes"] += size_bytes
        device_stats[device]["total_mb"] += size_mb
        device_stats[device]["count"] += 1
        device_stats[device]["tensors"].append(asdict(tensor_info))

        # Track last seen
        self._tensor_last_seen[tensor_id] = checkpoint_name

        return tensor_id

    def _detect_deallocations(
        self,
        previous_tensor_ids: set[int],
        current_tensor_ids: set[int],
        current_operation: str | None,
        lifecycle_recorder: Callable,
    ) -> None:
        """
        Detect and record deallocation events for tensors that are no longer alive.

        Args:
            previous_tensor_ids: Set of tensor IDs from previous snapshot
            current_tensor_ids: Set of tensor IDs in current snapshot
            current_operation: Name of the current operation (if any)
            lifecycle_recorder: Callback to record lifecycle events
        """
        deallocated_tensor_ids = previous_tensor_ids - current_tensor_ids
        for tensor_id in deallocated_tensor_ids:
            if tensor_id in self._tensor_registry:
                tensor_info = self._tensor_registry[tensor_id]
                lifecycle_recorder(
                    tensor_id,
                    "deallocated",
                    current_operation or "background",
                    tensor_info.size_mb,
                    tensor_info.shape,
                )

    def _build_snapshot_result(
        self,
        checkpoint_name: str,
        tensors: list[TensorInfo],
        device_stats: dict[str, dict[str, Any]],
    ) -> MemorySnapshot:
        """
        Build the final MemorySnapshot object from collected tensor data.

        Args:
            checkpoint_name: Name of this checkpoint
            tensors: List of all tensor info objects
            device_stats: Dictionary of device statistics

        Returns:
            MemorySnapshot object
        """
        # Calculate totals
        total_bytes = sum(t.size_bytes for t in tensors)
        total_mb = total_bytes / (1024**2)

        # Get top 10 largest allocations
        top_tensors = sorted(tensors, key=lambda t: t.size_bytes, reverse=True)[:10]
        top_allocations = [asdict(t) for t in top_tensors]

        # Convert device_stats to regular dict
        device_breakdown = dict(device_stats)

        return MemorySnapshot(
            timestamp=datetime.now().isoformat(),
            checkpoint_name=checkpoint_name,
            total_bytes=total_bytes,
            total_mb=total_mb,
            device_breakdown=device_breakdown,
            top_allocations=top_allocations,
            tensor_count=len(tensors),
        )
