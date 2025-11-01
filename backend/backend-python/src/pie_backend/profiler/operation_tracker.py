"""
Operation tracking module for memory profiling.

This module handles operation graph construction and tensor lifecycle tracking.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from .types import OperationNode, TensorInfo, TensorLifecycleEvent


class OperationTracker:
    """
    Tracks operations and tensor lifecycles.

    Builds an operation graph showing which operations allocate/access which tensors,
    and maintains a complete lifecycle history for every tensor.
    """

    def __init__(self):
        """Initialize the operation tracker."""
        # Tensor tracking
        self._tensor_registry: dict[int, TensorInfo] = {}
        self._tensor_lifecycle: dict[int, list[TensorLifecycleEvent]] = defaultdict(
            list
        )
        self._tensor_last_seen: dict[int, str] = {}

        # Operation tracking
        self._operation_graph: dict[str, OperationNode] = {}
        self._operation_allocations: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._operation_stack: list[str] = []

    @contextmanager
    def track_operation(self, operation_name: str, scanner):
        """
        Context manager to track memory allocations and tensor access within an operation.

        This builds an operation graph showing:
        - Which tensors were allocated
        - Which tensors were read/written
        - Memory deltas
        - Parent-child operation relationships

        Usage:
            with tracker.track_operation("model_forward", scanner):
                output = model.forward(input)

        Args:
            operation_name: Name of the operation being tracked
            scanner: TensorScanner instance for capturing snapshots
        """
        # Initialize operation node if first time seeing this operation
        if operation_name not in self._operation_graph:
            self._operation_graph[operation_name] = OperationNode(name=operation_name)

        # Track parent-child relationship
        if len(self._operation_stack) > 0:
            parent_op = self._operation_stack[-1]
            if parent_op in self._operation_graph:
                if (
                    operation_name
                    not in self._operation_graph[parent_op].child_operations
                ):
                    self._operation_graph[parent_op].child_operations.append(
                        operation_name
                    )

        # Capture memory before operation
        pre_snapshot = scanner.capture_snapshot(
            f"{operation_name}_pre",
            self.get_current_operation(),
            self.record_lifecycle_event,
        )
        pre_tensor_ids = set(self._tensor_registry.keys())

        # Set operation context
        self._operation_stack.append(operation_name)

        try:
            yield
        finally:
            # Clear operation context
            if self._operation_stack and self._operation_stack[-1] == operation_name:
                self._operation_stack.pop()

            # Capture memory after operation
            post_snapshot = scanner.capture_snapshot(
                f"{operation_name}_post",
                self.get_current_operation(),
                self.record_lifecycle_event,
            )
            post_tensor_ids = set(self._tensor_registry.keys())

            # Analyze tensor changes
            new_tensor_ids = post_tensor_ids - pre_tensor_ids
            deallocated_tensor_ids = pre_tensor_ids - post_tensor_ids
            existing_tensor_ids = pre_tensor_ids & post_tensor_ids

            # Update operation node
            op_node = self._operation_graph[operation_name]
            op_node.invocations += 1

            # Record new allocations
            for tensor_id in new_tensor_ids:
                if tensor_id in self._tensor_registry:
                    op_node.tensors_allocated.append(tensor_id)
                    self.record_lifecycle_event(
                        tensor_id,
                        "allocated",
                        operation_name,
                        self._tensor_registry[tensor_id].size_mb,
                        self._tensor_registry[tensor_id].shape,
                    )

            # Record tensor accesses (existing tensors = read)
            for tensor_id in existing_tensor_ids:
                if tensor_id not in op_node.tensors_read:
                    op_node.tensors_read.append(tensor_id)
                self.record_lifecycle_event(
                    tensor_id,
                    "accessed_read",
                    operation_name,
                    self._tensor_registry[tensor_id].size_mb,
                    self._tensor_registry[tensor_id].shape,
                )

            # Record deallocations
            for tensor_id in deallocated_tensor_ids:
                if tensor_id in self._tensor_registry:
                    self.record_lifecycle_event(
                        tensor_id,
                        "deallocated",
                        operation_name,
                        self._tensor_registry[tensor_id].size_mb,
                        self._tensor_registry[tensor_id].shape,
                    )

            # Calculate memory delta
            memory_delta_mb = post_snapshot.total_mb - pre_snapshot.total_mb
            tensor_delta = post_snapshot.tensor_count - pre_snapshot.tensor_count
            op_node.total_memory_delta_mb += memory_delta_mb

            # Record operation allocation info
            self._operation_allocations[operation_name].append(
                {
                    "memory_delta_mb": memory_delta_mb,
                    "tensor_delta": tensor_delta,
                    "tensors_allocated": len(new_tensor_ids),
                    "tensors_deallocated": len(deallocated_tensor_ids),
                    "tensors_accessed": len(existing_tensor_ids),
                    "pre_total_mb": pre_snapshot.total_mb,
                    "post_total_mb": post_snapshot.total_mb,
                    "timestamp": datetime.now().isoformat(),
                }
            )

    def record_lifecycle_event(
        self,
        tensor_id: int,
        event_type: str,
        operation: str,
        size_mb: float,
        shape: tuple,
    ) -> None:
        """
        Record a lifecycle event for a tensor.

        Args:
            tensor_id: ID of the tensor
            event_type: Type of event (allocated, accessed_read, deallocated)
            operation: Name of operation that triggered the event
            size_mb: Size of tensor in MB
            shape: Shape of tensor
        """
        event = TensorLifecycleEvent(
            tensor_id=tensor_id,
            event_type=event_type,
            operation=operation,
            timestamp=datetime.now().isoformat(),
            size_mb=size_mb,
            shape=shape,
        )
        self._tensor_lifecycle[tensor_id].append(event)

    def get_current_operation(self) -> str | None:
        """Get the name of the current operation (top of stack)."""
        return self._operation_stack[-1] if self._operation_stack else None

    def get_registry(self) -> dict[int, TensorInfo]:
        """Get the tensor registry."""
        return self._tensor_registry

    def get_lifecycle(self) -> dict[int, list[TensorLifecycleEvent]]:
        """Get the tensor lifecycle dictionary."""
        return self._tensor_lifecycle

    def get_operation_graph(self) -> dict[str, OperationNode]:
        """Get the operation graph."""
        return self._operation_graph

    def get_operation_allocations(self) -> dict[str, list[dict[str, Any]]]:
        """Get the operation allocations dictionary."""
        return self._operation_allocations

    def update_tensor_metadata(self, tensor_id: int, **kwargs) -> None:
        """
        Update metadata for a tensor in the registry.

        Args:
            tensor_id: ID of the tensor
            **kwargs: Attributes to update
        """
        if tensor_id in self._tensor_registry:
            for key, value in kwargs.items():
                setattr(self._tensor_registry[tensor_id], key, value)
