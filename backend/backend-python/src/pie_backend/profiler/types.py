"""Data types for the profiler module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TensorInfo:
    """Information about a single tensor allocation."""

    tensor_id: int  # Unique ID for this tensor (id(tensor))
    size_bytes: int
    size_mb: float
    shape: tuple[int, ...]
    dtype: str
    device: str
    requires_grad: bool
    allocated_by: str | None = None  # Which operation allocated this tensor
    purpose: str | None = None  # buffer, output, weight, cache, etc.
    is_persistent: bool = False  # Does it live across operations?
    is_reusable: bool = False  # Can memory be reused?


@dataclass
class MemorySnapshot:
    """Snapshot of memory state at a specific point in time."""

    timestamp: str
    checkpoint_name: str
    total_bytes: int
    total_mb: float
    device_breakdown: dict[str, dict[str, Any]]
    top_allocations: list[dict[str, Any]]
    tensor_count: int


@dataclass
class TensorLifecycleEvent:
    """Records an event in a tensor's lifecycle."""

    tensor_id: int
    event_type: str  # 'allocated', 'accessed_read', 'accessed_write', 'deallocated'
    operation: str
    timestamp: str
    size_mb: float
    shape: tuple[int, ...]


@dataclass
class OperationNode:
    """Represents an operation in the execution graph."""

    name: str
    invocations: int = 0
    total_memory_delta_mb: float = 0.0
    tensors_allocated: list[int] = field(default_factory=list)
    tensors_read: list[int] = field(default_factory=list)
    tensors_written: list[int] = field(default_factory=list)
    child_operations: list[str] = field(default_factory=list)


@dataclass
class TreeNode:
    """Represents a timing node in the hierarchical profiling tree (unified profiler)."""

    name: str  # Short name (e.g., "attn_metal_kernel")
    full_path: str  # Full hierarchical path (e.g., "forward.layer0.attn_metal_kernel")
    parent: str | None = None  # Full path of parent node
    children: list[str] = field(default_factory=list)  # Full paths of child nodes
    times: list[float] = field(default_factory=list)  # Raw timing samples in ms
    count: int = 0  # Number of invocations

    # Timing statistics (calculated from times)
    avg_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    std_dev_ms: float = 0.0

    # Data access tracking (total amount of data read/written)
    input_bytes: list[int] = field(
        default_factory=list
    )  # Input data accessed per invocation
    output_bytes: list[int] = field(
        default_factory=list
    )  # Output data written per invocation
    avg_input_mb: float = 0.0  # Average input data in MB
    avg_output_mb: float = 0.0  # Average output data in MB
    avg_total_mb: float = 0.0  # Average total data accessed (input + output)

    # Tensor tracking (optional - populated when memory profiling is enabled)
    typical_input_tensors: list[dict] = field(default_factory=list)
    typical_output_tensors: list[dict] = field(default_factory=list)

    # Module information (populated for nn.Module operations)
    module_type: str | None = None
    module_name: str | None = None
