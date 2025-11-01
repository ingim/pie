"""
DEPRECATED: Legacy profiler - use the unified profiler instead.

This module is kept for backward compatibility only. All functionality has been
migrated to the unified profiler (profiler/tracker.py) which provides:
- GPU synchronization for accurate timing (MPS/CUDA)
- Hierarchical timing tree
- Memory tracking and tensor I/O analysis
- Single unified JSON output

MIGRATION:
- Replace `from profiler import start_profile` with unified profiler (already done)
- The `profile_with_tensors()` function is now a no-op wrapper
- New code should use `profiler.start_profile()` which delegates to tracker.py

This file will be removed in a future release.
"""

from __future__ import annotations

import json
import time
from contextlib import ContextDecorator, contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .tracker import get_memory_tracker


class _TorchProfiler:
    """The internal profiler class. Users should interact with the global API.

    This is a singleton - only one instance should exist via the PROFILER global.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._node_map: dict[str, _TorchProfiler.Node] = {}
        self.root = self.Node(name="root", parent=None)
        self.active_node = self.root
        self.enabled = False  # Profiling disabled by default

    @dataclass
    class Node:
        """Represents a profiling node in the call tree."""

        name: str
        parent: _TorchProfiler.Node | None
        children: list[_TorchProfiler.Node] = field(default_factory=list)
        times: list[float] = field(
            default_factory=list
        )  # Raw timings for this node (self time)

        # Metrics calculated from timings
        count: int = 0
        mean: float = 0.0
        std: float = 0.0
        total_mean: float = 0.0  # Includes children's total_mean

    def _get_full_path(self, name: str) -> str:
        if self.active_node is self.root:
            return name
        return f"{self.active_node.name}.{name}"

    def start(self, name: str) -> _TorchProfiler.Timer:
        """Creates a new profiling scope context manager."""
        full_path = self._get_full_path(name)
        if full_path not in self._node_map:
            new_node = self.Node(name=full_path, parent=self.active_node)
            self.active_node.children.append(new_node)
            self._node_map[full_path] = new_node

        return self.Timer(self, self._node_map[full_path])

    class Timer(ContextDecorator):
        """Context manager for timing a specific profiling scope."""

        def __init__(self, profiler: _TorchProfiler, node: _TorchProfiler.Node):
            self.profiler = profiler
            self.node = node
            self.start_time: float = 0.0

        @staticmethod
        def _synchronize():
            """Synchronize the appropriate backend."""
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.synchronize()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.synchronize()

        def __enter__(self):
            self.profiler.active_node = self.node
            self._synchronize()
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *exc):
            _ = exc  # Exception info not currently used
            self._synchronize()
            elapsed_ms = (time.perf_counter() - self.start_time) * 1000
            self.node.times.append(elapsed_ms)

            if self.node.parent:
                self.profiler.active_node = self.node.parent
            return False

    def report(self):
        """Calculates statistics and prints the hierarchical report."""
        if not self.root.children:
            print("No profiling data collected.")
            return

        for node in self._node_map.values():
            if node.times:
                node.count = len(node.times)
                node.mean = float(np.mean(node.times))
                node.std = float(np.std(node.times))

        def calculate_total(node: _TorchProfiler.Node) -> float:
            children_total = sum(calculate_total(child) for child in node.children)
            node.total_mean = node.mean + children_total
            return node.total_mean

        calculate_total(self.root)

        print("\n--- 🌲 Performance Report 🌲 ---")
        headers = f"{'Operation':<50} {'Avg Latency (ms)':<20} {'% of Parent':<15}"
        headers += f" {'Std Dev (ms)':<20} {'Samples':<10}"
        print(headers)
        print("-" * 115)

        def print_node(
            node: _TorchProfiler.Node, indent: str = "", is_last: bool = True
        ):
            if not node.name.startswith("root"):
                name_part = node.name.split(".")[-1]
                line_char = "└── " if is_last else "├── "
                parent_total = (
                    node.parent.total_mean if node.parent else node.total_mean
                )
                percent_str = (
                    f"{(node.total_mean / parent_total * 100):.1f}%"
                    if parent_total > 1e-6
                    else ""
                )

                print(
                    f"{indent}{line_char}{name_part:<{46 - len(indent)}}"
                    f"{node.total_mean:<20.4f}"
                    f"{percent_str:<15}"
                    f"{node.std:<20.4f}"
                    f"{node.count:<10}"
                )
                child_indent = indent + ("    " if is_last else "│   ")
                for i, child in enumerate(node.children):
                    print_node(child, child_indent, i == len(node.children) - 1)

        for i, child in enumerate(self.root.children):
            print_node(child, is_last=i == len(self.root.children) - 1)

        print("-" * 115)

    def to_dict(self, include_samples: bool = False) -> dict:
        """Converts the profiling tree to a dictionary for JSON serialization.

        Args:
            include_samples: If True, includes individual sample times in the output
        """

        def calculate_total(node: _TorchProfiler.Node) -> float:
            if len(node.times) > 0:
                node.total_mean = float(np.mean(node.times))
            for child in node.children:
                calculate_total(child)
            return node.total_mean

        calculate_total(self.root)

        # Calculate the absolute root total (sum of all top-level operations)
        absolute_total = sum(child.total_mean for child in self.root.children)

        def node_to_dict(node: _TorchProfiler.Node, root_total: float) -> dict:
            """Recursively convert a node to a dictionary."""
            result = {
                "name": node.name.split(".")[-1] if "." in node.name else node.name,
                "full_path": node.name,
                "samples": len(node.times),
                "avg_latency_ms": float(np.mean(node.times)) if node.times else 0.0,
                "std_dev_ms": float(np.std(node.times)) if len(node.times) > 1 else 0.0,
                "min_ms": float(np.min(node.times)) if node.times else 0.0,
                "max_ms": float(np.max(node.times)) if node.times else 0.0,
                "total_mean_ms": node.total_mean,
            }

            if include_samples:
                result["times_ms"] = [float(t) for t in node.times]

            # Calculate percentage relative to parent
            if node.parent and node.parent.total_mean > 0:
                result["percent_of_parent"] = (
                    node.total_mean / node.parent.total_mean
                ) * 100

            # Calculate percentage relative to absolute root
            if root_total > 0:
                result["percent_of_total"] = (node.total_mean / root_total) * 100

            if node.children:
                result["children"] = [
                    node_to_dict(child, root_total) for child in node.children
                ]

            return result

        return {
            "timestamp": datetime.now().isoformat(),
            "profiling_tree": [
                node_to_dict(child, absolute_total) for child in self.root.children
            ],
        }

    def save_to_json(self, output_dir: str = ".", include_samples: bool = False) -> str:
        """
        Saves profiling results to a timestamped JSON file.

        Args:
            output_dir: Directory to save the JSON file (default: current directory)
            include_samples: If True, includes individual sample times in the output

        Returns:
            Path to the saved JSON file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_profiling_result.json"
        output_path = Path(output_dir) / filename

        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Convert to dict and save
        data = self.to_dict(include_samples=include_samples)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return str(output_path)

    def reset(self):
        """Clears all collected data for a fresh run."""
        self._node_map.clear()
        self.root = self.Node(name="root", parent=None)
        self.active_node = self.root


# --- GLOBAL PROFILER API ---
PROFILER = _TorchProfiler()


def set_profiling_enabled(enabled: bool) -> None:
    """
    DEPRECATED: Enable or disable profiling globally.

    This function is deprecated and now delegates to the unified profiler.
    Use `initialize_memory_tracker(enable_timing=True)` instead.

    Args:
        enabled: If True, profiling is enabled. If False, profiling is disabled.
    """
    PROFILER.enabled = enabled
    # Note: Unified profiler is initialized separately in server.py
    # This function is kept for backward compatibility but does nothing useful


def start_profile(name: str):
    """
    Starts a new profiling scope. Use as a context manager.

    This function now delegates to the unified profiler (tracker) when available,
    which provides both timing and memory tracking in a single hierarchical tree.
    Falls back to the old profiler if unified profiler is not initialized.

    Example:
        with start_profile("my_operation"):
            ...
    """
    # Try to use unified profiler first
    try:
        tracker = get_memory_tracker()
        # Use unified profiler if timing is enabled
        if tracker.enable_timing:
            return tracker.start_profile(name)
    except (ImportError, AttributeError):
        pass  # Fall back to old profiler

    # Fall back to old profiler
    if PROFILER.enabled:
        return PROFILER.start(name)
    return nullcontext()


@contextmanager
def profile_with_tensors(name: str, inputs: tuple = (), outputs_getter=None):
    """
    DEPRECATED: Convenience wrapper for profiling with automatic tensor tracking.

    This function is now a NO-OP wrapper that delegates to the unified profiler.
    The inputs and outputs parameters are ignored.

    MIGRATION: Replace with `from profiler import start_profile`:
        # Old (deprecated):
        with profile_with_tensors("operation", inputs=(x, y)) as profile:
            result = do_work(x, y)
            profile.set_outputs(result)

        # New (unified profiler):
        with start_profile("operation"):
            result = do_work(x, y)

    Args:
        name: Name of the operation to profile
        inputs: IGNORED - kept for backward compatibility
        outputs_getter: IGNORED - kept for backward compatibility
    """
    # Delegate to unified profiler via start_profile
    try:
        tracker = get_memory_tracker()
        if tracker.enable_timing:
            # Use unified profiler
            with tracker.start_profile(name):
                yield None
            return
    except (ImportError, AttributeError):
        pass

    # Fallback to legacy profiler if unified not available
    if not PROFILER.enabled:
        yield None
        return

    timer = PROFILER.start(name)
    _ = inputs  # Ignored
    _ = outputs_getter  # Ignored

    class OutputSetter:
        """Helper to allow setting outputs within the context (no-op for compatibility)."""

        def set_outputs(self, *outputs):
            """No-op - kept for backward compatibility."""
            _ = outputs

    output_setter = OutputSetter()

    with timer:
        yield output_setter
