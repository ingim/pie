"""Export utilities for profiler data serialization."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .types import OperationNode, TensorInfo, TreeNode


def export_timing_tree(
    timing_tree: dict[str, TreeNode],
) -> list[dict[str, Any]]:
    """
    Export timing tree as a hierarchical list (unified profiler format).

    Returns a list of root nodes, each containing nested children.
    """
    # Find root nodes (nodes with no parent)
    root_nodes = [node for node in timing_tree.values() if node.parent is None]

    def build_node_dict(node: TreeNode) -> dict[str, Any]:
        """Recursively build a dictionary representation of a tree node."""
        # Calculate self-time (exclusive time = total time - children time)
        children_total_ms = 0.0
        child_nodes = []

        if node.children:
            for child_path in node.children:
                if child_path in timing_tree:
                    child_node = timing_tree[child_path]
                    children_total_ms += child_node.avg_ms
                    child_nodes.append(build_node_dict(child_node))

        # Self time is the time spent in this node excluding children
        self_time_ms = max(0.0, node.avg_ms - children_total_ms)

        node_dict = {
            "name": node.name,
            "full_path": node.full_path,
            "avg_latency_ms": round(
                node.avg_ms, 3
            ),  # Inclusive time (includes children)
            "self_time_ms": round(self_time_ms, 3),  # Exclusive time (just this node)
            "min_ms": round(node.min_ms, 3),
            "max_ms": round(node.max_ms, 3),
            "std_dev_ms": round(node.std_dev_ms, 3),
            "samples": node.count,
        }

        # Add data access metrics if available
        if node.avg_total_mb > 0:
            node_dict["data_access"] = {
                "input_mb": round(node.avg_input_mb, 3),
                "output_mb": round(node.avg_output_mb, 3),
                "total_mb": round(node.avg_total_mb, 3),
            }

        # Add module information if available
        if node.module_type:
            node_dict["module_type"] = node.module_type
        if node.module_name:
            node_dict["module_name"] = node.module_name

        # Add tensor information if available
        if node.typical_input_tensors:
            node_dict["typical_input_tensors"] = node.typical_input_tensors
        if node.typical_output_tensors:
            node_dict["typical_output_tensors"] = node.typical_output_tensors

        # Add children if any, with (self) entry inserted first
        # if there's self-time AND other children
        if child_nodes:
            children_with_self = []

            # Only add (self) entry if there are other children AND meaningful self-time
            # This avoids redundant (self) entries for leaf nodes
            if child_nodes and self_time_ms > 0.001:  # More than 0.001ms
                children_with_self.append(
                    {
                        "name": "(self)",
                        "full_path": f"{node.full_path}.(self)",
                        "avg_latency_ms": round(self_time_ms, 3),
                        "self_time_ms": round(self_time_ms, 3),
                        "min_ms": round(self_time_ms, 3),
                        "max_ms": round(self_time_ms, 3),
                        "std_dev_ms": 0.0,
                        "samples": node.count,
                        "is_self_time": True,  # Mark this as a synthetic self-time node
                    }
                )

            # Add actual children
            children_with_self.extend(child_nodes)

            if children_with_self:
                node_dict["children"] = children_with_self

        return node_dict

    # Build tree starting from root nodes
    return [build_node_dict(root) for root in root_nodes]


def export_operation_graph(
    operation_graph: dict[str, OperationNode],
    tensor_registry: dict[int, TensorInfo],
) -> dict[str, Any]:
    """Export operation graph as a dictionary."""
    graph = {}
    for op_name, node in operation_graph.items():
        graph[op_name] = {
            "invocations": node.invocations,
            "total_memory_delta_mb": round(node.total_memory_delta_mb, 2),
            "tensors_allocated": len(set(node.tensors_allocated)),
            "tensors_read": len(set(node.tensors_read)),
            "tensors_written": len(set(node.tensors_written)),
            "child_operations": node.child_operations,
            "tensor_details": {
                "allocated": [
                    tensor_registry[tid].purpose
                    for tid in set(node.tensors_allocated)
                    if tid in tensor_registry
                ],
                "read": [
                    tensor_registry[tid].purpose
                    for tid in set(node.tensors_read)
                    if tid in tensor_registry
                ],
            },
        }
    return graph


def export_tensor_lifecycle(
    tensor_lifecycle: dict,
    tensor_registry: dict[int, TensorInfo],
) -> dict[str, Any]:
    """Export tensor lifecycle events."""
    lifecycle = {}
    for tensor_id, events in tensor_lifecycle.items():
        if tensor_id in tensor_registry:
            tensor_info = tensor_registry[tensor_id]
            lifecycle[str(tensor_id)] = {
                "purpose": tensor_info.purpose,
                "size_mb": tensor_info.size_mb,
                "shape": list(tensor_info.shape),
                "is_persistent": tensor_info.is_persistent,
                "is_reusable": tensor_info.is_reusable,
                "events": [asdict(event) for event in events],
            }
    return lifecycle


def generate_operation_summary(
    operation_allocations: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Generate a summary of operation-level memory allocations."""
    summary = {}
    for op_name, allocations in operation_allocations.items():
        total_delta = sum(a["memory_delta_mb"] for a in allocations)
        avg_delta = total_delta / len(allocations) if allocations else 0
        max_delta = max((a["memory_delta_mb"] for a in allocations), default=0)

        summary[op_name] = {
            "invocation_count": len(allocations),
            "total_memory_delta_mb": round(total_delta, 2),
            "avg_memory_delta_mb": round(avg_delta, 2),
            "max_memory_delta_mb": round(max_delta, 2),
        }
    return summary
