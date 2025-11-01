"""Tensor analysis utilities for profiling."""

from __future__ import annotations

import torch


def _classify_by_name(name: str) -> str | None:
    """Classify tensor purpose based on its name. Returns None if no match."""
    name_lower = name.lower()

    # Define classification patterns (order matters: most specific first)
    patterns = [
        (
            ["cache", "key_cache", "value_cache", "kv_cache", "k_cache", "v_cache"],
            "kv_cache",
        ),
        (["weight", "kernel"], "weight"),
        (["bias"], "bias"),
        (["embedding", "embed", "token_embed", "pos_embed"], "embedding"),
        (["grad", "gradient"], "gradient"),
        (["activation", "output", "hidden", "logit"], "activation"),
    ]

    for keywords, purpose in patterns:
        if any(keyword in name_lower for keyword in keywords):
            return purpose

    return None


def _classify_by_heuristics(tensor: torch.Tensor, shape: tuple) -> str:
    """Classify tensor purpose using shape and property-based heuristics."""
    # KV cache: large 5D tensors with specific shape pattern
    if len(shape) == 5 and shape[0] > 4096:
        return "kv_cache"

    # Weight/gradient detection
    if tensor.requires_grad:
        return "weight" if tensor.numel() > 1000000 else "gradient"

    # Embedding: 2D with large vocabulary size
    if len(shape) == 2 and shape[0] > 10000:
        return "embedding"

    # Intermediate buffer
    if len(shape) >= 2:
        return "buffer"

    return "unknown"


def classify_tensor_purpose(
    tensor: torch.Tensor, shape: tuple, tensor_name: str | None = None
) -> str:
    """
    Classify what this tensor is used for.

    Classification priority:
    1. Check tensor name/attributes for semantic hints (if provided)
    2. Fall back to shape and property-based heuristics

    Args:
        tensor: The tensor to classify
        shape: The tensor's shape
        tensor_name: Optional name/identifier for the tensor (e.g., from variable name
                     or nn.Module parameter name). If provided, will be used for
                     semantic classification. Caller is responsible for extracting
                     this information to avoid expensive introspection.

    Returns:
        A string indicating the tensor's purpose: "kv_cache", "weight", "bias",
        "embedding", "gradient", "activation", "buffer", or "unknown"
    """
    # Try name-based classification first (most reliable when available)
    if tensor_name:
        purpose = _classify_by_name(tensor_name)
        if purpose:
            return purpose

    # Fall back to heuristics
    return _classify_by_heuristics(tensor, shape)


def is_persistent_tensor(
    tensor_id: int, size_mb: float, tensor_last_seen: dict[int, str]
) -> bool:
    """Determine if tensor persists across operations."""
    # If we've seen this tensor before, it's persistent
    if tensor_id in tensor_last_seen:
        return True

    # Large tensors (>100MB) are likely persistent (weights, cache)
    if size_mb > 100:
        return True

    return False


def is_reusable_tensor(tensor: torch.Tensor, purpose: str) -> bool:
    """Determine if tensor memory can be reused."""
    # Weights, biases, cache, and embeddings are not reusable (persistent model state)
    if purpose in ["weight", "bias", "kv_cache", "embedding"]:
        return False

    # Gradients are reusable (overwritten each backward pass)
    if purpose == "gradient":
        return True

    # Activations and buffers without grad are reusable (temporary intermediate results)
    if purpose in ["activation", "buffer"] and not tensor.requires_grad:
        return True

    return False
