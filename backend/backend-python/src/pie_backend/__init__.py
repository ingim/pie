"""Python backend implementation using FlashInfer."""

# pylint: disable=invalid-name  # Module name "backend-python" required by existing structure

from .handler import Handler
from .model_factory import create_model_and_fusion_map

__all__ = ["Handler", "create_model_and_fusion_map"]
