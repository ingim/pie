"""Factory helpers for constructing backend models and fusion maps.

Registry-based model creation with platform-specific support.
"""

# pylint: disable=import-outside-toplevel  # Intentional for lazy loading

from __future__ import annotations

from typing import Callable, NamedTuple

from model.config import ModelInfo
from platform_detection import is_apple_silicon


class ArchitectureSpec(NamedTuple):
    """Specification for creating a model architecture."""

    create_model: Callable[[ModelInfo], tuple]
    description: str


def _create_l4ma_model(model_info: ModelInfo):
    """Create L4MA model and fusion map."""
    from model.l4ma import L4maArch, L4maForCausalLM, create_fusion_map

    arch = L4maArch(**model_info.architecture.__dict__)
    model = L4maForCausalLM(arch)
    fusion_map = create_fusion_map(model)
    return model, fusion_map


def _create_qwen3_model(model_info: ModelInfo):
    """Create Qwen3 model and fusion map (supports both CUDA and Apple Silicon)."""
    from model.qwen3 import Qwen3Arch, Qwen3ForCausalLM, create_fusion_map

    arch = Qwen3Arch(**model_info.architecture.__dict__)
    model = Qwen3ForCausalLM(arch)
    fusion_map = create_fusion_map(model)
    return model, fusion_map


# Build registry based on platform
ARCHITECTURE_REGISTRY: dict[str, ArchitectureSpec | None] = {
    # L4MA is always available (works with both flashinfer and metal_kernels)
    "l4ma": ArchitectureSpec(
        create_model=_create_l4ma_model,
        description="Llama-like architecture with FlashInfer/metal_kernels backend",
    ),
    # Qwen3 is now available on both platforms (uses conditional imports)
    "qwen3": ArchitectureSpec(
        create_model=_create_qwen3_model,
        description="Qwen3 architecture with FlashInfer/metal_kernels backend",
    ),
}

# Add FlashInfer-only architectures if not on Apple Silicon
IS_APPLE_SILICON = is_apple_silicon()

if not IS_APPLE_SILICON:

    def _create_qwen2_model(model_info: ModelInfo):
        from model.qwen2 import Qwen2Arch, Qwen2ForCausalLM, create_fusion_map

        arch = Qwen2Arch(**model_info.architecture.__dict__)
        model = Qwen2ForCausalLM(arch)
        fusion_map = create_fusion_map(model)
        return model, fusion_map

    def _create_gptoss_model(model_info: ModelInfo):
        from model.gptoss import GptOssArch, GptOssForCausalLM, create_fusion_map

        arch = GptOssArch(**model_info.architecture.__dict__)
        model = GptOssForCausalLM(arch)
        fusion_map = create_fusion_map(model)
        return model, fusion_map

    ARCHITECTURE_REGISTRY.update(
        {
            "qwen2": ArchitectureSpec(
                create_model=_create_qwen2_model,
                description="Qwen2 architecture (requires FlashInfer)",
            ),
            "gptoss": ArchitectureSpec(
                create_model=_create_gptoss_model,
                description="GPT OSS architecture (requires FlashInfer)",
            ),
        }
    )


def create_model_and_fusion_map(model_info: ModelInfo):
    """Create a model and fusion map based on architecture type.

    Args:
        model_info: Model information containing architecture details

    Returns:
        Tuple of (model, fusion_map)

    Raises:
        RuntimeError: If architecture is not supported on this platform
    """
    arch_type = model_info.architecture.type.lower()

    spec = ARCHITECTURE_REGISTRY.get(arch_type)
    if spec is None:
        supported = list(ARCHITECTURE_REGISTRY.keys())
        raise RuntimeError(
            f"Architecture '{arch_type}' is not supported on this platform. "
            f"Supported architectures: {supported}"
        )

    return spec.create_model(model_info)
