"""
FlashInfer-compatible API for metal_kernels.

This module provides a drop-in replacement for FlashInfer operations using
Metal acceleration on macOS with Apple Silicon. When Metal is not available,
it raises informative errors directing users to use PyTorch implementations.

Can be run in PyTorch-only mode via PIE_METAL_PYTORCH_MODE=1 for testing.
"""

import os
import platform
import sys
from typing import Any, Dict, Optional, Tuple

import torch

from .mps_shader_integration import get_mps_compiler
from .pytorch_reference import (
    append_paged_kv_cache_reference,
    attention_reference,
    rope_reference,
)

# Detect hardware capabilities
IS_MACOS = platform.system() == "Darwin"
IS_APPLE_SILICON = IS_MACOS and platform.processor() == "arm"

# Check if PyTorch-only mode is enabled
PYTORCH_MODE = os.environ.get("PIE_METAL_PYTORCH_MODE", "0") == "1"


def _validate_page_size(page_size: int) -> None:
    """
    Validate page_size configuration for Metal kernel compilation.

    Args:
        page_size: The KV cache page size to validate

    Raises:
        ValueError: If page_size is invalid
    """
    # Check if page_size is a power of 2
    if page_size <= 0 or (page_size & (page_size - 1)) != 0:
        raise ValueError(
            f"page_size must be a power of 2. Got: {page_size}. "
            f"Valid values: 8, 16, 32, 64, etc."
        )

    # Check dtype-specific memory constraints
    # Metal threadgroup memory limit: 32KB on Apple Silicon
    #
    # Accurate memory usage calculation for MAX_HEAD_DIM=256:
    # FP16 kernel threadgroup memory:
    #   - q_s[256]: 512 bytes
    #   - k_block[BLOCK_SIZE][256]: BLOCK_SIZE * 512 bytes
    #   - v_block[BLOCK_SIZE][256]: BLOCK_SIZE * 512 bytes
    #   - acc_i[256]: 1024 bytes
    #   - w_block[BLOCK_SIZE]: BLOCK_SIZE * 4 bytes
    #   - simd_scratch[4]: 16 bytes
    #   - m_i, l_i: 8 bytes
    # Total: 1560 + BLOCK_SIZE * 1028 bytes
    #
    # For power-of-2 page sizes:
    #   - page_size=16: 1560 + 16*1028 = 17,008 bytes ✓
    #   - page_size=32: 1560 + 32*1028 = 34,456 bytes ✗ (exceeds 32KB limit)
    #
    # F32 kernel uses MAX_F32_HEAD_DIM=128 and capped KERNEL_BLOCK_SIZE=16,
    # so it stays under 32KB automatically.
    #
    # Hard limit: page_size must be <= 16 (power of 2 constraint)
    if page_size > 16:
        estimated_memory = 1560 + page_size * 1028
        raise ValueError(
            f"page_size={page_size} exceeds Metal threadgroup memory limit (32KB). "
            f"Maximum supported: 16 (power of 2). "
            f"Your configuration would require ~{estimated_memory} bytes. "
            f"Valid values: 1, 2, 4, 8, 16"
        )


def _validate_mps_device(tensor: torch.Tensor, name: str) -> None:
    """Validate that a tensor is on MPS device.

    Args:
        tensor: Tensor to validate
        name: Name of the tensor for error messages

    Raises:
        RuntimeError: If tensor is not on MPS device
    """
    # Skip validation in PyTorch mode (allows CPU tensors)
    if PYTORCH_MODE:
        return

    if tensor.device.type != "mps":
        raise RuntimeError(
            f"metal_kernels requires all tensors to be on MPS device. "
            f"Tensor '{name}' is on {tensor.device}. "
            f"Please move all tensors to MPS before calling metal_kernels operations:\n"
            f"  tensor = tensor.to('mps')\n"
            f"Or set config.device='mps' when initializing the model."
        )


def _initialize_mps_backend(page_size: int = 16) -> bool:
    """
    Initialize MPS shader backend with dynamic BLOCK_SIZE configuration.

    Args:
        page_size: KV cache page size for BLOCK_SIZE compilation (default: 16)

    Returns:
        True if initialization succeeded, False otherwise
    """
    # Validate page_size before initialization
    try:
        _validate_page_size(page_size)
    except ValueError as e:
        print(f"❌ Invalid page_size configuration: {e}")
        sys.exit(1)

    # If PyTorch mode is enabled, skip Metal initialization
    if PYTORCH_MODE:
        print("⚠️  PIE_METAL_PYTORCH_MODE=1: Using PyTorch reference implementations")
        print("   Metal kernels disabled - operations will use pure PyTorch")
        print("   This mode is for testing/debugging only and will be slower")
        return True  # Return True to allow initialization to proceed

    if not IS_APPLE_SILICON:
        print("❌ metal_kernels requires macOS with Apple Silicon (M1/M2/M3)")
        print("   Install FlashInfer for other platforms:")
        print("   pip install flashinfer")
        return False

    # Validate MPS is available in PyTorch
    if not torch.backends.mps.is_available():
        print("❌ PyTorch MPS backend is not available")
        print("   Please ensure you have PyTorch 2.0+ with MPS support:")
        print("   pip install torch>=2.0.0")
        sys.exit(1)

    # Validate MPS device can be created
    try:
        test_tensor = torch.tensor([1.0], device="mps")
        del test_tensor
    except (RuntimeError, OSError) as e:
        print(f"❌ Cannot create tensors on MPS device: {e}")
        print("   Please check your PyTorch MPS installation")
        sys.exit(1)

    # Set default device to MPS for all PyTorch operations
    torch.set_default_device("mps")
    print("✅ Set default PyTorch device to 'mps'")
    print("   All tensors will be created on MPS device by default")

    try:
        # Initialize MPS shader compiler (singleton) with page_size configuration
        compiler = get_mps_compiler(page_size=page_size)
        if compiler.can_use_mps_kernels():
            print("✅ metal_kernels initialized with PyTorch MPS shader compilation")
            print(
                f"   Available shader libraries: {list(compiler.compiled_libraries.keys())}"
            )
            return True
        print("❌ MPS shader compilation not available")

    except (RuntimeError, ImportError, AttributeError) as e:
        print(f"❌ MPS shader initialization failed: {e}")

    # MPS initialization failed
    print("❌ MPS shader compilation failed")
    print("   Please check that:")
    print("   1. You have PyTorch 2.0+ with MPS support")
    print("   2. Xcode Command Line Tools are installed")
    print("   3. Your system supports Metal compute")
    sys.exit(1)


# Read page_size from environment variable or use default
_PAGE_SIZE_FROM_ENV = int(os.environ.get("PIE_METAL_PAGE_SIZE", "16"))

# Initialize on import with page_size from environment
_initialize_mps_backend(page_size=_PAGE_SIZE_FROM_ENV)


class BatchPrefillWithPagedKVCacheWrapper:
    """
    Drop-in replacement for flashinfer.BatchPrefillWithPagedKVCacheWrapper

    Handles prefill attention operations (processing multiple tokens per sequence).
    Plans operation parameters and executes attention using Metal kernels or PyTorch fallback.
    """

    def __init__(self, workspace_buffer: torch.Tensor, kv_layout: str = "NHD"):
        """
        Initialize prefill wrapper

        Args:
            workspace_buffer: Workspace memory (for FlashInfer compatibility, unused in Metal)
            kv_layout: KV cache layout, either "NHD" or "HND" (default: "NHD")
        """
        self.workspace_buffer = workspace_buffer
        self.kv_layout = kv_layout
        self._planned_params: Optional[Dict[str, Any]] = None
        self._is_planned = False
        self._pytorch_mode = PYTORCH_MODE

    def plan(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        page_size: int,
        pos_encoding_mode: str = "NONE",  # pylint: disable=unused-argument
        custom_mask: Optional[torch.Tensor] = None,
        q_data_type: torch.dtype = torch.float16,
    ) -> None:
        """
        Plan the prefill attention operation

        Stores all parameters needed for attention computation. Must be called before run().

        Args:
            qo_indptr: Query-output indptr tensor [batch_size + 1]
            paged_kv_indptr: KV page indptr tensor [batch_size + 1]
            paged_kv_indices: KV page indices tensor [num_pages]
            paged_kv_last_page_len: Last page lengths [batch_size]
            num_qo_heads: Number of query heads
            num_kv_heads: Number of key-value heads
            head_dim_qk: Head dimension for query/key
            page_size: KV cache page size
            pos_encoding_mode: Position encoding mode (unused, for compatibility)
            custom_mask: Optional attention mask
            q_data_type: Query tensor data type
        """
        # Validate all input tensors are on MPS device
        _validate_mps_device(qo_indptr, "qo_indptr")
        _validate_mps_device(paged_kv_indptr, "paged_kv_indptr")
        _validate_mps_device(paged_kv_indices, "paged_kv_indices")
        _validate_mps_device(paged_kv_last_page_len, "paged_kv_last_page_len")
        if custom_mask is not None:
            _validate_mps_device(custom_mask, "custom_mask")

        self._planned_params = {
            "qo_indptr": qo_indptr,
            "kv_page_indptr": paged_kv_indptr,
            "kv_page_indices": paged_kv_indices,
            "kv_last_page_lens": paged_kv_last_page_len,
            "num_query_heads": num_qo_heads,
            "num_kv_heads": num_kv_heads,
            "head_size": head_dim_qk,
            "page_size": page_size,
            "custom_mask": custom_mask,
            "q_data_type": q_data_type,
        }
        self._is_planned = True

    def run(self, query: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        """
        Execute prefill attention computation

        Args:
            query: Query tensor [num_tokens, num_query_heads, head_dim]
            kv_cache: Paged KV cache tensor [num_pages, 2, page_size, num_kv_heads, head_dim]

        Returns:
            Attention output tensor [num_tokens, num_query_heads * head_dim]
        """
        if not self._is_planned:
            raise RuntimeError("Must call plan() before run()")

        # Validate input tensors are on MPS device
        _validate_mps_device(query, "query")
        _validate_mps_device(kv_cache, "kv_cache")

        return self._run_metal(query, kv_cache)

    def _run_metal(self, query: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        """Execute using MPS shader kernels or PyTorch fallback"""
        assert self._planned_params is not None, "plan() must be called before run()"

        # If PyTorch mode is enabled, use PyTorch reference implementation
        if self._pytorch_mode:
            return attention_reference(
                query=query,
                kv_cache=kv_cache,
                kv_page_indices=self._planned_params["kv_page_indices"],
                kv_page_indptr=self._planned_params["kv_page_indptr"],
                kv_last_page_lens=self._planned_params["kv_last_page_lens"],
                qo_indptr=self._planned_params["qo_indptr"],
                custom_mask=self._planned_params["custom_mask"],
            )

        return get_mps_compiler().run_attention_mps(
            query=query,
            kv_cache=kv_cache,
            kv_page_indices=self._planned_params["kv_page_indices"],
            kv_page_indptr=self._planned_params["kv_page_indptr"],
            kv_last_page_lens=self._planned_params["kv_last_page_lens"],
            qo_indptr=self._planned_params["qo_indptr"],
            custom_mask=self._planned_params["custom_mask"],
        )


class BatchDecodeWithPagedKVCacheWrapper:
    """
    Drop-in replacement for flashinfer.BatchDecodeWithPagedKVCacheWrapper

    Handles decode attention operations (single token generation per sequence).
    """

    def __init__(self, workspace_buffer: torch.Tensor, kv_layout: str = "NHD"):
        """Initialize decode wrapper - similar to prefill wrapper"""
        self.workspace_buffer = workspace_buffer
        self.kv_layout = kv_layout
        self._planned_params: Optional[Dict[str, Any]] = None
        self._is_planned = False
        self._pytorch_mode = PYTORCH_MODE

    def plan(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        pos_encoding_mode: str = "NONE",  # pylint: disable=unused-argument
        q_data_type: torch.dtype = torch.float16,
    ) -> None:
        """Plan decode operation - simpler than prefill (single token per sequence)"""
        # Validate all input tensors are on MPS device
        _validate_mps_device(indptr, "indptr")
        _validate_mps_device(indices, "indices")
        _validate_mps_device(last_page_len, "last_page_len")

        self._planned_params = {
            "kv_page_indptr": indptr,
            "kv_page_indices": indices,
            "kv_last_page_lens": last_page_len,
            "num_query_heads": num_qo_heads,
            "num_kv_heads": num_kv_heads,
            "head_size": head_dim,
            "page_size": page_size,
            "q_data_type": q_data_type,
        }
        self._is_planned = True

    def run(self, query: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        """Execute decode attention - create qo_indptr for single tokens"""
        if not self._is_planned:
            raise RuntimeError("Must call plan() before run()")

        # Validate input tensors are on MPS device
        _validate_mps_device(query, "query")
        _validate_mps_device(kv_cache, "kv_cache")

        # For decode, we have one token per batch element
        assert self._planned_params is not None, "plan() must be called before run()"

        batch_size = self._planned_params["kv_page_indptr"].shape[0] - 1
        qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=query.device)

        # If PyTorch mode is enabled, use PyTorch reference implementation
        if self._pytorch_mode:
            return attention_reference(
                query=query,
                kv_cache=kv_cache,
                kv_page_indices=self._planned_params["kv_page_indices"],
                kv_page_indptr=self._planned_params["kv_page_indptr"],
                kv_last_page_lens=self._planned_params["kv_last_page_lens"],
                qo_indptr=qo_indptr,
                custom_mask=None,
            )

        return get_mps_compiler().run_attention_mps(
            query=query,
            kv_cache=kv_cache,
            kv_page_indices=self._planned_params["kv_page_indices"],
            kv_page_indptr=self._planned_params["kv_page_indptr"],
            kv_last_page_lens=self._planned_params["kv_last_page_lens"],
            qo_indptr=qo_indptr,
            custom_mask=None,
        )


def apply_rope_pos_ids_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rope_theta: float = 10000.0,
    interleave: bool = False,
) -> None:
    """
    Apply standard RoPE encoding in-place using Metal kernels.

    This is a drop-in replacement for FlashInfer's apply_rope_pos_ids_inplace.
    Simplified version without scaling factors (for Qwen3 and similar models).

    Parameters:
        q: Query ragged tensor, shape: (nnz, num_q_heads, head_dim) - modified in-place
        k: Key ragged tensor, shape: (nnz, num_k_heads, head_dim) - modified in-place
        pos_ids: Position indices, shape: (nnz)
        rope_theta: The theta value used in rope embedding (default: 10000.0)
        interleave: Whether to use interleaved layout (default: False)
    """
    # If PyTorch mode is enabled, use PyTorch reference implementation
    if PYTORCH_MODE:
        rope_reference(
            q,
            pos_ids,
            rope_theta=rope_theta,
            rope_factor=1.0,  # No scaling for standard RoPE
            interleaved=interleave,
            inplace=True,
        )
        rope_reference(
            k,
            pos_ids,
            rope_theta=rope_theta,
            rope_factor=1.0,  # No scaling for standard RoPE
            interleaved=interleave,
            inplace=True,
        )
        return

    # Validate all tensors are on MPS device
    _validate_mps_device(q, "q")
    _validate_mps_device(k, "k")
    _validate_mps_device(pos_ids, "pos_ids")

    # Apply RoPE to both query and key tensors using Metal kernels
    compiler = get_mps_compiler()
    compiler.run_rope_mps(
        q,
        pos_ids,
        rope_theta=rope_theta,
        rope_factor=1.0,  # No scaling for standard RoPE
        interleaved=interleave,
    )
    compiler.run_rope_mps(
        k,
        pos_ids,
        rope_theta=rope_theta,
        rope_factor=1.0,  # No scaling for standard RoPE
        interleaved=interleave,
    )


def apply_llama31_rope_pos_ids_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 32.0,
    rope_theta: float = 500000.0,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 4.0,
    old_context_len: int = 8192,
) -> None:
    """
    Apply LLaMA 3.1-style RoPE encoding in-place using Metal kernels.

    This is a drop-in replacement for FlashInfer's apply_llama31_rope_pos_ids_inplace.
    cos/sin values are computed on the fly inside the kernel.

    Parameters:
        q: Query ragged tensor, shape: (nnz, num_q_heads, head_dim) - modified in-place
        k: Key ragged tensor, shape: (nnz, num_k_heads, head_dim) - modified in-place
        pos_ids: Position indices, shape: (nnz)
        rotary_dim: The dimensions to apply RoPE. If None, apply to entire head dimension.
        interleave: Whether to use interleaved layout. If True, rotate even/odd dims separately.
                   If False, rotate first half and second half separately.
        rope_scale: The scaling factor used in rope embedding (default: 8.0)
        rope_theta: The theta value used in rope embedding (default: 500000.0)
        low_freq_factor: The low frequency factor used in Llama 3.1 RoPE (default: 1.0)
        high_freq_factor: The high frequency factor used in Llama 3.1 RoPE (default: 4.0)
        old_context_len: The old context length used in Llama 3.1 RoPE (default: 8192)

    Raises:
        ValueError: If unsupported parameters have non-default values
    """
    # Validate that unsupported parameters are using default values
    if rotary_dim is not None:
        raise ValueError(
            f"rotary_dim parameter is not supported in Metal RoPE implementation. "
            f"Got rotary_dim={rotary_dim}, expected None."
        )
    if low_freq_factor != 1.0:
        raise ValueError(
            f"low_freq_factor parameter is not supported in Metal RoPE implementation. "
            f"Got low_freq_factor={low_freq_factor}, expected 1.0."
        )
    if high_freq_factor != 4.0:
        raise ValueError(
            f"high_freq_factor parameter is not supported in Metal RoPE implementation. "
            f"Got high_freq_factor={high_freq_factor}, expected 4.0."
        )
    if old_context_len != 8192:
        raise ValueError(
            f"old_context_len parameter is not supported in Metal RoPE implementation. "
            f"Got old_context_len={old_context_len}, expected 8192."
        )

    # If PyTorch mode is enabled, use PyTorch reference implementation
    if PYTORCH_MODE:
        # Apply RoPE in-place with LLaMA 3.1-style wavelength-based scaling
        rope_reference(
            q,
            pos_ids,
            rope_theta=rope_theta,
            rope_factor=rope_scale,
            interleaved=interleave,
            inplace=True,
            low_freq_factor=low_freq_factor,
            high_freq_factor=high_freq_factor,
            old_context_len=old_context_len,
        )
        rope_reference(
            k,
            pos_ids,
            rope_theta=rope_theta,
            rope_factor=rope_scale,
            interleaved=interleave,
            inplace=True,
            low_freq_factor=low_freq_factor,
            high_freq_factor=high_freq_factor,
            old_context_len=old_context_len,
        )
        return

    # Validate all tensors are on MPS device
    _validate_mps_device(q, "q")
    _validate_mps_device(k, "k")
    _validate_mps_device(pos_ids, "pos_ids")

    # Apply RoPE to both query and key tensors using Metal kernels
    compiler = get_mps_compiler()
    compiler.run_rope_mps(
        q,
        pos_ids,
        rope_theta=rope_theta,
        rope_factor=rope_scale,
        interleaved=interleave,
    )
    compiler.run_rope_mps(
        k,
        pos_ids,
        rope_theta=rope_theta,
        rope_factor=rope_scale,
        interleaved=interleave,
    )


def append_paged_kv_cache(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    paged_kv_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    kv_layout: str = "NHD",  # pylint: disable=unused-argument
) -> None:
    """
    Append key-value states to paged KV cache using Metal kernels

    Args:
        append_key: Key states to append [num_tokens, num_kv_heads, head_dim]
        append_value: Value states to append [num_tokens, num_kv_heads, head_dim]
        batch_indices: Batch index for each token [num_tokens]
        positions: Position within sequence for each token [num_tokens]
        paged_kv_cache: Paged KV cache [num_pages, 2, page_size, num_kv_heads, head_dim]
        kv_indices: Page indices [num_pages]
        kv_indptr: Page indptr [batch_size + 1]
        kv_last_page_len: Last page lengths [batch_size]
        kv_layout: Layout string (unused, for compatibility)
    """
    # Validate all input tensors are on MPS device
    _validate_mps_device(append_key, "append_key")
    _validate_mps_device(append_value, "append_value")
    _validate_mps_device(batch_indices, "batch_indices")
    _validate_mps_device(positions, "positions")
    _validate_mps_device(paged_kv_cache, "paged_kv_cache")
    _validate_mps_device(kv_indices, "kv_indices")
    _validate_mps_device(kv_indptr, "kv_indptr")
    _validate_mps_device(kv_last_page_len, "kv_last_page_len")

    # If PyTorch mode is enabled, use PyTorch reference implementation
    if PYTORCH_MODE:
        # Extract dimensions
        num_tokens, num_kv_heads, head_dim = append_key.shape

        # Flatten key/value inputs:
        # [num_tokens, num_kv_heads, head_dim] -> [num_tokens, num_kv_heads * head_dim]
        append_key_flat = append_key.reshape(num_tokens, num_kv_heads * head_dim)
        append_value_flat = append_value.reshape(num_tokens, num_kv_heads * head_dim)

        # Split and flatten paged_kv_cache for reference function
        # Input shape: [num_pages, 2, page_size, num_kv_heads, head_dim]
        # Output shape: [num_pages, page_size, num_kv_heads * head_dim]
        paged_k = paged_kv_cache[:, 0, :, :, :].reshape(
            paged_kv_cache.shape[0], paged_kv_cache.shape[2], num_kv_heads * head_dim
        )
        paged_v = paged_kv_cache[:, 1, :, :, :].reshape(
            paged_kv_cache.shape[0], paged_kv_cache.shape[2], num_kv_heads * head_dim
        )

        append_paged_kv_cache_reference(
            append_key_flat,
            append_value_flat,
            paged_k,
            paged_v,
            batch_indices,
            positions,
            kv_indices,
            kv_indptr,
            kv_last_page_len,
            _num_kv_heads=num_kv_heads,
            _head_size=head_dim,
        )
        return

    try:
        # Try to use Metal append_kv_cache kernel via MPSShaderCompiler
        compiler = get_mps_compiler()
        if (
            compiler.can_use_mps_kernels()
            and "append_kv_cache" in compiler.compiled_libraries
        ):
            # Extract dimensions
            num_tokens, num_kv_heads, head_dim = append_key.shape

            # Reshape inputs for Metal kernel
            # Metal expects [num_tokens, num_kv_heads * head_dim] for key/value
            k_flat = append_key.contiguous().reshape(
                num_tokens, num_kv_heads * head_dim
            )
            v_flat = append_value.contiguous().reshape(
                num_tokens, num_kv_heads * head_dim
            )

            # OPTIMIZATION: Pass unified KV cache buffer (same as attention kernel)
            # Layout: [num_pages, 2, page_size, num_kv_heads, head_dim]
            # The cache is already contiguous, flatten and pass as unified buffer
            # Kernel will handle interleaved K/V with proper offset calculations

            # Extract shape info BEFORE flattening (needed by compiler)
            _num_pages, _, page_size, _, _ = paged_kv_cache.shape

            # Flatten entire unified cache (no copy - just a view!)
            paged_kv_unified = paged_kv_cache.view(-1)

            compiler.run_append_paged_kv_cache_mps(
                k_flat,
                v_flat,
                paged_kv_unified,
                batch_indices,
                positions,
                kv_indices,
                kv_indptr,
                kv_last_page_len,
                num_kv_heads=num_kv_heads,
                head_size=head_dim,
                page_size=page_size,
            )

            # No copy needed! Kernel wrote directly to unified cache with proper offsets
            return
    except Exception as e:
        # Re-raise to expose the issue - NO FALLBACK for Metal operations
        raise RuntimeError(f"Metal append_paged_kv_cache failed: {e}") from e

    # NOTE: PyTorch fallback removed - we must use Metal kernels
    # If you reach here, Metal kernel was not available
    raise RuntimeError(
        "Metal append_paged_kv_cache kernel not available. "
        "metal_kernels requires Metal support on Apple Silicon."
    )


def get_seq_lens(
    kv_page_indptr: torch.Tensor, kv_last_page_lens: torch.Tensor, page_size: int
) -> torch.Tensor:
    """Calculate sequence lengths from paging metadata"""
    # Validate input tensors are on MPS device
    _validate_mps_device(kv_page_indptr, "kv_page_indptr")
    _validate_mps_device(kv_last_page_lens, "kv_last_page_lens")

    batch_size = kv_page_indptr.shape[0] - 1
    seq_lens = torch.zeros(batch_size, dtype=torch.int32, device=kv_page_indptr.device)

    for i in range(batch_size):
        num_pages = kv_page_indptr[i + 1] - kv_page_indptr[i]
        if num_pages > 0:
            seq_lens[i] = (num_pages - 1) * page_size + kv_last_page_lens[i]

    return seq_lens


def get_batch_indices_positions(
    append_indptr: torch.Tensor, seq_lens: torch.Tensor, nnz: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get batch indices and positions for tokens"""
    # Validate input tensors are on MPS device
    _validate_mps_device(append_indptr, "append_indptr")
    _validate_mps_device(seq_lens, "seq_lens")

    device = append_indptr.device

    # Compute batch indices (self-contained implementation)
    batch_indices = torch.empty(nnz, dtype=torch.int32, device=device)
    for batch_idx in range(append_indptr.numel() - 1):
        start_idx = int(append_indptr[batch_idx].item())
        end_idx = int(append_indptr[batch_idx + 1].item())
        batch_indices[start_idx:end_idx] = batch_idx

    # Compute positions
    # NOTE: seq_lens represents the KV cache length AFTER appending the new tokens.
    # The controller updates kv_last_page_lens to include space for new tokens.
    # Therefore, positions should be: [seq_len - num_new, seq_len)
    # This gives the range where new tokens will be written.
    positions = torch.empty(nnz, dtype=torch.int32, device=device)
    for batch_idx in range(append_indptr.numel() - 1):
        start_idx = int(append_indptr[batch_idx].item())
        end_idx = int(append_indptr[batch_idx + 1].item())
        num_new = end_idx - start_idx
        seq_len = int(seq_lens[batch_idx].item())
        pos_start = seq_len - num_new
        pos_end = seq_len

        positions[start_idx:end_idx] = torch.arange(
            pos_start, pos_end, dtype=torch.int32, device=device
        )

    return batch_indices, positions


# Optional: Basic sampling operations for completeness
class image:  # pylint: disable=invalid-name
    """Image operations namespace (FlashInfer API compatibility)"""

    @staticmethod
    def decode_image(
        image_blob: bytes, dtype: torch.dtype, device: str
    ) -> torch.Tensor:
        """Decode image from bytes to tensor - not yet implemented for Metal."""
        raise NotImplementedError(
            "Image decoding not yet implemented in metal_kernels. "
            "This feature will be added in a future release."
        )


class sampling:  # pylint: disable=invalid-name
    """Sampling operations namespace (basic PyTorch implementations)"""

    @staticmethod
    def sampling_from_probs(probs: torch.Tensor) -> torch.Tensor:
        """Sample tokens from probability distribution"""
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @staticmethod
    def top_p_sampling_from_probs(
        probs: torch.Tensor, top_p: torch.Tensor
    ) -> torch.Tensor:
        """Top-p (nucleus) sampling from probability distribution"""
        # Simple implementation - can be enhanced
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Create mask for tokens to keep
        mask = cumulative_probs <= top_p.unsqueeze(-1)
        mask[..., 0] = True  # Always keep at least one token

        # Zero out tokens beyond threshold
        sorted_probs[~mask] = 0

        # Sample from filtered distribution
        sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)

        # Map back to original token indices
        return torch.gather(
            sorted_indices, -1, sampled_sorted_idx.unsqueeze(-1)
        ).squeeze(-1)

    @staticmethod
    def top_k_sampling_from_probs(
        probs: torch.Tensor, top_k: torch.Tensor
    ) -> torch.Tensor:
        """Top-k sampling - not yet implemented for Metal."""
        raise NotImplementedError(
            "top_k_sampling not yet implemented in metal_kernels. "
            "This feature will be added in a future release."
        )

    @staticmethod
    def min_p_sampling_from_probs(
        probs: torch.Tensor, min_p: torch.Tensor
    ) -> torch.Tensor:
        """Min-p sampling - not yet implemented for Metal."""
        raise NotImplementedError(
            "min_p_sampling not yet implemented in metal_kernels. "
            "This feature will be added in a future release."
        )

    @staticmethod
    def top_k_top_p_sampling_from_probs(
        probs: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor
    ) -> torch.Tensor:
        """Combined top-k and top-p sampling - not yet implemented for Metal."""
        raise NotImplementedError(
            "top_k_top_p_sampling not yet implemented in metal_kernels. "
            "This feature will be added in a future release."
        )


# Export all public functions and classes
__all__ = [
    "BatchPrefillWithPagedKVCacheWrapper",
    "BatchDecodeWithPagedKVCacheWrapper",
    "apply_rope_pos_ids_inplace",
    "apply_llama31_rope_pos_ids_inplace",
    "append_paged_kv_cache",
    "get_seq_lens",
    "get_batch_indices_positions",
    "image",
    "sampling",
]
