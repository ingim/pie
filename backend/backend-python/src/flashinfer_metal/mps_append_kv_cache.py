"""
Metal append_paged_kv_cache operation implementation.

This module handles compilation and execution of Metal append KV cache kernels
for PyTorch MPS backend.
"""

import torch

from .mps_config import MPS_COMPILE_AVAILABLE
from .mps_shader_compiler import BaseShaderCompiler


class AppendKVCacheCompiler(BaseShaderCompiler):
    """Compiles and runs Metal append_paged_kv_cache kernels."""

    def __init__(self):
        super().__init__()
        self._compile_append_kv_cache_kernels()

    def _compile_append_kv_cache_kernels(self):
        """Compile append_paged_kv_cache Metal kernels."""
        if not MPS_COMPILE_AVAILABLE:
            return

        # Read append_kv_cache kernel source
        append_source = self._read_metal_file("metal_append_paged_kv_cache.metal")
        if not append_source:
            print("⚠️  Append KV cache kernel source not found")
            return

        try:
            # Compile the append_kv_cache shader library
            if self._compile_shader(append_source, "append_kv_cache"):
                print("✅ Compiled append_paged_kv_cache kernels for MPS")
        except (RuntimeError, OSError) as e:
            print(f"⚠️  Failed to compile append_kv_cache kernels: {e}")

    def run_append_paged_kv_cache_mps(
        self,
        k_input: torch.Tensor,
        v_input: torch.Tensor,
        paged_kv_cache: torch.Tensor,
        kv_batch_indices: torch.Tensor,
        kv_positions: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
        page_size: int,
    ) -> None:
        """
        Run append_paged_kv_cache using compiled MPS kernels with unified buffer.

        Modifies paged_kv_cache in-place.

        Args:
            k_input: Key states to append [num_tokens, num_kv_heads * head_size]
            v_input: Value states to append [num_tokens, num_kv_heads * head_size]
            paged_kv_cache: Unified KV cache buffer (flattened 1D tensor)
            kv_batch_indices: Batch index for each token [num_tokens]
            kv_positions: Position within sequence [num_tokens]
            kv_page_indices: Page indices [max_num_pages]
            kv_page_indptr: Page indptr [batch_size + 1]
            kv_last_page_lens: Last page lengths [batch_size]
            num_kv_heads: Number of KV heads (required)
            head_size: Head dimension (required)
            page_size: Page size (required - cannot be inferred from flattened buffer)
        """
        if (
            not self.can_use_mps_kernels()
            or "append_kv_cache" not in self.compiled_libraries
        ):
            raise RuntimeError("Append KV cache MPS kernels not available")

        # Ensure all tensors are on MPS device first
        k_input = k_input.to("mps") if k_input.device.type != "mps" else k_input
        v_input = v_input.to("mps") if v_input.device.type != "mps" else v_input
        paged_kv_cache = (
            paged_kv_cache.to("mps")
            if paged_kv_cache.device.type != "mps"
            else paged_kv_cache
        )

        # Validate required parameters
        if num_kv_heads is None or head_size is None or page_size is None:
            raise ValueError(
                "num_kv_heads, head_size, and page_size must be provided "
                "to run_append_paged_kv_cache_mps"
            )

        # Get dimensions
        num_tokens = k_input.shape[0]
        batch_size = kv_page_indptr.shape[0] - 1

        # Calculate max_num_pages from unified buffer
        # Unified buffer layout: [num_pages * 2 * page_size * num_kv_heads * head_size]
        max_num_pages = paged_kv_cache.numel() // (
            2 * page_size * num_kv_heads * head_size
        )

        # Create params tensor matching AppendPagedKVCacheParams struct
        params = torch.tensor(
            [
                num_tokens,
                num_kv_heads,
                head_size,
                page_size,
                max_num_pages,
                batch_size,
                num_kv_heads * head_size,  # k_stride_token
                head_size,  # k_stride_head
                num_kv_heads * head_size,  # v_stride_token
                head_size,  # v_stride_head
            ],
            dtype=torch.int32,
            device="mps",
        )

        lib = self.compiled_libraries["append_kv_cache"]

        # Select kernel based on dtype
        if k_input.dtype == torch.bfloat16:
            kernel_name = "metal_append_paged_kv_cache_bfloat16"
        elif k_input.dtype == torch.float32:
            kernel_name = "metal_append_paged_kv_cache_float32"
        elif k_input.dtype == torch.float16:
            kernel_name = "metal_append_paged_kv_cache_float16"
        else:
            raise RuntimeError(
                f"append_paged_kv_cache only supports float16, bfloat16, and "
                f"float32, got {k_input.dtype}. "
                f"Please convert tensors before calling "
                f"(e.g., tensor.to(torch.float32))."
            )

        if hasattr(lib, kernel_name):
            # append_kv_cache uses 3D grid: (num_tokens, num_kv_heads, head_size)
            # Each thread handles one element at position (token_idx, head_idx, head_offset)
            # The Metal kernel expects thread_position_in_grid with x=token, y=head, z=offset

            # Call Metal append_kv_cache kernel (modifies unified cache in-place)
            getattr(lib, kernel_name)(
                k_input,  # buffer(0)
                v_input,  # buffer(1)
                paged_kv_cache,  # buffer(2) - unified KV cache
                kv_batch_indices.to(torch.int32),  # buffer(3)
                kv_positions.to(torch.int32),  # buffer(4)
                kv_page_indices.to(torch.int32),  # buffer(5)
                kv_page_indptr.to(torch.int32),  # buffer(6)
                kv_last_page_lens.to(torch.int32),  # buffer(7)
                params.float(),  # buffer(8) - Metal expects float buffer
                threads=(num_tokens, num_kv_heads, head_size),
                group_size=(8, 8, 8),  # Use 3D threadgroup to match 3D dispatch
            )
        else:
            raise RuntimeError(
                f"Append KV cache kernel {kernel_name} not found in compiled library"
            )
