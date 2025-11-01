"""
Metal attention operation implementation.

This module handles compilation and execution of Metal attention kernels
for PyTorch MPS backend.
"""

from typing import Optional

import torch

from .mps_config import (
    DEBUG_ATOL,
    DEBUG_ENABLED,
    DEBUG_RTOL,
    DEBUG_VERBOSITY,
    MPS_COMPILE_AVAILABLE,
    VERBOSITY_DETAILED,
)
from .mps_shader_compiler import BaseShaderCompiler


class AttentionCompiler(BaseShaderCompiler):
    """Compiles and runs Metal attention kernels."""

    def __init__(self, page_size: int = 16):
        """
        Initialize AttentionCompiler with dynamic BLOCK_SIZE.

        Args:
            page_size: KV cache page size for BLOCK_SIZE compilation (default: 16)
        """
        super().__init__(page_size=page_size)
        self._compile_attention_kernels()

    def _compile_attention_kernels(self):
        """Compile the actual optimized Metal attention kernels."""
        if not MPS_COMPILE_AVAILABLE:
            return

        # Read the actual optimized kernel sources
        common_source = self._read_metal_file("metal_attention_common.metal")
        simdgroup_source = self._read_metal_file("metal_attention_simdgroup_opt.metal")

        if not common_source or not simdgroup_source:
            print("⚠️  Could not find optimized Metal kernel sources, using fallback")
            self._compile_simple_kernels()
            return

        # Process the common header to resolve includes
        processed_common = self._process_common_header(common_source)

        # Process the simdgroup source to resolve includes
        processed_simdgroup = self._resolve_includes(simdgroup_source, processed_common)

        # Transform Params struct to params_raw for torch.mps.compile_shader compatibility
        processed_simdgroup = processed_simdgroup.replace(
            "constant Params& params [[buffer(7)]]",
            "device const float* params_raw [[buffer(7)]]",
        )

        # Replace parameter access patterns
        param_replacements = [
            (
                "const int num_qo = params.num_qo;",
                "const int num_qo = (int)params_raw[0];",
            ),
            (
                "const int head_dim = params.head_dim;",
                "const int head_dim = (int)params_raw[1];",
            ),
            (
                "const int kv_head_dim = params.kv_head_dim;",
                "const int kv_head_dim = (int)params_raw[2];",
            ),
            (
                "const int head_size = params.head_size;",
                "const int head_size = (int)params_raw[3];",
            ),
            (
                "const int page_size = params.page_size;",
                "const int page_size = (int)params_raw[4];",
            ),
            (
                "const int num_query_heads = params.num_query_heads;",
                "const int num_query_heads = (int)params_raw[5];",
            ),
            (
                "const int num_kv_heads = params.num_kv_heads;",
                "const int num_kv_heads = (int)params_raw[6];",
            ),
            ("const float scale = params.scale;", "const float scale = params_raw[7];"),
        ]

        for old, new in param_replacements:
            processed_simdgroup = processed_simdgroup.replace(old, new)

        # Inject BLOCK_SIZE define based on page_size configuration
        # This allows dynamic configuration without modifying Metal source code
        block_size_define = f"#define BLOCK_SIZE {self.page_size}"

        # Compile the full optimized attention kernels with injected BLOCK_SIZE
        full_source = f"""
#include <metal_stdlib>
using namespace metal;

// Dynamically injected BLOCK_SIZE from configuration
{block_size_define}

{processed_common}

{processed_simdgroup}
"""

        try:
            # Compile the actual optimized shader library
            if self._compile_shader(full_source, "attention"):
                print(
                    f"✅ Compiled OPTIMIZED Metal attention kernels for MPS "
                    f"(BLOCK_SIZE={self.page_size})"
                )
                # Warm up: trigger PSO creation to catch threadgroup memory errors early
                self._warmup_kernel(
                    "attention", "batch_prefill_attention_unified_fp16_simdgroup_kernel"
                )
                self._warmup_kernel(
                    "attention", "batch_prefill_attention_unified_f32_simdgroup_kernel"
                )
            else:
                print("   Falling back to simple implementation")
                self._compile_simple_kernels()

        except (RuntimeError, OSError) as e:
            print(f"⚠️  Failed to compile optimized kernels: {e}")
            print("   Falling back to simple implementation")
            # Fall back to simpler implementation
            self._compile_simple_kernels()

    def _compile_simple_kernels(self):
        """Fallback removed - optimized kernels are required."""
        print("❌ Optimized kernel compilation failed - no fallback available")
        return

    def run_attention_mps(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        qo_indptr: torch.Tensor,
        custom_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run attention using compiled MPS kernels.

        All dtype conversions happen here before passing to Metal kernels.
        Metal kernels expect float16 (half) types.
        """

        if not self.can_use_mps_kernels():
            raise RuntimeError("MPS kernels not available")

        if "attention" not in self.compiled_libraries:
            raise RuntimeError("Attention kernel not compiled")

        # DEBUG MODE: Collect input metadata before any modifications
        input_metadata = []
        pytorch_reference = None
        if DEBUG_ENABLED:
            # pylint: disable=import-outside-toplevel
            from . import debug_utils
            from . import pytorch_reference

            input_metadata = [
                debug_utils.collect_tensor_metadata(query, "query"),
                debug_utils.collect_tensor_metadata(kv_cache, "kv_cache"),
                debug_utils.collect_tensor_metadata(kv_page_indices, "kv_page_indices"),
                debug_utils.collect_tensor_metadata(kv_page_indptr, "kv_page_indptr"),
                debug_utils.collect_tensor_metadata(
                    kv_last_page_lens, "kv_last_page_lens"
                ),
                debug_utils.collect_tensor_metadata(qo_indptr, "qo_indptr"),
            ]

        # Store original dtype
        original_dtype = query.dtype

        # Dtype validation: Metal kernels support float32 and float16 only
        with start_profile("attn_dtype_validation"):
            if query.dtype not in [torch.float32, torch.float16]:
                raise ValueError(
                    f"Unsupported dtype {query.dtype} for Metal attention kernel. "
                    f"Supported: float32, float16. Use dtype='float16' in config."
                )

            # Ensure tensors are on MPS device
            query = query.to("mps") if query.device.type != "mps" else query
            kv_cache = kv_cache.to("mps") if kv_cache.device.type != "mps" else kv_cache

        # Get dimensions
        with start_profile("attn_buffer_prep"):
            num_tokens, num_heads, head_dim = query.shape
            # Kernel expects 1D output buffer, we'll reshape it later
            output = torch.empty(
                num_tokens * num_heads * head_dim, device="mps", dtype=query.dtype
            )

        # Run the attention kernel
        with start_profile("attn_kernel_dispatch"):
            result = self._run_full_attention(
                query,
                kv_cache,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_lens,
                qo_indptr,
                output,
            )

        # Convert back to original dtype if needed
        if original_dtype != result.dtype:
            result = result.to(original_dtype)

        # DEBUG MODE: Run PyTorch reference and compare
        if DEBUG_ENABLED:
            from . import debug_utils  # pylint: disable=import-outside-toplevel

            assert pytorch_reference is not None, "pytorch_reference must be imported"

            # Run PyTorch reference (use original inputs before dtype conversion)
            # Need to reconstruct original query/kv_cache with original dtype
            query_for_ref = (
                query.to(original_dtype) if query.dtype != original_dtype else query
            )
            kv_cache_for_ref = (
                kv_cache.to(original_dtype)
                if kv_cache.dtype != original_dtype
                else kv_cache
            )

            pytorch_output = pytorch_reference.attention_reference(
                query_for_ref,
                kv_cache_for_ref,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_lens,
                qo_indptr,
                custom_mask,
            )

            # Compare Metal vs PyTorch outputs with dtype-aware tolerances
            # Half-precision requires relaxed tolerances due to different computation order
            if original_dtype == torch.float32:
                atol, rtol = DEBUG_ATOL, DEBUG_RTOL
            elif original_dtype == torch.float16:
                atol, rtol = 5e-4, 5e-3  # Relaxed for float16
            else:  # bfloat16
                atol, rtol = 2e-3, 2e-2  # Relaxed for bfloat16

            matches, diagnostics = debug_utils.compare_tensors(
                result, pytorch_output, atol=atol, rtol=rtol, operation_name="Attention"
            )

            # Generate and print report
            report = debug_utils.generate_report(
                diagnostics, input_metadata, verbosity=DEBUG_VERBOSITY
            )
            if report:
                print(report)

            # Warning if significant mismatch detected (> 1% of elements)
            # Small mismatch rates (< 1%) are expected for half-precision due to rounding
            if not matches and DEBUG_VERBOSITY >= VERBOSITY_DETAILED:
                mismatch_pct = diagnostics.get("mismatch_percentage", 0)
                if mismatch_pct > 1.0:
                    print(
                        f"WARNING: Attention Metal kernel output differs from "
                        f"PyTorch reference ({mismatch_pct:.2f}% mismatches)"
                    )

        return result

    # Metal constant memory limit (64KB)
    CONSTANT_MEMORY_LIMIT = 64 * 1024

    def _validate_constant_buffer_sizes(
        self,
        qo_indptr: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
    ) -> None:
        """Validate that index buffers fit in Metal constant memory (64KB).

        Metal constant address space has a 64KB limit. This validation ensures
        index buffers don't exceed this limit before kernel launch.

        Args:
            qo_indptr: Query offset indirection pointer
            kv_page_indptr: KV page indirection pointer
            kv_page_indices: KV page indices
            kv_last_page_lens: Last page lengths

        Raises:
            RuntimeError: If total buffer size exceeds constant memory limit
        """
        total_size = (
            qo_indptr.numel() * qo_indptr.element_size()
            + kv_page_indptr.numel() * kv_page_indptr.element_size()
            + kv_page_indices.numel() * kv_page_indices.element_size()
            + kv_last_page_lens.numel() * kv_last_page_lens.element_size()
        )

        if total_size > self.CONSTANT_MEMORY_LIMIT:
            raise RuntimeError(
                f"Index buffers ({total_size} bytes) exceed Metal constant "
                f"memory limit ({self.CONSTANT_MEMORY_LIMIT} bytes). "
                f"Consider reducing max_num_pages or batch_size."
            )

        # Log warning if approaching limit (>80% usage)
        usage_percent = (total_size / self.CONSTANT_MEMORY_LIMIT) * 100
        if usage_percent > 80:
            print(
                f"⚠️  Constant memory usage: {usage_percent:.1f}% ({total_size} bytes)"
            )

    def _run_full_attention(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        qo_indptr: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Run the actual optimized Metal attention kernel.

        Unified KV Cache Buffer Design:
        -------------------------------
        Buffer Layout: [num_pages, 2, page_size, num_kv_heads, head_dim]

        Within each page:
          - K cache: offset 0 to (page_size * num_kv_heads * head_dim - 1)
          - V cache: offset (page_size * num_kv_heads * head_dim) to end

        The Metal kernel uses calculate_k_offset() and calculate_v_offset() to
        compute absolute offsets from the unified buffer start for accessing
        K and V data respectively.
        """

        lib = self.compiled_libraries["attention"]

        # Validate that index buffers fit in Metal constant memory (64KB limit)
        self._validate_constant_buffer_sizes(
            qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_lens
        )

        with start_profile("attn_params_setup"):
            # Extract dimensions
            num_tokens, num_heads, head_dim = query.shape
            _num_pages, _, page_size, num_kv_heads, _ = kv_cache.shape
            scale = 1.0 / (head_dim**0.5)

            # Create the Params struct exactly as expected by the Metal kernel
            # This matches what the test uses: 8 parameters
            params_data = [
                num_tokens,  # num_qo (seq_len in test)
                num_heads * head_dim,  # head_dim (total_head_dim in test)
                num_kv_heads
                * head_dim,  # kv_head_dim (total KV head dim for GQA support)
                head_dim,  # head_size
                page_size,  # page_size
                num_heads,  # num_query_heads
                num_kv_heads,  # num_kv_heads (actual KV heads from cache)
                scale,  # scale (1/sqrt(head_dim))
            ]

            # Convert to Metal-compatible parameter buffer
            params = torch.tensor(params_data, dtype=torch.float32, device="mps")

        # Prepare tensors in the exact format expected by the kernel
        # - Pass unified KV cache as single buffer (no copying)
        # Layout: [num_pages, 2, page_size, num_kv_heads, head_dim]
        # kv_cache.contiguous().view(-1) creates a contiguous view
        #
        # Memory layout per page:
        #   - K cache: elements [0 ... page_size*kv_head_dim-1]
        #   - V cache: elements [page_size*kv_head_dim ... 2*page_size*kv_head_dim-1]

        # Single unified KV cache buffer
        # Ensure contiguous for optimal memory access
        paged_kv_cache = kv_cache.contiguous().view(-1)

        # Reshape query for kernel:
        # [num_tokens, num_heads, head_dim] -> 1D [num_tokens * num_heads * head_dim]
        q_input = query.contiguous().view(-1)

        # Create debug buffer (optional, can be None)
        debug_out = torch.zeros(20, dtype=torch.float32, device="mps")

        # Launch the actual optimized kernel!
        # Select kernel based on dtype: f32 for float32, fp16 for float16
        if query.dtype == torch.float32:
            kernel_name = "batch_prefill_attention_unified_f32_simdgroup_kernel"
        else:
            kernel_name = "batch_prefill_attention_unified_fp16_simdgroup_kernel"

        # Configure dispatch parameters
        threads_per_threadgroup = 128
        total_threads = num_tokens * threads_per_threadgroup

        if not hasattr(lib, kernel_name):
            raise RuntimeError(f"Kernel {kernel_name} not found in compiled library")

        # Call the optimized simdgroup kernel with exact parameter layout and dispatch config
        getattr(lib, kernel_name)(
            q_input,  # q_input [buffer(0)]
            paged_kv_cache,  # paged_kv_cache [buffer(1)] - unified KV cache
            qo_indptr.to(torch.int32),  # qo_indptr [buffer(2)]
            kv_page_indptr.to(torch.int32),  # kv_page_indptr [buffer(3)]
            kv_page_indices.to(torch.int32),  # kv_page_indices [buffer(4)]
            kv_last_page_lens.to(torch.int32),  # kv_last_page_lens [buffer(5)]
            output,  # output [buffer(6)]
            params,  # params [buffer(7)]
            debug_out,  # debug_out [buffer(8)]
            threads=(total_threads, 1, 1),
            group_size=(threads_per_threadgroup, 1, 1),
        )

        # Output is already in correct 1D layout written by kernel
        # Kernel writes with stride (num_heads * head_dim) per token
        # Reshape to 2D as per FlashInfer API: [num_tokens, num_heads * head_dim]
        total_head_dim = num_heads * head_dim
        return output.view(num_tokens, total_head_dim)
