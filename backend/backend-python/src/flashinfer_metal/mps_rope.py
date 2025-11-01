"""
Metal RoPE (Rotary Position Embedding) operation implementation.

This module handles compilation and execution of Metal RoPE kernels
for PyTorch MPS backend.
"""

import torch
from .mps_shader_compiler import BaseShaderCompiler
from .mps_config import (
    MPS_COMPILE_AVAILABLE,
    DEBUG_ENABLED,
    DEBUG_ATOL,
    DEBUG_RTOL,
    DEBUG_VERBOSITY,
    VERBOSITY_DETAILED,
)


class RoPECompiler(BaseShaderCompiler):
    """Compiles and runs Metal RoPE kernels."""

    def __init__(self):
        super().__init__()
        self._compile_rope_kernels()

    def _compile_rope_kernels(self):
        """Compile RoPE (Rotary Position Embedding) Metal kernels."""
        if not MPS_COMPILE_AVAILABLE:
            return

        # Read RoPE kernel source
        rope_source = self._read_metal_file("metal_rope.metal")
        if not rope_source:
            print("⚠️  RoPE kernel source not found")
            return

        try:
            # Compile the RoPE shader library
            if self._compile_shader(rope_source, "rope"):
                print("✅ Compiled RoPE kernels for MPS")
        except (RuntimeError, OSError) as e:
            print(f"⚠️  Failed to compile RoPE kernels: {e}")

    def run_rope_mps(
        self,
        input_qk: torch.Tensor,
        position_ids: torch.Tensor,
        rope_theta: float = 10000.0,
        rope_factor: float = 1.0,
        interleaved: bool = False,
    ) -> None:
        """
        Run RoPE (Rotary Position Embedding) using compiled MPS kernels.

        Modifies input_qk in-place.

        IMPORTANT: Input tensors must be on MPS device for the kernel to work.

        Args:
            input_qk: Input tensor [num_tokens, num_heads, head_size] on MPS device
            position_ids: Position IDs [num_tokens] on MPS device
            rope_theta: RoPE theta parameter (default: 10000.0)
            rope_factor: RoPE scaling factor (default: 1.0)
            interleaved: Layout mode (default: False for non-interleaved)
        """
        if not self.can_use_mps_kernels() or "rope" not in self.compiled_libraries:
            raise RuntimeError("RoPE MPS kernels not available")

        # DEBUG MODE: Run PyTorch reference and compare
        input_metadata = []
        original_data = None
        pytorch_output = None
        if DEBUG_ENABLED:
            # pylint: disable=import-outside-toplevel
            from . import debug_utils
            from . import pytorch_reference

            # Clone input for comparison (since Metal kernel modifies in-place)
            input_clone_for_pytorch = input_qk.clone()
            input_clone_for_metal = input_qk.clone()

            # Collect input metadata
            input_metadata = [
                debug_utils.collect_tensor_metadata(input_qk, "input_qk"),
                debug_utils.collect_tensor_metadata(position_ids, "position_ids"),
            ]

            # Run PyTorch reference on clone
            pytorch_output = pytorch_reference.rope_reference(
                input_clone_for_pytorch,
                position_ids,
                rope_theta,
                rope_factor,
                interleaved,
            )

            # Temporarily modify input_qk to point to our metal clone
            original_data = input_qk.data
            input_qk.data = input_clone_for_metal.data

        # Verify tensors are on MPS device
        if input_qk.device.type != "mps":
            raise RuntimeError(f"input_qk must be on MPS device, got {input_qk.device}")
        if position_ids.device.type != "mps":
            position_ids = position_ids.to("mps")

        # Get dimensions
        num_tokens, num_heads, head_size = input_qk.shape

        # DEBUG: Log input parameters
        if DEBUG_ENABLED and DEBUG_VERBOSITY >= VERBOSITY_DETAILED:
            print("\n🔍 [RoPE MPS] Input parameters:")
            print(f"   input_qk.shape: {input_qk.shape}")
            print(f"   input_qk.dtype: {input_qk.dtype}")
            print(f"   input_qk.device: {input_qk.device}")
            print(f"   position_ids.shape: {position_ids.shape}")
            first_5 = (
                position_ids[:5].cpu().tolist()
                if len(position_ids) >= 5
                else position_ids.cpu().tolist()
            )
            print(f"   position_ids first 5: {first_5}")
            print(f"   rope_theta: {rope_theta}")
            print(f"   rope_factor: {rope_factor}")
            print(f"   interleaved: {interleaved}")

        # KEY INSIGHT: Flatten 3D tensor to 1D for Metal kernel
        # The Metal kernel uses 3D grid dispatch internally, but expects 1D buffer
        # IMPORTANT: If tensor is not contiguous, .contiguous() creates a COPY
        # We need to either:
        # 1. Work with contiguous tensor and copy back, OR
        # 2. Make input contiguous in-place

        # Check if input is contiguous
        was_contiguous = input_qk.is_contiguous()
        input_qk_contiguous = None

        if was_contiguous:
            # Can directly flatten as a view (shares memory)
            input_qk_flat = input_qk.view(-1)
        else:
            # Make a contiguous copy
            input_qk_contiguous = input_qk.contiguous()
            input_qk_flat = input_qk_contiguous.view(-1)

        # Create params tensor matching RoPEParams struct
        params = torch.tensor(
            [
                num_tokens,
                num_heads,
                head_size,
                rope_theta,
                rope_factor,
                1 if interleaved else 0,  # bool as int
            ],
            dtype=torch.float32,
            device="mps",
        )

        # DEBUG: Log params being passed to kernel
        if DEBUG_ENABLED and DEBUG_VERBOSITY >= VERBOSITY_DETAILED:
            print(f"   params tensor: {params.cpu().tolist()}")
            print(f"   input_qk_flat.shape: {input_qk_flat.shape}")
            print(f"   grid threads: ({num_tokens}, {num_heads}, {head_size // 2})")

        lib = self.compiled_libraries["rope"]

        # Select kernel based on dtype
        kernel_name = "metal_rope_float32"
        if input_qk.dtype == torch.float16:
            kernel_name = "metal_rope_float16"
        elif input_qk.dtype == torch.bfloat16:
            kernel_name = "metal_rope_bfloat16"

        if hasattr(lib, kernel_name):
            # RoPE uses 3D grid: (num_tokens, num_heads, head_size/2)
            # Each thread handles one rotation pair
            num_pairs = head_size // 2

            # Call Metal RoPE kernel with flattened tensor (modifies in-place)
            # The flattened view shares memory with input_qk, so modifications
            # are automatically reflected in the original 3D tensor
            # IMPORTANT: Metal kernel expects 3D grid where:
            #   gid.x = token_idx (0 to num_tokens-1)
            #   gid.y = head_idx (0 to num_heads-1)
            #   gid.z = pair_idx (0 to num_pairs-1)
            getattr(lib, kernel_name)(
                input_qk_flat,  # buffer(0): flattened input/output tensor
                position_ids.to(torch.int32),  # buffer(1): position IDs
                params,  # buffer(2): RoPEParams
                threads=(num_tokens, num_heads, num_pairs),  # 3D grid dispatch
                group_size=(8, 8, 4),  # 3D threadgroup size (total 256 threads)
            )

            # If input was not contiguous, copy the modified data back
            if not was_contiguous:
                assert (
                    input_qk_contiguous is not None
                ), "input_qk_contiguous must be set"
                input_qk.copy_(input_qk_contiguous)

        else:
            raise RuntimeError(
                f"RoPE kernel {kernel_name} not found in compiled library"
            )

        # DEBUG MODE: Compare outputs
        if DEBUG_ENABLED:
            from . import debug_utils  # pylint: disable=import-outside-toplevel

            assert original_data is not None, "original_data must be set in debug mode"
            assert (
                pytorch_output is not None
            ), "pytorch_output must be set in debug mode"

            # Restore original data pointer and get metal output
            metal_output = input_qk.data
            input_qk.data = original_data

            # Compare Metal vs PyTorch
            matches, diagnostics = debug_utils.compare_tensors(
                metal_output,
                pytorch_output,
                atol=DEBUG_ATOL,
                rtol=DEBUG_RTOL,
                operation_name="RoPE",
            )

            # Generate and print report
            report = debug_utils.generate_report(
                diagnostics, input_metadata, verbosity=DEBUG_VERBOSITY
            )
            if report:
                print(report)

            # If mismatch detected at high verbosity, could raise error
            if not matches and DEBUG_VERBOSITY >= VERBOSITY_DETAILED:
                print(
                    "WARNING: RoPE Metal kernel output differs from PyTorch reference"
                )

            # Copy the metal result to the original input_qk
            input_qk.data.copy_(metal_output)
