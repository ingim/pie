"""
Debug utilities for comparing Metal kernels vs PyTorch reference implementations.

Provides:
- Tensor comparison with configurable tolerances
- Detailed mismatch analysis
- Pattern detection for common errors
- Formatted diagnostic reports
"""

from typing import Any, Dict, List, Optional, Tuple

import torch


def compare_tensors(
    metal_output: torch.Tensor,
    pytorch_output: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    operation_name: str = "Operation",
) -> Tuple[bool, Dict[str, Any]]:
    """
    Compare two tensors with configurable tolerances.

    Args:
        metal_output: Output from Metal kernel
        pytorch_output: Output from PyTorch reference
        atol: Absolute tolerance
        rtol: Relative tolerance
        operation_name: Name of the operation being compared

    Returns:
        (matches, diagnostics) where diagnostics contains detailed comparison info
    """
    diagnostics = {
        "operation_name": operation_name,
        "shapes_match": metal_output.shape == pytorch_output.shape,
        "dtypes_match": metal_output.dtype == pytorch_output.dtype,
        "devices_match": metal_output.device == pytorch_output.device,
    }

    # Check shape mismatch
    if not diagnostics["shapes_match"]:
        diagnostics["error"] = (
            f"Shape mismatch: Metal={metal_output.shape}, PyTorch={pytorch_output.shape}"
        )
        return False, diagnostics

    # Move to CPU for comparison and convert to same dtype
    metal_cpu = metal_output.detach().cpu().float()
    pytorch_cpu = pytorch_output.detach().cpu().float()

    # Check for NaN or Inf
    metal_has_nan = torch.isnan(metal_cpu).any().item()
    metal_has_inf = torch.isinf(metal_cpu).any().item()
    pytorch_has_nan = torch.isnan(pytorch_cpu).any().item()
    pytorch_has_inf = torch.isinf(pytorch_cpu).any().item()

    diagnostics["metal_has_nan"] = metal_has_nan
    diagnostics["metal_has_inf"] = metal_has_inf
    diagnostics["pytorch_has_nan"] = pytorch_has_nan
    diagnostics["pytorch_has_inf"] = pytorch_has_inf

    if metal_has_nan or metal_has_inf or pytorch_has_nan or pytorch_has_inf:
        diagnostics["error"] = "NaN or Inf detected in outputs"
        return False, diagnostics

    # Compute differences
    abs_diff = torch.abs(metal_cpu - pytorch_cpu)
    rel_diff = abs_diff / (torch.abs(pytorch_cpu) + 1e-8)

    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()

    diagnostics["max_abs_diff"] = max_abs_diff
    diagnostics["mean_abs_diff"] = mean_abs_diff
    diagnostics["max_rel_diff"] = max_rel_diff
    diagnostics["mean_rel_diff"] = mean_rel_diff

    # Check if within tolerance
    matches = torch.allclose(metal_cpu, pytorch_cpu, atol=atol, rtol=rtol)

    # Find mismatch locations if not matching
    if not matches:
        # Match torch.allclose logic: abs(a - b) <= (atol + rtol * abs(b))
        tolerance = atol + rtol * torch.abs(pytorch_cpu)
        mismatch_mask = abs_diff > tolerance
        num_mismatches = mismatch_mask.sum().item()
        total_elements = metal_cpu.numel()
        mismatch_percentage = 100.0 * num_mismatches / total_elements

        diagnostics["num_mismatches"] = num_mismatches
        diagnostics["total_elements"] = total_elements
        diagnostics["mismatch_percentage"] = mismatch_percentage

        # Find worst mismatches
        flat_abs_diff = abs_diff.view(-1)
        worst_indices = torch.topk(flat_abs_diff, min(5, flat_abs_diff.numel())).indices
        diagnostics["worst_mismatch_indices"] = worst_indices.tolist()
        diagnostics["worst_mismatch_values"] = [
            (
                metal_cpu.view(-1)[idx].item(),
                pytorch_cpu.view(-1)[idx].item(),
                flat_abs_diff[idx].item(),
            )
            for idx in worst_indices
        ]

    return matches, diagnostics


def collect_tensor_metadata(
    tensor: torch.Tensor, name: str = "Tensor"
) -> Dict[str, Any]:
    """
    Collect comprehensive metadata about a tensor.

    Args:
        tensor: Input tensor
        name: Name of the tensor

    Returns:
        Dictionary containing tensor metadata
    """
    tensor_cpu = tensor.detach().cpu().float()

    metadata = {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": tensor.numel(),
        "is_contiguous": tensor.is_contiguous(),
        "strides": list(tensor.stride()) if tensor.numel() > 0 else [],
    }

    # Statistics (only if tensor has elements)
    if tensor.numel() > 0:
        metadata["min"] = tensor_cpu.min().item()
        metadata["max"] = tensor_cpu.max().item()
        metadata["mean"] = tensor_cpu.mean().item()
        metadata["std"] = tensor_cpu.std().item()
        metadata["zero_percentage"] = (
            100.0 * (tensor_cpu == 0).sum().item() / tensor.numel()
        )
        metadata["has_nan"] = torch.isnan(tensor_cpu).any().item()
        metadata["has_inf"] = torch.isinf(tensor_cpu).any().item()

    return metadata


def detect_error_patterns(diagnostics: Dict[str, Any]) -> List[str]:
    """
    Detect common error patterns from diagnostics.

    Args:
        diagnostics: Diagnostics from compare_tensors

    Returns:
        List of detected error patterns and recommendations
    """
    patterns = []

    # Check for catastrophic failures
    if diagnostics.get("metal_has_nan") or diagnostics.get("metal_has_inf"):
        patterns.append(
            "CRITICAL: Metal output contains NaN or Inf - kernel may not be executing correctly"
        )

    if diagnostics.get("pytorch_has_nan") or diagnostics.get("pytorch_has_inf"):
        patterns.append(
            "CRITICAL: PyTorch output contains NaN or Inf - reference implementation issue"
        )

    # Check for dtype issues
    if not diagnostics.get("dtypes_match"):
        patterns.append("Dtype mismatch detected - ensure proper type conversions")

    # Check for device issues
    if not diagnostics.get("devices_match"):
        patterns.append("Device mismatch detected - tensors on different devices")

    # Check for zero output (kernel not running)
    max_abs = diagnostics.get("max_abs_diff", 0)
    if max_abs == 0:
        patterns.append("Outputs are identical - this is good!")

    # Check for precision issues
    max_rel = diagnostics.get("max_rel_diff", 0)
    if max_rel < 0.01:  # Less than 1% relative error
        patterns.append(
            "Small relative error detected - likely numerical precision differences (acceptable)"
        )
    elif max_rel < 0.1:  # Less than 10% relative error
        patterns.append(
            "Moderate relative error - check for float16/bfloat16 precision loss"
        )
    else:
        patterns.append(
            "Large relative error - likely a logic error in kernel implementation"
        )

    # Check mismatch distribution
    mismatch_pct = diagnostics.get("mismatch_percentage", 0)
    if mismatch_pct > 0:
        if mismatch_pct < 1:
            patterns.append(
                f"Sparse mismatches ({mismatch_pct:.2f}%) - may be isolated edge cases"
            )
        elif mismatch_pct < 50:
            patterns.append(
                f"Moderate mismatches ({mismatch_pct:.2f}%) - partial kernel execution issue"
            )
        else:
            patterns.append(
                f"Widespread mismatches ({mismatch_pct:.2f}%) - fundamental kernel issue"
            )

    return patterns


def generate_report(
    diagnostics: Dict[str, Any],
    input_metadata: Optional[List[Dict[str, Any]]] = None,
    verbosity: int = 1,
) -> str:
    """
    Generate a formatted diagnostic report.

    Args:
        diagnostics: Diagnostics from compare_tensors
        input_metadata: Optional list of input tensor metadata
        verbosity: 0=SILENT, 1=SUMMARY, 2=DETAILED, 3=FULL

    Returns:
        Formatted report string
    """
    if verbosity == 0:  # SILENT
        return ""

    lines = []
    lines.append("=" * 80)
    lines.append(
        f"DEBUG REPORT: {diagnostics.get('operation_name', 'Unknown Operation')}"
    )
    lines.append("=" * 80)

    # Basic comparison results
    if "error" in diagnostics:
        lines.append(f"\nERROR: {diagnostics['error']}")
        lines.append("=" * 80)
        return "\n".join(lines)

    lines.append("\nOUTPUT COMPARISON:")
    lines.append(f"  Max absolute difference: {diagnostics.get('max_abs_diff', 0):.6e}")
    lines.append(
        f"  Mean absolute difference: {diagnostics.get('mean_abs_diff', 0):.6e}"
    )
    lines.append(f"  Max relative difference: {diagnostics.get('max_rel_diff', 0):.4%}")
    lines.append(
        f"  Mean relative difference: {diagnostics.get('mean_rel_diff', 0):.4%}"
    )

    if verbosity == 1:  # SUMMARY
        lines.append(
            "\nRESULT: "
            + (
                "PASS - Outputs match within tolerance"
                if diagnostics.get("num_mismatches", 0) == 0
                else "FAIL - Outputs differ"
            )
        )
        lines.append("=" * 80)
        return "\n".join(lines)

    # DETAILED or FULL verbosity
    if diagnostics.get("num_mismatches", 0) > 0:
        lines.append("\nMISMATCH ANALYSIS:")
        lines.append(f"  Number of mismatches: {diagnostics['num_mismatches']:,}")
        lines.append(f"  Total elements: {diagnostics['total_elements']:,}")
        lines.append(
            f"  Mismatch percentage: {diagnostics['mismatch_percentage']:.2f}%"
        )

        if verbosity >= 2 and "worst_mismatch_values" in diagnostics:
            lines.append("\n  Worst mismatches (Metal, PyTorch, Diff):")
            for i, (metal_val, pytorch_val, diff) in enumerate(
                diagnostics["worst_mismatch_values"]
            ):
                idx = diagnostics["worst_mismatch_indices"][i]
                lines.append(
                    f"    [{idx}]: {metal_val:.6e}, {pytorch_val:.6e}, diff={diff:.6e}"
                )

    # Pattern detection
    patterns = detect_error_patterns(diagnostics)
    if patterns and verbosity >= 2:
        lines.append("\nPATTERN ANALYSIS:")
        for pattern in patterns:
            lines.append(f"  - {pattern}")

    # Input metadata
    if input_metadata and verbosity >= 2:
        lines.append("\nINPUT METADATA:")
        for meta in input_metadata:
            lines.append(f"\n  {meta['name']}:")
            lines.append(f"    Shape: {meta['shape']}")
            lines.append(f"    Dtype: {meta['dtype']}")
            lines.append(f"    Device: {meta['device']}")
            if "min" in meta:
                lines.append(f"    Range: [{meta['min']:.4f}, {meta['max']:.4f}]")
                lines.append(f"    Mean: {meta['mean']:.4f}, Std: {meta['std']:.4f}")
                if meta["zero_percentage"] > 10:
                    lines.append(f"    Zero percentage: {meta['zero_percentage']:.1f}%")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def configure(
    enabled: bool = True, verbosity: int = 1, atol: float = 1e-5, rtol: float = 1e-3
):
    """
    Programmatically configure debug mode.

    Args:
        enabled: Enable debug mode
        verbosity: 0=SILENT, 1=SUMMARY, 2=DETAILED, 3=FULL
        atol: Absolute tolerance
        rtol: Relative tolerance
    """
    import os  # pylint: disable=import-outside-toplevel

    os.environ["PIE_METAL_DEBUG"] = "1" if enabled else "0"
    os.environ["PIE_METAL_DEBUG_VERBOSITY"] = str(verbosity)
    os.environ["PIE_METAL_DEBUG_ATOL"] = str(atol)
    os.environ["PIE_METAL_DEBUG_RTOL"] = str(rtol)
