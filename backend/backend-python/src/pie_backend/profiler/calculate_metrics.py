"""
Operation Metrics Calculator

Calculates FLOPs, data volume, and arithmetic intensity for neural network operations.
Used for bottleneck analysis and roofline modeling.
"""

from typing import Dict, List, Tuple, Any


def calculate_operation_metrics(
    module_type: str,
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    dtype: str = "float16",
) -> Dict[str, Any]:
    """
    Calculate FLOPs and data volume for an operation.

    Args:
        module_type: Type of module (e.g., "Linear", "RMSNorm", "SiLU")
        input_shapes: List of input tensor shapes
        output_shape: Output tensor shape
        dtype: Data type ("float16" or "float32")

    Returns:
        Dictionary containing:
        - flops: Total floating point operations
        - input_bytes: Total input data in bytes
        - output_bytes: Output data in bytes
        - params_bytes: Parameter data in bytes (weights)
        - total_data_bytes: Total data transferred
        - arithmetic_intensity: FLOPs per byte (FLOPs/total_data_bytes)
    """
    metrics = {
        "flops": 0,
        "input_bytes": 0,
        "output_bytes": 0,
        "params_bytes": 0,
        "total_data_bytes": 0,
        "arithmetic_intensity": 0.0,
    }

    # Bytes per element based on dtype
    bytes_per_element = 2 if dtype == "float16" else 4

    if not input_shapes or not output_shape:
        return metrics

    try:
        if module_type == "Linear":
            _calculate_linear_metrics(
                metrics, input_shapes, output_shape, bytes_per_element
            )
        elif module_type == "RMSNorm":
            _calculate_rmsnorm_metrics(
                metrics, input_shapes, output_shape, bytes_per_element
            )
        elif module_type == "SiLU":
            _calculate_silu_metrics(
                metrics, input_shapes, output_shape, bytes_per_element
            )
        elif module_type in ["GELU", "ReLU", "Tanh", "Sigmoid"]:
            _calculate_activation_metrics(
                metrics, input_shapes, output_shape, bytes_per_element
            )
        elif module_type == "Softmax":
            _calculate_softmax_metrics(
                metrics, input_shapes, output_shape, bytes_per_element
            )
        elif module_type == "LayerNorm":
            _calculate_layernorm_metrics(
                metrics, input_shapes, output_shape, bytes_per_element
            )
        else:
            # Generic element-wise operation fallback
            _calculate_elementwise_metrics(
                metrics, input_shapes, output_shape, bytes_per_element
            )

        # Calculate arithmetic intensity
        if metrics["total_data_bytes"] > 0:
            metrics["arithmetic_intensity"] = (
                metrics["flops"] / metrics["total_data_bytes"]
            )

    except Exception as e:  # pylint: disable=broad-except
        # If calculation fails, return zeros rather than crashing
        print(f"Warning: Failed to calculate metrics for {module_type}: {e}")

    return metrics


def _calculate_linear_metrics(
    metrics: Dict[str, Any],
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    bytes_per_element: int,
) -> None:
    """Calculate metrics for Linear (matrix multiply) operation."""
    if not input_shapes or len(input_shapes[0]) < 2:
        return

    # Linear: Y = X @ W + b
    # X: [batch, in_features], W: [in_features, out_features]
    batch_size = _prod(input_shapes[0][:-1])  # Handle multi-dim batch
    in_features = input_shapes[0][-1]
    out_features = output_shape[-1]

    # FLOPs: 2 * batch * in_features * out_features (matmul = multiply + add)
    metrics["flops"] = 2 * batch_size * in_features * out_features

    # Data volume
    metrics["input_bytes"] = batch_size * in_features * bytes_per_element
    metrics["params_bytes"] = (
        in_features * out_features * bytes_per_element
    )  # Weight matrix
    metrics["output_bytes"] = batch_size * out_features * bytes_per_element
    metrics["total_data_bytes"] = (
        metrics["input_bytes"] + metrics["params_bytes"] + metrics["output_bytes"]
    )


def _calculate_rmsnorm_metrics(
    metrics: Dict[str, Any],
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    bytes_per_element: int,
) -> None:
    """Calculate metrics for RMSNorm operation."""
    _ = input_shapes  # Unused but required for interface consistency

    # RMSNorm: output = input * scale / sqrt(mean(input^2) + eps)
    # Operations per element: square, sum, divide, sqrt, multiply
    num_elements = _prod(output_shape)

    # Approximate FLOPs per element:
    # - square: 1
    # - sum reduction: log2(N) operations per batch
    # - mean: 1 divide
    # - sqrt: ~5 ops
    # - scale multiply: 1
    # Total: ~10 operations per element (rough estimate)
    metrics["flops"] = num_elements * 10

    # Data volume
    metrics["input_bytes"] = num_elements * bytes_per_element
    metrics["params_bytes"] = output_shape[-1] * bytes_per_element  # scale parameter
    metrics["output_bytes"] = num_elements * bytes_per_element
    metrics["total_data_bytes"] = (
        metrics["input_bytes"] + metrics["params_bytes"] + metrics["output_bytes"]
    )


def _calculate_silu_metrics(
    metrics: Dict[str, Any],
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    bytes_per_element: int,
) -> None:
    """Calculate metrics for SiLU (Swish) activation: x * sigmoid(x)."""
    _ = input_shapes  # Unused but required for interface consistency
    # SiLU: x * sigmoid(x) = x * (1 / (1 + exp(-x)))
    # Operations per element: exp, add, divide, multiply
    num_elements = _prod(output_shape)

    # Approximate FLOPs: exp(~10 ops) + add(1) + div(1) + mul(1) = ~13 ops
    metrics["flops"] = num_elements * 13

    # Data volume (element-wise, no parameters)
    metrics["input_bytes"] = num_elements * bytes_per_element
    metrics["output_bytes"] = num_elements * bytes_per_element
    metrics["total_data_bytes"] = metrics["input_bytes"] + metrics["output_bytes"]


def _calculate_activation_metrics(
    metrics: Dict[str, Any],
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    bytes_per_element: int,
) -> None:
    """Calculate metrics for generic activation functions."""
    _ = input_shapes  # Unused but required for interface consistency
    num_elements = _prod(output_shape)

    # Approximate FLOPs: ~5 operations per element
    metrics["flops"] = num_elements * 5

    # Data volume (element-wise, no parameters)
    metrics["input_bytes"] = num_elements * bytes_per_element
    metrics["output_bytes"] = num_elements * bytes_per_element
    metrics["total_data_bytes"] = metrics["input_bytes"] + metrics["output_bytes"]


def _calculate_softmax_metrics(
    metrics: Dict[str, Any],
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    bytes_per_element: int,
) -> None:
    """Calculate metrics for Softmax operation."""
    _ = input_shapes  # Unused but required for interface consistency
    # Softmax: exp(x_i) / sum(exp(x_j))
    # Operations: exp per element, sum reduction, divide per element
    num_elements = _prod(output_shape)

    # Approximate: exp(~10 ops) + divide(1) = ~11 ops per element
    metrics["flops"] = num_elements * 11

    # Data volume
    metrics["input_bytes"] = num_elements * bytes_per_element
    metrics["output_bytes"] = num_elements * bytes_per_element
    metrics["total_data_bytes"] = metrics["input_bytes"] + metrics["output_bytes"]


def _calculate_layernorm_metrics(
    metrics: Dict[str, Any],
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    bytes_per_element: int,
) -> None:
    """Calculate metrics for LayerNorm operation."""
    _ = input_shapes  # Unused but required for interface consistency
    # Similar to RMSNorm but with mean subtraction
    num_elements = _prod(output_shape)

    # ~12 operations per element
    metrics["flops"] = num_elements * 12

    # Data volume
    metrics["input_bytes"] = num_elements * bytes_per_element
    metrics["params_bytes"] = output_shape[-1] * 2 * bytes_per_element  # gamma, beta
    metrics["output_bytes"] = num_elements * bytes_per_element
    metrics["total_data_bytes"] = (
        metrics["input_bytes"] + metrics["params_bytes"] + metrics["output_bytes"]
    )


def _calculate_elementwise_metrics(
    metrics: Dict[str, Any],
    input_shapes: List[Tuple[int, ...]],
    output_shape: Tuple[int, ...],
    bytes_per_element: int,
) -> None:
    """Calculate metrics for generic element-wise operations."""
    num_elements = _prod(output_shape)

    # Generic element-wise: ~2 operations per element
    metrics["flops"] = num_elements * 2

    # Data volume: sum all inputs + output
    total_input_elements = sum(_prod(shape) for shape in input_shapes)
    metrics["input_bytes"] = total_input_elements * bytes_per_element
    metrics["output_bytes"] = num_elements * bytes_per_element
    metrics["total_data_bytes"] = metrics["input_bytes"] + metrics["output_bytes"]


def _prod(shape: Tuple[int, ...]) -> int:
    """Calculate product of all dimensions in a shape."""
    result = 1
    for dim in shape:
        result *= dim
    return result


def calculate_attention_metrics(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    kv_seq_len: int | None = None,
    dtype: str = "float16",
) -> Dict[str, Any]:
    """
    Calculate FLOPs and data volume for attention operation.

    Attention: softmax(Q @ K^T / sqrt(d)) @ V

    Args:
        batch_size: Batch size
        seq_len: Query sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        kv_seq_len: Key/Value sequence length (defaults to seq_len if None)
        dtype: Data type

    Returns:
        Dictionary with flops, data_bytes, arithmetic_intensity
    """
    if kv_seq_len is None:
        kv_seq_len = seq_len

    bytes_per_element = 2 if dtype == "float16" else 4

    # FLOPs breakdown:
    # 1. Q @ K^T: [batch, heads, seq_len, head_dim] @ [batch, heads, head_dim, kv_seq_len]
    #    = 2 * batch * heads * seq_len * kv_seq_len * head_dim
    qk_flops = 2 * batch_size * num_heads * seq_len * kv_seq_len * head_dim

    # 2. Softmax: ~5 ops per element (exp, sum, divide)
    #    Applied to [batch, heads, seq_len, kv_seq_len]
    softmax_flops = 5 * batch_size * num_heads * seq_len * kv_seq_len

    # 3. Attention @ V: [batch, heads, seq_len, kv_seq_len] @ [batch, heads, kv_seq_len, head_dim]
    #    = 2 * batch * heads * seq_len * kv_seq_len * head_dim
    av_flops = 2 * batch_size * num_heads * seq_len * kv_seq_len * head_dim

    total_flops = qk_flops + softmax_flops + av_flops

    # Data volume:
    # Input: Q, K, V tensors
    q_bytes = batch_size * seq_len * num_heads * head_dim * bytes_per_element
    k_bytes = batch_size * kv_seq_len * num_heads * head_dim * bytes_per_element
    v_bytes = batch_size * kv_seq_len * num_heads * head_dim * bytes_per_element

    # Intermediate: attention scores
    scores_bytes = batch_size * num_heads * seq_len * kv_seq_len * bytes_per_element

    # Output
    output_bytes = batch_size * seq_len * num_heads * head_dim * bytes_per_element

    total_data_bytes = q_bytes + k_bytes + v_bytes + scores_bytes + output_bytes

    return {
        "flops": total_flops,
        "data_bytes": total_data_bytes,
        "arithmetic_intensity": total_flops / max(total_data_bytes, 1),
        "input_bytes": q_bytes + k_bytes + v_bytes,
        "output_bytes": output_bytes,
    }


def get_hardware_specs(device: str = "mps") -> Dict[str, Any]:
    """
    Get hardware specifications for bottleneck analysis.

    Args:
        device: Device type ("mps" or "cuda")

    Returns:
        Dictionary with hardware specs:
        - name: Human-readable name
        - peak_tflops_fp16: Peak compute throughput in TFLOPs/s
        - peak_bandwidth_gbs: Peak memory bandwidth in GB/s
        - ridge_point: Arithmetic intensity where performance transitions
    """
    specs = {
        "mps": {
            "name": "Apple Silicon (M1/M2/M3)",
            "peak_tflops_fp16": 10.0,  # M1: ~10 TFLOPs FP16
            "peak_bandwidth_gbs": 200,  # M1: ~200 GB/s unified memory
            "peak_memory_gb": 16,  # Varies by model
        },
        "cuda": {
            "name": "NVIDIA GPU (A100 example)",
            "peak_tflops_fp16": 312.0,  # A100: 312 TFLOPs FP16
            "peak_bandwidth_gbs": 1555,  # A100: 1555 GB/s HBM2
            "peak_memory_gb": 40,  # A100 40GB
        },
    }

    device_key = device.lower()
    if device_key not in specs:
        device_key = "mps"  # Default to MPS

    spec = specs[device_key].copy()

    # Calculate ridge point: the arithmetic intensity where performance transitions
    # from bandwidth-bound to compute-bound
    spec["ridge_point"] = (
        spec["peak_tflops_fp16"] * 1e12 / (spec["peak_bandwidth_gbs"] * 1e9)
    )

    return spec
