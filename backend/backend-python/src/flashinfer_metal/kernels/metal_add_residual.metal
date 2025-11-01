#include <metal_stdlib>
using namespace metal;

/**
 * @brief Metal kernel for bfloat16 residual addition
 *
 * Performs element-wise addition: output[i] = input[i] + residual[i]
 * Uses bfloat16 data type for reduced precision computation
 */
kernel void add_residual_bfloat16_kernel(
    device const ushort* input          [[buffer(0)]],  // bfloat16 as uint16
    device const ushort* residual       [[buffer(1)]],  // bfloat16 as uint16
    device ushort* output               [[buffer(2)]],  // bfloat16 as uint16
    constant uint& num_elements         [[buffer(3)]],
    uint index                          [[thread_position_in_grid]]
) {
    if (index >= num_elements) {
        return;
    }

    // Convert uint16 to bfloat16, perform addition, convert back
    bfloat input_bf = as_type<bfloat>(input[index]);
    bfloat residual_bf = as_type<bfloat>(residual[index]);
    bfloat result_bf = input_bf + residual_bf;

    output[index] = as_type<ushort>(result_bf);
}

/**
 * @brief Metal kernel for float32 residual addition
 *
 * Performs element-wise addition: output[i] = input[i] + residual[i]
 * Uses full float32 precision
 */
kernel void add_residual_float32_kernel(
    device const float* input           [[buffer(0)]],
    device const float* residual        [[buffer(1)]],
    device float* output                [[buffer(2)]],
    constant uint& num_elements         [[buffer(3)]],
    uint index                          [[thread_position_in_grid]]
) {
    if (index >= num_elements) {
        return;
    }

    output[index] = input[index] + residual[index];
}

/**
 * @brief Metal kernel for in-place bfloat16 residual addition
 *
 * Performs in-place element-wise addition: input_output[i] += residual[i]
 * Modifies the input buffer directly for memory efficiency
 */
kernel void add_residual_inplace_bfloat16_kernel(
    device ushort* input_output         [[buffer(0)]],  // bfloat16 as uint16, modified in-place
    device const ushort* residual       [[buffer(1)]],  // bfloat16 as uint16
    constant uint& num_elements         [[buffer(2)]],
    uint index                          [[thread_position_in_grid]]
) {
    if (index >= num_elements) {
        return;
    }

    // Convert uint16 to bfloat16, perform addition, convert back
    bfloat input_bf = as_type<bfloat>(input_output[index]);
    bfloat residual_bf = as_type<bfloat>(residual[index]);
    bfloat result_bf = input_bf + residual_bf;

    input_output[index] = as_type<ushort>(result_bf);
}

/**
 * @brief Metal kernel for in-place float32 residual addition
 *
 * Performs in-place element-wise addition: input_output[i] += residual[i]
 * Modifies the input buffer directly for memory efficiency
 */
kernel void add_residual_inplace_float32_kernel(
    device float* input_output          [[buffer(0)]],  // Modified in-place
    device const float* residual        [[buffer(1)]],
    constant uint& num_elements         [[buffer(2)]],
    uint index                          [[thread_position_in_grid]]
) {
    if (index >= num_elements) {
        return;
    }

    input_output[index] += residual[index];
}

/**
 * @brief Vectorized Metal kernel for bfloat16 residual addition
 *
 * Processes 4 elements per thread for better memory bandwidth utilization
 * Handles remainder elements that don't fit in vector operations
 */
kernel void add_residual_bfloat16_vectorized_kernel(
    device const ushort4* input         [[buffer(0)]],  // bfloat16 as uint16, vectorized
    device const ushort4* residual      [[buffer(1)]],  // bfloat16 as uint16, vectorized
    device ushort4* output              [[buffer(2)]],  // bfloat16 as uint16, vectorized
    device const ushort* input_remainder [[buffer(3)]], // Non-vectorized remainder
    device const ushort* residual_remainder [[buffer(4)]], // Non-vectorized remainder
    device ushort* output_remainder     [[buffer(5)]],  // Non-vectorized remainder
    constant uint& num_vector_elements  [[buffer(6)]],  // Number of vector4 elements
    constant uint& num_remainder_elements [[buffer(7)]], // Number of remainder elements
    constant uint& remainder_offset     [[buffer(8)]],  // Offset to remainder elements
    uint index                          [[thread_position_in_grid]]
) {
    // Process vectorized elements
    if (index < num_vector_elements) {
        ushort4 input_vec = input[index];
        ushort4 residual_vec = residual[index];
        ushort4 result_vec;

        // Convert and add each component
        for (int i = 0; i < 4; i++) {
            bfloat input_bf = as_type<bfloat>(input_vec[i]);
            bfloat residual_bf = as_type<bfloat>(residual_vec[i]);
            result_vec[i] = as_type<ushort>(input_bf + residual_bf);
        }

        output[index] = result_vec;
    }

    // Process remainder elements
    uint remainder_index = index - num_vector_elements;
    if (remainder_index < num_remainder_elements) {
        bfloat input_bf = as_type<bfloat>(input_remainder[remainder_index]);
        bfloat residual_bf = as_type<bfloat>(residual_remainder[remainder_index]);
        output_remainder[remainder_index] = as_type<ushort>(input_bf + residual_bf);
    }
}

/**
 * @brief Vectorized Metal kernel for float32 residual addition
 *
 * Processes 4 elements per thread using float4 vectors
 * More efficient for large tensors due to better memory coalescing
 */
kernel void add_residual_float32_vectorized_kernel(
    device const float4* input          [[buffer(0)]],  // Vectorized input
    device const float4* residual       [[buffer(1)]],  // Vectorized residual
    device float4* output               [[buffer(2)]],  // Vectorized output
    device const float* input_remainder [[buffer(3)]],  // Non-vectorized remainder
    device const float* residual_remainder [[buffer(4)]], // Non-vectorized remainder
    device float* output_remainder      [[buffer(5)]],  // Non-vectorized remainder
    constant uint& num_vector_elements  [[buffer(6)]],  // Number of vector4 elements
    constant uint& num_remainder_elements [[buffer(7)]], // Number of remainder elements
    uint index                          [[thread_position_in_grid]]
) {
    // Process vectorized elements
    if (index < num_vector_elements) {
        output[index] = input[index] + residual[index];
    }

    // Process remainder elements
    uint remainder_index = index - num_vector_elements;
    if (remainder_index < num_remainder_elements) {
        output_remainder[remainder_index] = input_remainder[remainder_index] + residual_remainder[remainder_index];
    }
}