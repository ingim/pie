#include <metal_stdlib>
using namespace metal;

// SiLU activation function: x / (1 + exp(-x))
float silu_activation(float x) {
    return x / (1.0f + exp(-x));
}

// SiLU and multiply kernel for bfloat16
kernel void silu_and_mul_bfloat16_kernel(
    device const bfloat* gate [[buffer(0)]],     // Gate projection input [num_tokens, intermediate_size]
    device const bfloat* up [[buffer(1)]],       // Up projection input [num_tokens, intermediate_size]
    device bfloat* output [[buffer(2)]],         // Output [num_tokens, intermediate_size]
    constant uint& num_tokens [[buffer(3)]],
    constant uint& intermediate_size [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint token_idx = gid.y; // grid Y dimension enumerates tokens
    uint dim_idx = gid.x;   // grid X dimension enumerates columns

    if (token_idx >= num_tokens || dim_idx >= intermediate_size) {
        return;
    }

    uint idx = token_idx * intermediate_size + dim_idx;

    float gate_val = float(gate[idx]);
    float up_val = float(up[idx]);

    // Apply SiLU activation to gate, then multiply by up
    float result = silu_activation(gate_val) * up_val;

    output[idx] = bfloat(result);
}

// SiLU and multiply kernel for float32
kernel void silu_and_mul_float32_kernel(
    device const float* gate [[buffer(0)]],      // Gate projection input [num_tokens, intermediate_size]
    device const float* up [[buffer(1)]],        // Up projection input [num_tokens, intermediate_size]
    device float* output [[buffer(2)]],          // Output [num_tokens, intermediate_size]
    constant uint& num_tokens [[buffer(3)]],
    constant uint& intermediate_size [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint token_idx = gid.y;
    uint dim_idx = gid.x;

    if (token_idx >= num_tokens || dim_idx >= intermediate_size) {
        return;
    }

    uint idx = token_idx * intermediate_size + dim_idx;

    float gate_val = gate[idx];
    float up_val = up[idx];

    // Apply SiLU activation to gate, then multiply by up
    output[idx] = silu_activation(gate_val) * up_val;
}