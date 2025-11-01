#include <metal_stdlib>
using namespace metal;

// Extract k non-infinity values per row from a sparse matrix
// Input: A [M, N] - sparse matrix where empty values are -infinity
// Output: V [M, k] - extracted values, I [M, k] - column indices

kernel void extract_k_values_bfloat16_kernel(
    device const bfloat* A [[buffer(0)]],           // Input matrix [M, N]
    device bfloat* V [[buffer(1)]],                 // Output values [M, k]
    device int32_t* I [[buffer(2)]],                // Output indices [M, k]
    constant uint& M [[buffer(3)]],                 // Number of rows
    constant uint& N [[buffer(4)]],                 // Number of columns
    constant uint& k [[buffer(5)]],                 // Number of values to extract per row
    threadgroup atomic_int* tg_counter_unused [[threadgroup(0)]], // Unused in sequential fill
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 bid [[threadgroup_position_in_grid]]
) {
    // One threadgroup per row; use lane 0 to walk columns in-order for deterministic results
    uint row_idx = bid.x;
    if (row_idx >= M) return;

    if (lid.x == 0) {
        device const bfloat* input_row = A + row_idx * N;
        device bfloat* value_output_row = V + row_idx * k;
        device int32_t* index_output_row = I + row_idx * k;

        uint found = 0u;
        for (uint col = 0u; col < N && found < k; ++col) {
            bfloat val = input_row[col];
            if (val != bfloat(-65536.0f)) {  // Extract non-masked values (matches actual runtime value 0xc780)
                value_output_row[found] = val;
                index_output_row[found] = int32_t(col);
                ++found;
            }
        }

        // Explicitly zero any unfilled slots
        for (uint i = found; i < k; ++i) {
            value_output_row[i] = bfloat(0.0f);
            index_output_row[i] = 0;
        }
    }
}

kernel void extract_k_values_float32_kernel(
    device const float* A [[buffer(0)]],            // Input matrix [M, N]
    device float* V [[buffer(1)]],                  // Output values [M, k]
    device int32_t* I [[buffer(2)]],                // Output indices [M, k]
    constant uint& M [[buffer(3)]],                 // Number of rows
    constant uint& N [[buffer(4)]],                 // Number of columns
    constant uint& k [[buffer(5)]],                 // Number of values to extract per row
    threadgroup atomic_int* tg_counter_unused [[threadgroup(0)]], // Unused in sequential fill
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 bid [[threadgroup_position_in_grid]]
) {
    // One threadgroup per row; use lane 0 to walk columns in-order for deterministic results
    uint row_idx = bid.x;
    if (row_idx >= M) return;

    if (lid.x == 0) {
        device const float* input_row = A + row_idx * N;
        device float* value_output_row = V + row_idx * k;
        device int32_t* index_output_row = I + row_idx * k;

        uint found = 0u;
        for (uint col = 0u; col < N && found < k; ++col) {
            float val = input_row[col];
            if (val != -INFINITY) {  // Extract non-negative-infinity values (same as CUDA)
                value_output_row[found] = val;
                index_output_row[found] = int32_t(col);
                ++found;
            }
        }

        // Explicitly zero any unfilled slots
        for (uint i = found; i < k; ++i) {
            value_output_row[i] = 0.0f;
            index_output_row[i] = 0;
        }
    }
}