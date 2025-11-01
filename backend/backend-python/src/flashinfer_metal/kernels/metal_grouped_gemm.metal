#include <metal_stdlib>
using namespace metal;

// Grouped GEMM Metal implementation
// Performs multiple GEMM operations in parallel
// Each threadgroup handles one GEMM operation

struct GroupedGemmParams {
    uint32_t num_groups;     // Number of GEMM operations
    bool transa;             // Transpose A matrices
    bool transb;             // Transpose B matrices
    uint32_t has_bias;       // Whether bias is provided (1 = yes, 0 = no)
};

// Single GEMM kernel that processes one group
// This kernel is launched multiple times, once per group
kernel void metal_grouped_gemm_bfloat16(
    device const bfloat* A_data        [[buffer(0)]],  // Concatenated A matrices
    device const bfloat* B_data        [[buffer(1)]],  // Concatenated B matrices  
    device bfloat* C_data              [[buffer(2)]],  // Concatenated C matrices
    device const bfloat* bias_data     [[buffer(3)]],  // Concatenated bias vectors (optional)
    device const uint32_t* A_offsets   [[buffer(4)]],  // Byte offsets for each A matrix
    device const uint32_t* B_offsets   [[buffer(5)]],  // Byte offsets for each B matrix
    device const uint32_t* C_offsets   [[buffer(6)]],  // Byte offsets for each C matrix
    device const uint32_t* bias_offsets [[buffer(7)]], // Byte offsets for each bias vector
    device const uint32_t* m_array     [[buffer(8)]],  // m dimension for each group
    device const uint32_t* n_array     [[buffer(9)]],  // n dimension for each group
    device const uint32_t* k_array     [[buffer(10)]], // k dimension for each group
    constant GroupedGemmParams& params [[buffer(11)]],
    threadgroup bfloat* tile_A         [[threadgroup(0)]], // Tile memory for A
    threadgroup bfloat* tile_B         [[threadgroup(1)]], // Tile memory for B
    uint3 gid                          [[thread_position_in_grid]],
    uint3 lid                          [[thread_position_in_threadgroup]],
    uint3 tid                          [[threadgroup_position_in_grid]]
) {
    const uint32_t group_id = tid.z;  // Which GEMM operation this threadgroup handles
    const uint32_t thread_row = lid.y;
    const uint32_t thread_col = lid.x;
    const uint32_t tile_row = tid.y;
    const uint32_t tile_col = tid.x;
    
    if (group_id >= params.num_groups) {
        return;
    }

    // Get dimensions for this group
    const uint32_t m = m_array[group_id];
    const uint32_t n = n_array[group_id];
    const uint32_t k = k_array[group_id];
    
    // Get data pointers for this group (using byte offsets)
    device const bfloat* A = (device const bfloat*)((device const char*)A_data + A_offsets[group_id]);
    device const bfloat* B = (device const bfloat*)((device const char*)B_data + B_offsets[group_id]);
    device bfloat* C = (device bfloat*)((device char*)C_data + C_offsets[group_id]);
    device const bfloat* bias = nullptr;
    if (params.has_bias) {
        bias = (device const bfloat*)((device const char*)bias_data + bias_offsets[group_id]);
    }

    // Tile sizes (adjust based on shared memory and performance)
    const uint32_t TILE_M = 16;
    const uint32_t TILE_N = 16;
    const uint32_t TILE_K = 16;
    
    // Calculate which output tile this threadgroup computes
    const uint32_t global_row = tile_row * TILE_M + thread_row;
    const uint32_t global_col = tile_col * TILE_N + thread_col;
    
    if (global_row >= m || global_col >= n) {
        return;
    }

    // Initialize accumulator
    float acc = 0.0f;
    
    // Iterate over tiles in the k dimension
    for (uint32_t tile_k = 0; tile_k < (k + TILE_K - 1) / TILE_K; ++tile_k) {
        // Load tile of A into shared memory
        uint32_t a_row = tile_row * TILE_M + thread_row;
        uint32_t a_col = tile_k * TILE_K + thread_col;
        
        if (a_row < m && a_col < k) {
            uint32_t a_idx;
            if (params.transa) {
                a_idx = a_col * m + a_row;  // A is transposed: [k, m]
            } else {
                a_idx = a_row * k + a_col;  // A is normal: [m, k]
            }
            tile_A[thread_row * TILE_K + thread_col] = A[a_idx];
        } else {
            tile_A[thread_row * TILE_K + thread_col] = bfloat(0.0f);
        }
        
        // Load tile of B into shared memory
        uint32_t b_row = tile_k * TILE_K + thread_row;
        uint32_t b_col = tile_col * TILE_N + thread_col;
        
        if (b_row < k && b_col < n) {
            uint32_t b_idx;
            if (params.transb) {
                b_idx = b_col * k + b_row;  // B is transposed: [n, k]
            } else {
                b_idx = b_row * n + b_col;  // B is normal: [k, n]
            }
            tile_B[thread_row * TILE_N + thread_col] = B[b_idx];
        } else {
            tile_B[thread_row * TILE_N + thread_col] = bfloat(0.0f);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product for this tile
        for (uint32_t kk = 0; kk < TILE_K; ++kk) {
            float a_val = float(tile_A[thread_row * TILE_K + kk]);
            float b_val = float(tile_B[kk * TILE_N + thread_col]);
            acc = fma(a_val, b_val, acc);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Add bias if provided
    if (params.has_bias && bias != nullptr) {
        acc += float(bias[global_col]);
    }
    
    // Store result
    uint32_t c_idx = global_row * n + global_col;
    C[c_idx] = bfloat(acc);
}

// Float32 version for higher precision
kernel void metal_grouped_gemm_float32(
    device const float* A_data         [[buffer(0)]],
    device const float* B_data         [[buffer(1)]],
    device float* C_data               [[buffer(2)]],
    device const float* bias_data      [[buffer(3)]],
    device const uint32_t* A_offsets   [[buffer(4)]],
    device const uint32_t* B_offsets   [[buffer(5)]],
    device const uint32_t* C_offsets   [[buffer(6)]],
    device const uint32_t* bias_offsets [[buffer(7)]],
    device const uint32_t* m_array     [[buffer(8)]],
    device const uint32_t* n_array     [[buffer(9)]],
    device const uint32_t* k_array     [[buffer(10)]],
    constant GroupedGemmParams& params [[buffer(11)]],
    threadgroup float* tile_A          [[threadgroup(0)]],
    threadgroup float* tile_B          [[threadgroup(1)]],
    uint3 gid                          [[thread_position_in_grid]],
    uint3 lid                          [[thread_position_in_threadgroup]],
    uint3 tid                          [[threadgroup_position_in_grid]]
) {
    const uint32_t group_id = tid.z;
    const uint32_t thread_row = lid.y;
    const uint32_t thread_col = lid.x;
    const uint32_t tile_row = tid.y;
    const uint32_t tile_col = tid.x;
    
    if (group_id >= params.num_groups) {
        return;
    }

    const uint32_t m = m_array[group_id];
    const uint32_t n = n_array[group_id];
    const uint32_t k = k_array[group_id];
    
    device const float* A = (device const float*)((device const char*)A_data + A_offsets[group_id]);
    device const float* B = (device const float*)((device const char*)B_data + B_offsets[group_id]);
    device float* C = (device float*)((device char*)C_data + C_offsets[group_id]);
    device const float* bias = nullptr;
    if (params.has_bias) {
        bias = (device const float*)((device const char*)bias_data + bias_offsets[group_id]);
    }

    const uint32_t TILE_M = 16;
    const uint32_t TILE_N = 16;
    const uint32_t TILE_K = 16;
    
    const uint32_t global_row = tile_row * TILE_M + thread_row;
    const uint32_t global_col = tile_col * TILE_N + thread_col;
    
    if (global_row >= m || global_col >= n) {
        return;
    }

    float acc = 0.0f;
    
    for (uint32_t tile_k = 0; tile_k < (k + TILE_K - 1) / TILE_K; ++tile_k) {
        uint32_t a_row = tile_row * TILE_M + thread_row;
        uint32_t a_col = tile_k * TILE_K + thread_col;
        
        if (a_row < m && a_col < k) {
            uint32_t a_idx = params.transa ? (a_col * m + a_row) : (a_row * k + a_col);
            tile_A[thread_row * TILE_K + thread_col] = A[a_idx];
        } else {
            tile_A[thread_row * TILE_K + thread_col] = 0.0f;
        }
        
        uint32_t b_row = tile_k * TILE_K + thread_row;
        uint32_t b_col = tile_col * TILE_N + thread_col;
        
        if (b_row < k && b_col < n) {
            uint32_t b_idx = params.transb ? (b_col * k + b_row) : (b_row * n + b_col);
            tile_B[thread_row * TILE_N + thread_col] = B[b_idx];
        } else {
            tile_B[thread_row * TILE_N + thread_col] = 0.0f;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (uint32_t kk = 0; kk < TILE_K; ++kk) {
            float a_val = tile_A[thread_row * TILE_K + kk];
            float b_val = tile_B[kk * TILE_N + thread_col];
            acc = fma(a_val, b_val, acc);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (params.has_bias && bias != nullptr) {
        acc += bias[global_col];
    }
    
    uint32_t c_idx = global_row * n + global_col;
    C[c_idx] = acc;
}