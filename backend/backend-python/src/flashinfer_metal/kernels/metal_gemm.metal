#include <metal_stdlib>
using namespace metal;

// Metal implementation of GEMM operation for C++ row-major matrices
// Computes: C[i][j] = sum_k(A[i][k] * B_op[k][j]) where B_op = B or B^T
// - transb=0: B_op[k][j] = B[k][j] (B not transposed)
// - transb=1: B_op[k][j] = B[j][k] (B transposed, access B[j*ldb + k])
// Leading dimensions lda/ldb/ldc = number of columns per row in storage

struct GemmParams {
    uint32_t m;           // Number of rows in A and C
    uint32_t n;           // Number of columns in B and C
    uint32_t k;           // Number of columns in A / rows in B
    uint32_t lda;         // Leading dimension of A (row-major: number of columns)
    uint32_t ldb;         // Leading dimension of B (row-major: number of columns)
    uint32_t ldc;         // Leading dimension of C (row-major: number of columns)
    uint32_t transa;      // Whether A is transposed (0/1 flag)
    uint32_t transb;      // Whether B is transposed (0/1 flag)
    uint32_t use_bias;    // Whether to add bias vector (0/1 flag)
};

// Tile size for efficient matrix multiplication using threadgroup memory
constant uint TILE_SIZE = 16;

// CUDA-accurate matrix swapping version for debugging
// CUDA does: C_col[n,m] = swapped_A[n,k] * swapped_B[k,m] where:
// swapped_A = original_B, swapped_B = original_A
kernel void metal_gemm_float32_swapped(
    device const float* A              [[buffer(0)]],  // Input matrix A (float32)
    device const float* B              [[buffer(1)]],  // Input matrix B (float32)
    device const float* bias           [[buffer(2)]],  // Optional bias vector (float32)
    device float* C                    [[buffer(3)]],  // Output matrix C (float32)
    constant GemmParams& params        [[buffer(4)]],
    uint3 gid                          [[thread_position_in_grid]]
) {
    const uint row = gid.y;  // Row in output C
    const uint col = gid.x;  // Column in output C

    // Return if thread is out of bounds
    if (row >= params.m || col >= params.n) {
        return;
    }

    float sum = 0.0f;

    // CUDA computes: C_col[col][row] = sum_k(B_as_A[col,k] * A_as_B[k,row])
    // But stores result in C[row][col] (row-major result)
    for (uint k = 0; k < params.k; ++k) {
        uint a_as_b_idx, b_as_a_idx;

        // Access A as if it's the second matrix (B) in the swapped computation
        // Want A_as_B[k,row] where A_as_B has dimensions [k,m]
        bool swapped_transb = params.transa;  // Original transa becomes transb in swap
        if (swapped_transb != 0u) {
            // A is transposed: A_as_B^T[k,row] = A[row,k]
            a_as_b_idx = row * params.lda + k;
        } else {
            // A not transposed: A_as_B[k,row] = A[k,row] but A is stored [m,k]
            // So this would be A[k][row] but k index is limited to k, row to m
            // This doesn't make sense... let me reconsider
            a_as_b_idx = k * params.lda + row;
        }

        // Access B as if it's the first matrix (A) in the swapped computation
        // Want B_as_A[col,k] where B_as_A has dimensions [n,k]
        bool swapped_transa = params.transb;  // Original transb becomes transa in swap
        if (swapped_transa != 0u) {
            // B is transposed: B_as_A^T[col,k] = B[k,col]
            b_as_a_idx = k * params.ldb + col;
        } else {
            // B not transposed: B_as_A[col,k] = B[col,k]
            b_as_a_idx = col * params.ldb + k;
        }

        sum += B[b_as_a_idx] * A[a_as_b_idx];
    }

    // Add bias if provided
    if ((params.use_bias != 0u) && bias != nullptr) {
        sum += bias[col];
    }

    // Write result
    C[row * params.ldc + col] = sum;
}

// Simple non-tiled version for testing accumulation order
kernel void metal_gemm_float32_simple(
    device const float* A              [[buffer(0)]],  // Input matrix A (float32)
    device const float* B              [[buffer(1)]],  // Input matrix B (float32)
    device const float* bias           [[buffer(2)]],  // Optional bias vector (float32)
    device float* C                    [[buffer(3)]],  // Output matrix C (float32)
    constant GemmParams& params        [[buffer(4)]],
    uint3 gid                          [[thread_position_in_grid]]
) {
    const uint row = gid.y;
    const uint col = gid.x;

    // Return if thread is out of bounds
    if (row >= params.m || col >= params.n) {
        return;
    }

    float sum = 0.0f;

    // Direct dot product - no tiling
    for (uint k = 0; k < params.k; ++k) {
        uint a_idx, b_idx;

        // A access
        if (params.transa != 0u) {
            a_idx = k * params.lda + row;  // A^T[k][row] = A[row][k] transposed
        } else {
            a_idx = row * params.lda + k;  // A[row][k]
        }

        // B access
        if (params.transb != 0u) {
            b_idx = col * params.ldb + k;  // B^T[k][col] = B[col][k] transposed
        } else {
            b_idx = k * params.ldb + col;  // B[k][col]
        }

        sum += A[a_idx] * B[b_idx];
    }

    // Add bias if provided
    if ((params.use_bias != 0u) && bias != nullptr) {
        sum += bias[col];
    }

    // Write result
    C[row * params.ldc + col] = sum;
}

kernel void metal_gemm_float32(
    device const float* A              [[buffer(0)]],  // Input matrix A (float32)
    device const float* B              [[buffer(1)]],  // Input matrix B (float32)
    device const float* bias           [[buffer(2)]],  // Optional bias vector (float32)
    device float* C                    [[buffer(3)]],  // Output matrix C (float32)
    constant GemmParams& params        [[buffer(4)]],
    threadgroup float* tile_A          [[threadgroup(0)]],  // Tile memory for A
    threadgroup float* tile_B          [[threadgroup(1)]],  // Tile memory for B
    uint3 gid                          [[thread_position_in_grid]],
    uint3 lid                          [[thread_position_in_threadgroup]]
) {
    const uint row = gid.y;
    const uint col = gid.x;
    const uint local_row = lid.y;
    const uint local_col = lid.x;

    // Return if thread is out of bounds
    if (row >= params.m || col >= params.n) {
        return;
    }

    float sum = 0.0f;  // Use float for accumulation to match cuBLAS precision

    // Tile-based matrix multiplication
    for (uint tile_k = 0; tile_k < params.k; tile_k += TILE_SIZE) {
        // Load tile from A into threadgroup memory
        uint a_row = row;
        uint a_col = tile_k + local_col;

        if (a_row < params.m && a_col < params.k) {
            uint a_idx;
            if (params.transa != 0u) {
                // A is transposed: access A^T which means A[a_col][a_row] in original storage
                a_idx = a_col * params.lda + a_row;
            } else {
                // A is not transposed: access A[a_row][a_col]
                a_idx = a_row * params.lda + a_col;
            }
            tile_A[local_row * TILE_SIZE + local_col] = A[a_idx];
        } else {
            tile_A[local_row * TILE_SIZE + local_col] = 0.0f;
        }

        // Load tile from B into threadgroup memory
        uint b_row = tile_k + local_row;
        uint b_col = col;

        if (b_row < params.k && b_col < params.n) {
            uint b_idx;
            if (params.transb != 0u) {
                // B is transposed: access B^T which means B[b_col][b_row] in original storage
                b_idx = b_col * params.ldb + b_row;
            } else {
                // B is not transposed: access B[b_row][b_col]
                b_idx = b_row * params.ldb + b_col;
            }
            tile_B[local_row * TILE_SIZE + local_col] = B[b_idx];
        } else {
            tile_B[local_row * TILE_SIZE + local_col] = 0.0f;
        }

        // Synchronize threads in threadgroup
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product for this tile
        for (uint k = 0; k < TILE_SIZE; ++k) {
            float a_val = tile_A[local_row * TILE_SIZE + k];
            float b_val = tile_B[k * TILE_SIZE + local_col];
            sum += a_val * b_val;
        }

        // Synchronize before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Add bias if provided (matches cuBLAS behavior: beta = 1.0f when bias is present)
    if ((params.use_bias != 0u) && bias != nullptr) {
        sum += bias[col];
    }

    // Write result to output matrix C[row][col]
    C[row * params.ldc + col] = sum;

    // Debug: Print specific problematic values
    if (row == 0 && col == 271) {
        // This will show up in Metal shader compile output or debug logs
        // Unfortunately can't use printf in Metal, but this documents the location
    }
    if (row == 426 && col == 3216) {
        // Another problematic position
    }
}