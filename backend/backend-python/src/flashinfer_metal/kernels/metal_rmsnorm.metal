#include <metal_stdlib>
using namespace metal;

// CUDA-compatible rsqrt approximation to match FlashInfer's rsqrt.approx.ftz.f32
// This function approximates CUDA hardware behavior for consistent results
inline float precise_rsqrt(float x) {
    // Use hardware fast inverse square root and refine once (approx CUDA rsqrt.approx)
    float y = fast::rsqrt(x);
    y = y * (1.5f - 0.5f * x * y * y);
    return y;
}

// Metal implementation of RMSNorm (Root Mean Square Normalization)
// Corresponds to flashinfer::norm::RMSNorm<T> from FlashInfer
// Formula: output = input * rsqrt(mean(input^2) + eps) * weight

struct RMSNormParams {
    uint32_t num_tokens;     // Number of tokens (sequence length)
    uint32_t hidden_size;    // Hidden dimension size
    float eps;               // Epsilon for numerical stability (e.g., 1e-5)
};

// RMSNorm kernel using threadgroup reduction for computing variance
// Each threadgroup processes one token, threads cooperate to compute RMS
kernel void metal_rmsnorm_bfloat16(
    device const bfloat* input         [[buffer(0)]],  // [num_tokens, hidden_size] input tensor
    device const bfloat* weight        [[buffer(1)]],  // [hidden_size] scale weights
    device bfloat* output              [[buffer(2)]],  // [num_tokens, hidden_size] output tensor
    constant RMSNormParams& params     [[buffer(3)]],
    threadgroup float* shared_sum      [[threadgroup(0)]], // Shared memory for reduction
    uint3 gid                          [[thread_position_in_grid]],
    uint3 lid                          [[thread_position_in_threadgroup]],
    uint3 tid                          [[threadgroup_position_in_grid]]
) {
    const uint32_t token_idx = tid.x;
    const uint32_t thread_id = lid.x;
    const uint32_t threads_per_group = 256; // Match typical Metal threadgroup size

    if (token_idx >= params.num_tokens) {
        return;
    }

    // Calculate base pointers for this token
    device const bfloat* input_token = input + token_idx * params.hidden_size;
    device bfloat* output_token = output + token_idx * params.hidden_size;

    // Phase 1: Compute sum of squares using threadgroup reduction
    float local_sum = 0.0f;

    // Each thread processes multiple elements (grid-stride loop)
    for (uint32_t i = thread_id; i < params.hidden_size; i += threads_per_group) {
        float val = float(input_token[i]);
        local_sum = fma(val, val, local_sum);
    }

    // Store local sum in shared memory
    shared_sum[thread_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction to compute total sum
    for (uint32_t stride = threads_per_group / 2; stride > 0; stride /= 2) {
        if (thread_id < stride) {
            shared_sum[thread_id] += shared_sum[thread_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 computes the RMS normalization factor
    float rms_scale = 0.0f;
    if (thread_id == 0) {
        float mean_square = shared_sum[0] / float(params.hidden_size);
        rms_scale = precise_rsqrt(mean_square + params.eps);
    }

    // Broadcast RMS scale to all threads in threadgroup
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_id == 0) {
        shared_sum[0] = rms_scale;  // Reuse shared memory for broadcast
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    rms_scale = shared_sum[0];

    // Phase 2: Apply normalization and scaling
    for (uint32_t i = thread_id; i < params.hidden_size; i += threads_per_group) {
        float scaled = float(input_token[i]) * rms_scale * float(weight[i]);
        output_token[i] = bfloat(scaled);
    }
}

// Float32 version of RMSNorm kernel for Python bindings
kernel void metal_rmsnorm_float32(
    device const float* input         [[buffer(0)]],  // [num_tokens, hidden_size] input tensor
    device const float* weight        [[buffer(1)]],  // [hidden_size] scale weights
    device float* output              [[buffer(2)]],  // [num_tokens, hidden_size] output tensor
    constant RMSNormParams& params    [[buffer(3)]],
    threadgroup float* shared_sum     [[threadgroup(0)]], // Shared memory for reduction
    uint3 gid                         [[thread_position_in_grid]],
    uint3 lid                         [[thread_position_in_threadgroup]],
    uint3 tid                         [[threadgroup_position_in_grid]]
) {
    const uint32_t token_idx = tid.x;
    const uint32_t thread_id = lid.x;
    const uint32_t threads_per_group = 256;

    if (token_idx >= params.num_tokens) {
        return;
    }

    // Calculate base pointers for this token
    device const float* input_token = input + token_idx * params.hidden_size;
    device float* output_token = output + token_idx * params.hidden_size;

    // Phase 1: Compute sum of squares using threadgroup reduction
    float local_sum = 0.0f;

    // Each thread processes multiple elements (grid-stride loop)
    for (uint32_t i = thread_id; i < params.hidden_size; i += threads_per_group) {
        float val = input_token[i];
        local_sum = fma(val, val, local_sum);
    }

    // Store local sum in shared memory
    shared_sum[thread_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction to compute total sum
    for (uint32_t stride = threads_per_group / 2; stride > 0; stride /= 2) {
        if (thread_id < stride) {
            shared_sum[thread_id] += shared_sum[thread_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 computes the RMS normalization factor
    float rms_scale = 0.0f;
    if (thread_id == 0) {
        float mean_square = shared_sum[0] / float(params.hidden_size);
        rms_scale = precise_rsqrt(mean_square + params.eps);
    }

    // Broadcast RMS scale to all threads in threadgroup
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_id == 0) {
        shared_sum[0] = rms_scale;  // Reuse shared memory for broadcast
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    rms_scale = shared_sum[0];

    // Phase 2: Apply normalization and scaling
    for (uint32_t i = thread_id; i < params.hidden_size; i += threads_per_group) {
        float scaled = input_token[i] * rms_scale * weight[i];
        output_token[i] = scaled;
    }
}

// Alternative implementation using SIMD group operations for better efficiency
// This version uses Metal's SIMD group reductions for smaller, faster reductions
kernel void metal_rmsnorm_simd_bfloat16(
    device const bfloat* input         [[buffer(0)]],  // [num_tokens, hidden_size]
    device const bfloat* weight        [[buffer(1)]],  // [hidden_size]
    device bfloat* output              [[buffer(2)]],  // [num_tokens, hidden_size]
    constant RMSNormParams& params     [[buffer(3)]],
    threadgroup float* shared_sum      [[threadgroup(0)]],
    uint3 gid                          [[thread_position_in_grid]],
    uint3 lid                          [[thread_position_in_threadgroup]],
    uint3 tid                          [[threadgroup_position_in_grid]],
    uint simd_lane_id                  [[thread_index_in_simdgroup]],
    uint simd_group_id                 [[simdgroup_index_in_threadgroup]]
) {
    const uint32_t token_idx = tid.x;
    const uint32_t thread_id = lid.x;
    const uint32_t threads_per_group = 256;
    const uint32_t simd_size = 32;  // Metal SIMD group size
    const uint32_t num_simd_groups = threads_per_group / simd_size;

    if (token_idx >= params.num_tokens) {
        return;
    }

    device const bfloat* input_token = input + token_idx * params.hidden_size;
    device bfloat* output_token = output + token_idx * params.hidden_size;

    // Phase 1: Compute sum of squares with SIMD group reduction
    float local_sum = 0.0f;

    for (uint32_t i = thread_id; i < params.hidden_size; i += threads_per_group) {
        float val = float(input_token[i]);
        local_sum = fma(val, val, local_sum);
    }

    // SIMD group reduction (more efficient than threadgroup for this size)
    local_sum = simd_sum(local_sum);

    // Store SIMD group sums in shared memory
    if (simd_lane_id == 0) {
        shared_sum[simd_group_id] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction across SIMD groups
    if (simd_group_id == 0) {
        float sum = (simd_lane_id < num_simd_groups) ? shared_sum[simd_lane_id] : 0.0f;
        sum = simd_sum(sum);

        if (simd_lane_id == 0) {
            float mean_square = sum / float(params.hidden_size);
            float rms_scale = precise_rsqrt(mean_square + params.eps);
            shared_sum[0] = rms_scale;  // Store for broadcast
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Broadcast RMS scale and apply normalization
    float rms_scale = shared_sum[0];

    for (uint32_t i = thread_id; i < params.hidden_size; i += threads_per_group) {
        float scaled = float(input_token[i]) * rms_scale * float(weight[i]);
        output_token[i] = bfloat(scaled);
    }
}