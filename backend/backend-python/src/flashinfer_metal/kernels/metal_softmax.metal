#include <metal_stdlib>
using namespace metal;

/**
 * Metal kernel for softmax computation with numerical stability
 *
 * Each threadgroup processes one sequence (row) of the batch
 * Uses shared memory for max reduction and sum reduction
 *
 * Grid: [batch_size, 1, 1]
 * Threadgroup: [min(vocab_size, 1024), 1, 1]
 */
kernel void softmax_kernel(
    device const float* input            [[buffer(0)]],  // [batch_size, vocab_size]
    device float* output                 [[buffer(1)]],  // [batch_size, vocab_size]
    constant uint32_t& batch_size        [[buffer(2)]],
    constant uint32_t& vocab_size        [[buffer(3)]],
    constant float& temperature          [[buffer(4)]],
    threadgroup float* shared_memory     [[threadgroup(0)]],  // Size: threadgroup_size
    uint gid                            [[threadgroup_position_in_grid]],
    uint tid                            [[thread_position_in_threadgroup]],
    uint threadgroup_size               [[threads_per_threadgroup]]
) {
    uint batch_idx = gid;

    if (batch_idx >= batch_size) return;

    // Pointers to current sequence
    device const float* seq_input = input + batch_idx * vocab_size;
    device float* seq_output = output + batch_idx * vocab_size;

    // Phase 1: Find maximum value for numerical stability
    float local_max = -INFINITY;
    for (uint i = tid; i < vocab_size; i += threadgroup_size) {
        float val = seq_input[i] / temperature;  // Apply temperature scaling
        local_max = max(local_max, val);
    }

    // Store local max in shared memory
    shared_memory[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce to find global max (sequential reduction for correctness)
    // Use simple sequential reduction to ensure all values are included
    if (tid == 0) {
        for (uint i = 1; i < threadgroup_size; ++i) {
            shared_memory[0] = max(shared_memory[0], shared_memory[i]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max = shared_memory[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (uint i = tid; i < vocab_size; i += threadgroup_size) {
        float val = seq_input[i] / temperature - global_max;

        // Clamp to prevent overflow/underflow: exp(-700) ≈ 0, exp(700) is near float limit
        val = clamp(val, -700.0f, 700.0f);

        float exp_val = exp(val);
        seq_output[i] = exp_val;  // Store temporarily
        local_sum += exp_val;
    }

    // Store local sum in shared memory
    shared_memory[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce to find global sum (sequential reduction for correctness)
    // Use simple sequential reduction to ensure all values are included
    if (tid == 0) {
        for (uint i = 1; i < threadgroup_size; ++i) {
            shared_memory[0] += shared_memory[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_sum = shared_memory[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Normalize by sum to get probabilities
    // Add safety check for numerical stability
    float inv_sum = 1.0f / max(global_sum, 1e-10f);

    for (uint i = tid; i < vocab_size; i += threadgroup_size) {
        seq_output[i] *= inv_sum;
    }
}

// Large vocabulary kernel commented out for now - implement later if needed
// TODO: Implement large vocabulary softmax kernel for vocab_size > 1024