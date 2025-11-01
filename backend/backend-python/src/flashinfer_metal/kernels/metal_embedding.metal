#include <metal_stdlib>
using namespace metal;

// Metal implementation of embedding lookup operation matching CUDA backend behavior
// Corresponds to embedding_lookup_kernel_128bit in common.cu

struct EmbeddingParams {
    uint32_t num_tokens;       // Number of tokens to lookup (batch dimension)
    uint32_t hidden_size;      // Embedding dimension
    uint32_t vocab_size;       // Vocabulary size (for bounds checking)
};

// Optimized embedding lookup using SIMD groups for memory coalescing
// Each threadgroup processes one token lookup, with threads copying embedding vectors efficiently
kernel void metal_embedding_lookup_bfloat16(
    device const bfloat* embedding_matrix  [[buffer(0)]],  // [vocab_size, hidden_size] embedding table
    device const int32_t* indices         [[buffer(1)]],  // [num_tokens] token indices to lookup
    device bfloat* output                  [[buffer(2)]],  // [num_tokens, hidden_size] output embeddings
    constant EmbeddingParams& params       [[buffer(3)]],
    uint3 gid                              [[thread_position_in_grid]],
    uint3 lid                              [[thread_position_in_threadgroup]],
    uint3 tid                              [[threadgroup_position_in_grid]]
) {
    // Each threadgroup processes one token lookup
    const uint32_t token_idx = tid.x;

    if (token_idx >= params.num_tokens) {
        return;
    }

    // Get the vocabulary index for this token (with bounds checking)
    const int32_t vocab_idx = indices[token_idx];
    if (vocab_idx < 0 || vocab_idx >= static_cast<int32_t>(params.vocab_size)) {
        // Invalid index - zero out the output (matches CUDA behavior)
        for (uint32_t i = lid.x; i < params.hidden_size; i += 32) {  // 32 threads per SIMD group
            if (i < params.hidden_size) {
                // Use explicit bfloat constructor for zero to avoid half-to-bfloat assignment error
                output[token_idx * params.hidden_size + i] = bfloat(0.0f);
            }
        }
        return;
    }

    // Calculate source and destination pointers
    device const bfloat* source_embedding = embedding_matrix + vocab_idx * params.hidden_size;
    device bfloat* dest_embedding = output + token_idx * params.hidden_size;

    // Use all threads in the threadgroup to copy the embedding vector
    // This matches the CUDA grid-stride loop pattern for optimal memory throughput
    const uint32_t threads_per_group = 32;  // Match CUDA block size optimization

    for (uint32_t i = lid.x; i < params.hidden_size; i += threads_per_group) {
        if (i < params.hidden_size) {
            dest_embedding[i] = source_embedding[i];
        }
    }
}

// Alternative implementation using Metal's SIMD operations for even better performance
// This version processes multiple elements per thread using vector loads when possible
kernel void metal_embedding_lookup_vectorized_bfloat16(
    device const bfloat* embedding_matrix  [[buffer(0)]],  // [vocab_size, hidden_size]
    device const int32_t* indices         [[buffer(1)]],  // [num_tokens]
    device bfloat* output                  [[buffer(2)]],  // [num_tokens, hidden_size]
    constant EmbeddingParams& params       [[buffer(3)]],
    uint3 gid                              [[thread_position_in_grid]],
    uint3 lid                              [[thread_position_in_threadgroup]],
    uint3 tid                              [[threadgroup_position_in_grid]]
) {
    const uint32_t token_idx = tid.x;

    if (token_idx >= params.num_tokens) {
        return;
    }

    const int32_t vocab_idx = indices[token_idx];
    if (vocab_idx < 0 || vocab_idx >= static_cast<int32_t>(params.vocab_size)) {
        // Zero out invalid lookups
        for (uint32_t i = lid.x * 4; i < params.hidden_size; i += 128) {  // 32 threads × 4 elements
            for (uint32_t j = 0; j < 4 && (i + j) < params.hidden_size; ++j) {
                // Use explicit bfloat constructor for zero to avoid half-to-bfloat assignment error
                output[token_idx * params.hidden_size + i + j] = bfloat(0.0f);
            }
        }
        return;
    }

    // Vector copy: process 4 bfloat16 elements per thread iteration
    device const bfloat* source_base = embedding_matrix + vocab_idx * params.hidden_size;
    device bfloat* dest_base = output + token_idx * params.hidden_size;

    // Each thread handles 4 consecutive elements to improve memory bandwidth
    const uint32_t elements_per_thread = 4;
    const uint32_t threads_per_group = 32;

    for (uint32_t base_i = lid.x * elements_per_thread;
         base_i < params.hidden_size;
         base_i += threads_per_group * elements_per_thread) {

        // Load 4 elements at once when aligned and within bounds
        if (base_i + elements_per_thread <= params.hidden_size) {
            // Vectorized copy for aligned access
            for (uint32_t j = 0; j < elements_per_thread; ++j) {
                dest_base[base_i + j] = source_base[base_i + j];
            }
        } else {
            // Handle remaining elements individually for bounds safety
            for (uint32_t j = 0; j < elements_per_thread && (base_i + j) < params.hidden_size; ++j) {
                dest_base[base_i + j] = source_base[base_i + j];
            }
        }
    }
}

// Float32 version of embedding lookup kernel
kernel void metal_embedding_lookup_float32(
    device const float* embedding_matrix  [[buffer(0)]],  // [vocab_size, hidden_size] embedding table
    device const int32_t* indices         [[buffer(1)]],  // [num_tokens] token indices to lookup
    device float* output                   [[buffer(2)]],  // [num_tokens, hidden_size] output embeddings
    constant EmbeddingParams& params       [[buffer(3)]],
    uint3 gid                              [[thread_position_in_grid]],
    uint3 lid                              [[thread_position_in_threadgroup]],
    uint3 tid                              [[threadgroup_position_in_grid]]
) {
    // Each threadgroup processes one token lookup
    const uint32_t token_idx = tid.x;

    if (token_idx >= params.num_tokens) {
        return;
    }

    // Get the vocabulary index for this token (with bounds checking)
    const int32_t vocab_idx = indices[token_idx];
    if (vocab_idx < 0 || vocab_idx >= static_cast<int32_t>(params.vocab_size)) {
        // Invalid index - zero out the output
        for (uint32_t i = lid.x; i < params.hidden_size; i += 32) {
            if (i < params.hidden_size) {
                output[token_idx * params.hidden_size + i] = 0.0f;
            }
        }
        return;
    }

    // Calculate source and destination pointers
    device const float* source_embedding = embedding_matrix + vocab_idx * params.hidden_size;
    device float* dest_embedding = output + token_idx * params.hidden_size;

    // Use all threads in the threadgroup to copy the embedding vector
    const uint32_t threads_per_group = 32;

    for (uint32_t i = lid.x; i < params.hidden_size; i += threads_per_group) {
        if (i < params.hidden_size) {
            dest_embedding[i] = source_embedding[i];
        }
    }
}