#include <metal_stdlib>
using namespace metal;

// Include common constants and utilities
#include "metal_attention_common.metal"

// Online softmax attention with FlashAttention-style algorithm.
// Uses simdgroup reductions, cache-aware interleaved access patterns,
// and explicit loop unrolling for optimal memory bandwidth on Apple Silicon.
//
// --- Unified KV Cache Buffer Design ---
// This kernel uses a SINGLE unified buffer for both K and V caches (buffer(1)).
// Buffer layout: [num_pages, 2, page_size, num_kv_heads, head_dim]
//
// Within each page (size = 2 * page_size * num_kv_heads * head_dim elements):
//   - K cache: offset 0 to (page_size * num_kv_heads * head_dim - 1)
//   - V cache: offset (page_size * num_kv_heads * head_dim) to end
//
// The kernel uses calculate_k_offset() and calculate_v_offset() helper functions
// to compute absolute offsets from the unified buffer start.
//
// Example for page_size=16, num_kv_heads=2, head_dim=64:
//   Page 0: [K_data: 2048 elements][V_data: 2048 elements]
//   Page 1: [K_data: 2048 elements][V_data: 2048 elements]
//   ...

kernel void batch_prefill_attention_unified_fp16_simdgroup_kernel(
    device const half* q_input [[buffer(0)]],
    device const half* paged_kv_cache [[buffer(1)]],
    constant int* qo_indptr [[buffer(2)]],
    constant int* kv_page_indptr [[buffer(3)]],
    constant int* kv_page_indices [[buffer(4)]],
    constant int* kv_last_page_lens [[buffer(5)]],
    device half* output [[buffer(6)]],
    constant Params& params [[buffer(7)]],
    device float* debug_out [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid_in_tgp [[thread_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // Read parameters from the uniform buffer
    const int num_qo = params.num_qo;
    const int head_dim = params.head_dim;
    const int kv_head_dim = params.kv_head_dim;
    const int head_size = params.head_size;
    const int page_size = params.page_size;
    const int num_query_heads = params.num_query_heads;
    const int num_kv_heads = params.num_kv_heads;
    const float scale = params.scale;

    // Each threadgroup handles one query token.
    uint qo_idx = tgid.x;
    if (qo_idx >= uint(num_qo)) return;

    // Check if head_size (dimension per head) exceeds our threadgroup memory limit
    if (head_size > MAX_HEAD_DIM) return;

    const int num_simd_groups = TGP_SIZE / SIMD_SIZE;

    // --- Shared Memory Declaration ---
    threadgroup half q_s[MAX_HEAD_DIM];
    threadgroup half k_block[KERNEL_BLOCK_SIZE][MAX_HEAD_DIM];
    threadgroup half v_block[KERNEL_BLOCK_SIZE][MAX_HEAD_DIM];

    // Online softmax accumulators: m_i (running max), l_i (running sum), acc_i (output)
    threadgroup float m_i;
    threadgroup float l_i;
    threadgroup float acc_i[MAX_HEAD_DIM];

    // Scratchpad for simdgroup reductions (one slot per simdgroup)
    threadgroup float simd_scratch[4];  // Max 4 simdgroups for TGP_SIZE=128

    // Use the explicitly provided number of query heads
    const int num_heads = num_query_heads;

    // --- Get Sequence and KV Page Information ---
    int seq_id = find_sequence_id(qo_indptr, int(qo_idx));
    int kv_start_page_pos = kv_page_indptr[seq_id];
    int kv_end_page_pos = kv_page_indptr[seq_id + 1];
    int num_pages = kv_end_page_pos - kv_start_page_pos;

    if (qo_idx == 0 && tid_in_tgp == 0 && debug_out != nullptr) {
        debug_out[0] = scale;
        debug_out[1] = (float)head_dim;
        debug_out[2] = (float)page_size;
        debug_out[3] = (float)num_qo;
        debug_out[5] = (float)num_pages;
    }

    if (num_pages <= 0) {
        for (int d = tid_in_tgp; d < head_dim; d += TGP_SIZE) {
            output[qo_idx * head_dim + d] = 0.0h;
        }
        return;
    }
    int last_page_len = kv_last_page_lens[seq_id];
    int total_kv_len = (num_pages - 1) * page_size + last_page_len;
    int seq_start = qo_indptr[seq_id];
    int seq_pos = int(qo_idx) - seq_start;
    seq_pos = max(seq_pos, 0);
    // Causal masking: query at position seq_pos attends to [0, kv_seq_start + seq_pos]
    // where kv_seq_start = total_kv_len - num_qo (queries append to end of KV sequence)
    int kv_seq_start = total_kv_len - num_qo;
    int effective_kv_len = min(total_kv_len, kv_seq_start + seq_pos + 1);
    if (qo_idx == 0 && tid_in_tgp == 0 && debug_out != nullptr) {
        debug_out[4] = (float)total_kv_len;
        debug_out[6] = (float)last_page_len;
        // DEBUG: KV page indexing values
        debug_out[10] = (float)seq_id;
        debug_out[11] = (float)kv_start_page_pos;
        debug_out[12] = (float)kv_end_page_pos;
        if (num_pages > 0) {
            debug_out[13] = (float)kv_page_indices[kv_start_page_pos];  // First page index
        }
    }

    if (effective_kv_len <= 0) {
        for (int d = tid_in_tgp; d < head_dim; d += TGP_SIZE) {
            output[qo_idx * head_dim + d] = 0.0h;
        }
        return;
    }

    // --- Per-head processing: compute attention independently for each head ---
    for (int h = 0; h < num_heads; ++h) {
        // Cache-aware interleaved layout and query loading
        // Initialize and load query data in parallel to reduce barriers
        if (tid_in_tgp == 0) {
            m_i = -INFINITY;
            l_i = 0.0f;
        }

        int q_base = int(qo_idx) * head_dim + h * head_size;

        // Cache-aware vectorized query loading with simdgroup coordination
        if ((head_size & 3) == 0 && TGP_SIZE >= 4) {
            // Compute vector group info for cache-aware access
            VectorGroupInfo vg_info = compute_vector_group_info(tid_in_tgp, simd_lane_id, simd_group_id);
            int vectors_per_iter = vg_info.vectors_per_simdgroup * ((TGP_SIZE + SIMD_SIZE - 1) / SIMD_SIZE);

            // Process vectors in simdgroup-aligned chunks for optimal memory coalescing
            for (int vec_base = 0; vec_base < head_size / VECTOR_WIDTH; vec_base += vectors_per_iter) {
                int vec_idx = vec_base + vg_info.global_vector_id;
                if (vec_idx < head_size / VECTOR_WIDTH) {
                    int d = vec_idx * VECTOR_WIDTH;
                    // Load with cache-aligned access pattern
                    half4 q_vec = *((device const half4*)(q_input + q_base + d));
                    // Store interleaved for better subsequent gather performance
                    q_s[d + vg_info.lane_in_vector] = q_vec[vg_info.lane_in_vector];
                    acc_i[d + vg_info.lane_in_vector] = 0.0f;
                }
            }
            // Handle remaining elements
            for (int d = (head_size / VECTOR_WIDTH) * VECTOR_WIDTH + tid_in_tgp; d < head_size; d += TGP_SIZE) {
                q_s[d] = q_input[q_base + d];
                acc_i[d] = 0.0f;
            }
        } else {
            for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
                acc_i[d] = 0.0f;
                q_s[d] = q_input[q_base + d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Main Loop: Process KV Cache in Parallel Blocks ---
        for (int block_start = 0; block_start < effective_kv_len; block_start += KERNEL_BLOCK_SIZE) {
            // Load K/V for this block and head slice
            if (tid_in_tgp < KERNEL_BLOCK_SIZE) {
                int global_key_idx = block_start + tid_in_tgp;
                if (global_key_idx < effective_kv_len) {
                    // Map global key index to page and offset within page (bit ops for power-of-2)
                    int page_offset = global_key_idx >> (__builtin_ctz(page_size));
                    int in_page_offset = global_key_idx & (page_size - 1);
                    int page_idx = kv_page_indices[kv_start_page_pos + page_offset];
                    int kv_head = map_query_to_kv_head(h, num_query_heads, num_kv_heads);
                    uint k_offset = calculate_k_offset(in_page_offset, page_size, kv_head_dim, head_size, page_idx, kv_head);

                    // Debug: Show address calculation for first few keys
                    if (qo_idx == 0 && tid_in_tgp == 0 && h == 0 && global_key_idx < 3 && debug_out != nullptr) {
                        int debug_idx = 15 + global_key_idx * 3;  // Use debug[15-23] for address info
                        debug_out[debug_idx] = (float)k_offset;
                        debug_out[debug_idx + 1] = (float)page_idx;
                        debug_out[debug_idx + 2] = (global_key_idx < total_kv_len) ? paged_kv_cache[k_offset] : -999.0f;
                    }

                    // Calculate V offset within unified KV cache
                    uint v_offset = calculate_v_offset(in_page_offset, page_size, kv_head_dim, head_size, page_idx, kv_head);

                    // Load K and V vectors with explicit unrolling for better memory bandwidth
                    int d = 0;
                    if ((head_size & 3) == 0) {
                        #pragma unroll 2
                        for (; d + 7 < head_size; d += 8) {
                            // Load 2 vectors at once to maximize memory bandwidth
                            half4 k_vec0 = *((device const half4*)(paged_kv_cache + k_offset + d));
                            half4 k_vec1 = *((device const half4*)(paged_kv_cache + k_offset + d + 4));
                            half4 v_vec0 = *((device const half4*)(paged_kv_cache + v_offset + d));
                            half4 v_vec1 = *((device const half4*)(paged_kv_cache + v_offset + d + 4));

                            // Store to threadgroup memory using vector stores
                            *((threadgroup half4*)&k_block[tid_in_tgp][d]) = k_vec0;
                            *((threadgroup half4*)&k_block[tid_in_tgp][d + 4]) = k_vec1;
                            *((threadgroup half4*)&v_block[tid_in_tgp][d]) = v_vec0;
                            *((threadgroup half4*)&v_block[tid_in_tgp][d + 4]) = v_vec1;
                        }
                        // Handle remaining vectors
                        for (; d + 3 < head_size; d += 4) {
                            half4 k_vec = *((device const half4*)(paged_kv_cache + k_offset + d));
                            half4 v_vec = *((device const half4*)(paged_kv_cache + v_offset + d));
                            *((threadgroup half4*)&k_block[tid_in_tgp][d]) = k_vec;
                            *((threadgroup half4*)&v_block[tid_in_tgp][d]) = v_vec;
                        }
                    } else {
                        // Fallback: process 4 at a time even if not aligned
                        for (; d + 3 < head_size; d += 4) {
                            k_block[tid_in_tgp][d] = paged_kv_cache[k_offset + d];
                            k_block[tid_in_tgp][d+1] = paged_kv_cache[k_offset + d + 1];
                            k_block[tid_in_tgp][d+2] = paged_kv_cache[k_offset + d + 2];
                            k_block[tid_in_tgp][d+3] = paged_kv_cache[k_offset + d + 3];

                            v_block[tid_in_tgp][d] = paged_kv_cache[v_offset + d];
                            v_block[tid_in_tgp][d+1] = paged_kv_cache[v_offset + d + 1];
                            v_block[tid_in_tgp][d+2] = paged_kv_cache[v_offset + d + 2];
                            v_block[tid_in_tgp][d+3] = paged_kv_cache[v_offset + d + 3];
                        }
                    }
                    // Handle remaining elements
                    for (; d < head_size; ++d) {
                        k_block[tid_in_tgp][d] = paged_kv_cache[k_offset + d];
                        v_block[tid_in_tgp][d] = paged_kv_cache[v_offset + d];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Compute Q·K dot product for attention score
            float score = 0.0f;
            int global_key_idx_score = block_start + tid_in_tgp;

            if (tid_in_tgp < KERNEL_BLOCK_SIZE && global_key_idx_score < effective_kv_len) {
                int d = 0;
                if ((head_size & 3) == 0) {
                    #pragma unroll 4
                    for (; d + 7 < head_size; d += 8) {
                        // Load 2 vectors from Q and K for better prefetch
                        half4 q_vec0 = *reinterpret_cast<threadgroup const half4*>(&q_s[d]);
                        half4 q_vec1 = *reinterpret_cast<threadgroup const half4*>(&q_s[d + 4]);
                        half4 k_vec0 = *reinterpret_cast<threadgroup const half4*>(&k_block[tid_in_tgp][d]);
                        half4 k_vec1 = *reinterpret_cast<threadgroup const half4*>(&k_block[tid_in_tgp][d + 4]);

                        // Convert to float and compute dot products
                        float4 q_f0 = float4(q_vec0);
                        float4 q_f1 = float4(q_vec1);
                        float4 k_f0 = float4(k_vec0);
                        float4 k_f1 = float4(k_vec1);

                        // Vector multiply-add (FMA)
                        float4 prod0 = q_f0 * k_f0;
                        float4 prod1 = q_f1 * k_f1;
                        score += prod0.x + prod0.y + prod0.z + prod0.w;
                        score += prod1.x + prod1.y + prod1.z + prod1.w;
                    }
                    // Handle remaining vectors
                    for (; d + 3 < head_size; d += 4) {
                        half4 q_vec = *reinterpret_cast<threadgroup const half4*>(&q_s[d]);
                        half4 k_vec = *reinterpret_cast<threadgroup const half4*>(&k_block[tid_in_tgp][d]);
                        float4 q_f = float4(q_vec);
                        float4 k_f = float4(k_vec);
                        float4 prod = q_f * k_f;
                        score += prod.x + prod.y + prod.z + prod.w;
                    }
                } else {
                    // Fallback to manual unrolling for non-multiple-of-4 dimensions
                    for (; d < (head_size & ~3); d += 4) {
                        score += float(q_s[d]) * float(k_block[tid_in_tgp][d]);
                        score += float(q_s[d+1]) * float(k_block[tid_in_tgp][d+1]);
                        score += float(q_s[d+2]) * float(k_block[tid_in_tgp][d+2]);
                        score += float(q_s[d+3]) * float(k_block[tid_in_tgp][d+3]);
                    }
                }

                // Handle remaining dimensions
                for (; d < head_size; ++d) {
                    score += float(q_s[d]) * float(k_block[tid_in_tgp][d]);
                }

                score *= scale;
            } else {
                score = -INFINITY;
            }

            // Reduce to find max score in this block
            float m_j = simdgroup_max_reduction(
                (tid_in_tgp < KERNEL_BLOCK_SIZE) ? score : -INFINITY,
                simd_scratch, tid_in_tgp, simd_lane_id, simd_group_id, KERNEL_BLOCK_SIZE
            );

            // Online softmax: update running max and rescale previous accumulators
            // Thread 0 updates m_i and broadcasts old value via simd_scratch
            if (tid_in_tgp == 0) {
                float m_prev_local = m_i;
                m_i = max(m_i, m_j);
                simd_scratch[0] = m_prev_local;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float m_prev = simd_scratch[0]; // All threads read broadcasted value

            float scale_factor = fast::exp(m_prev - m_i);
            if (tid_in_tgp == 0) {
                l_i *= scale_factor;
            }
            for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
                acc_i[d] *= scale_factor;
            }

            // Compute softmax weight: exp(score - max)
            float w = (score > -INFINITY) ? fast::exp(score - m_i) : 0.0f;

            // Sum weights across block for normalization
            float l_j = simdgroup_sum_reduction(
                (tid_in_tgp < KERNEL_BLOCK_SIZE) ? w : 0.0f,
                simd_scratch, tid_in_tgp, simd_lane_id, simd_group_id, KERNEL_BLOCK_SIZE
            );

            if (tid_in_tgp == 0) {
                l_i += l_j;
            }

            // Store weights for dimension-parallel weighted sum computation
            threadgroup float w_block[KERNEL_BLOCK_SIZE];
            if (tid_in_tgp < KERNEL_BLOCK_SIZE) {
                w_block[tid_in_tgp] = w;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Accumulate weighted values: output += w * V
            int num_keys_in_block = min((int)KERNEL_BLOCK_SIZE, effective_kv_len - block_start);

            for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
                float sum_wv_d = 0.0f;

                int j = 0;
                #pragma unroll 2
                for (; j + 7 < num_keys_in_block; j += 8) {
                    // Sequential weight loads, strided V loads (hardware gather)
                    float w0 = w_block[j];
                    float w1 = w_block[j+1];
                    float w2 = w_block[j+2];
                    float w3 = w_block[j+3];
                    float w4 = w_block[j+4];
                    float w5 = w_block[j+5];
                    float w6 = w_block[j+6];
                    float w7 = w_block[j+7];
                    float v0 = float(v_block[j][d]);
                    float v1 = float(v_block[j+1][d]);
                    float v2 = float(v_block[j+2][d]);
                    float v3 = float(v_block[j+3][d]);
                    float v4 = float(v_block[j+4][d]);
                    float v5 = float(v_block[j+5][d]);
                    float v6 = float(v_block[j+6][d]);
                    float v7 = float(v_block[j+7][d]);

                    sum_wv_d += w0 * v0 + w1 * v1 + w2 * v2 + w3 * v3;
                    sum_wv_d += w4 * v4 + w5 * v5 + w6 * v6 + w7 * v7;
                }
                for (; j + 3 < num_keys_in_block; j += 4) {
                    float w0 = w_block[j];
                    float w1 = w_block[j+1];
                    float w2 = w_block[j+2];
                    float w3 = w_block[j+3];

                    float v0 = float(v_block[j][d]);
                    float v1 = float(v_block[j+1][d]);
                    float v2 = float(v_block[j+2][d]);
                    float v3 = float(v_block[j+3][d]);

                    sum_wv_d += w0 * v0 + w1 * v1 + w2 * v2 + w3 * v3;
                }

                // Handle remaining keys
                for (; j < num_keys_in_block; ++j) {
                    sum_wv_d += w_block[j] * float(v_block[j][d]);
                }

                acc_i[d] += sum_wv_d;
            }
        }

        // Normalize and write output for this head
        int out_base = int(qo_idx) * head_dim + h * head_size;
        float inv_l_i = (l_i > 1e-9f) ? (1.0f / l_i) : 0.0f;

        if ((head_size & 3) == 0 && TGP_SIZE >= 4) {
            VectorGroupInfo vg_info = compute_vector_group_info(tid_in_tgp, simd_lane_id, simd_group_id);
            int vectors_per_iter = vg_info.vectors_per_simdgroup * ((TGP_SIZE + SIMD_SIZE - 1) / SIMD_SIZE);

            for (int vec_base = 0; vec_base < head_size / VECTOR_WIDTH; vec_base += vectors_per_iter) {
                int vec_idx = vec_base + vg_info.global_vector_id;
                if (vec_idx < head_size / VECTOR_WIDTH) {
                    int d = vec_idx * VECTOR_WIDTH;
                    half4 out_vec = half4(
                        half(acc_i[d + 0] * inv_l_i),
                        half(acc_i[d + 1] * inv_l_i),
                        half(acc_i[d + 2] * inv_l_i),
                        half(acc_i[d + 3] * inv_l_i)
                    );
                    output[out_base + d + vg_info.lane_in_vector] = out_vec[vg_info.lane_in_vector];
                }
            }
            for (int d = (head_size / VECTOR_WIDTH) * VECTOR_WIDTH + tid_in_tgp; d < head_size; d += TGP_SIZE) {
                output[out_base + d] = half(acc_i[d] * inv_l_i);
            }
        } else {
            for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
                output[out_base + d] = half(acc_i[d] * inv_l_i);
            }
        }
        if (h < num_heads - 1) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    threadgroup_barrier(mem_flags::mem_device);
}

kernel void batch_prefill_attention_unified_f32_simdgroup_kernel(
    device const float* q_input [[buffer(0)]],
    device const float* paged_kv_cache [[buffer(1)]],
    constant int* qo_indptr [[buffer(2)]],
    constant int* kv_page_indptr [[buffer(3)]],
    constant int* kv_page_indices [[buffer(4)]],
    constant int* kv_last_page_lens [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant Params& params [[buffer(7)]],
    device float* debug_out [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid_in_tgp [[thread_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // Read parameters from the uniform buffer
    const int num_qo = params.num_qo;
    const int head_dim = params.head_dim;
    const int kv_head_dim = params.kv_head_dim;
    const int head_size = params.head_size;
    const int page_size = params.page_size;
    const int num_query_heads = params.num_query_heads;
    const int num_kv_heads = params.num_kv_heads;
    const float scale = params.scale;

    // Each threadgroup handles one query token.
    uint qo_idx = tgid.x;
    if (qo_idx >= uint(num_qo)) return;

    // F32 uses 2x memory, so lower head_size limit to fit in 32KB threadgroup memory
    const int MAX_F32_HEAD_DIM = 128;
    if (head_size > MAX_F32_HEAD_DIM) return;

    const int num_simd_groups = TGP_SIZE / SIMD_SIZE;

    threadgroup float q_s[MAX_F32_HEAD_DIM];
    threadgroup float k_block[KERNEL_BLOCK_SIZE][MAX_F32_HEAD_DIM];
    threadgroup float v_block[KERNEL_BLOCK_SIZE][MAX_F32_HEAD_DIM];

    // Online softmax accumulators: m_i (running max), l_i (running sum), acc_i (output)
    threadgroup float m_i;
    threadgroup float l_i;
    threadgroup float acc_i[MAX_F32_HEAD_DIM];

    // Scratchpad for simdgroup reductions (one slot per simdgroup)
    threadgroup float simd_scratch[4];

    // Use the explicitly provided number of query heads
    const int num_heads = num_query_heads;

    // --- Get Sequence and KV Page Information ---
    int seq_id = find_sequence_id(qo_indptr, int(qo_idx));
    int kv_start_page_pos = kv_page_indptr[seq_id];
    int kv_end_page_pos = kv_page_indptr[seq_id + 1];
    int num_pages = kv_end_page_pos - kv_start_page_pos;

    if (qo_idx == 0 && tid_in_tgp == 0 && debug_out != nullptr) {
        debug_out[0] = scale;
        debug_out[1] = (float)head_dim;
        debug_out[2] = (float)page_size;
        debug_out[3] = (float)num_qo;
        debug_out[5] = (float)num_pages;
    }

    if (num_pages <= 0) {
        for (int d = tid_in_tgp; d < head_dim; d += TGP_SIZE) {
            output[qo_idx * head_dim + d] = 0.0f;
        }
        return;
    }
    int last_page_len = kv_last_page_lens[seq_id];
    int total_kv_len = (num_pages - 1) * page_size + last_page_len;
    int seq_start = qo_indptr[seq_id];
    int seq_pos = int(qo_idx) - seq_start;
    seq_pos = max(seq_pos, 0);
    // Causal masking: query at position seq_pos attends to [0, kv_seq_start + seq_pos]
    // where kv_seq_start = total_kv_len - num_qo (queries append to end of KV sequence)
    int kv_seq_start = total_kv_len - num_qo;
    int effective_kv_len = min(total_kv_len, kv_seq_start + seq_pos + 1);
    if (qo_idx == 0 && tid_in_tgp == 0 && debug_out != nullptr) {
        debug_out[4] = (float)total_kv_len;
        debug_out[6] = (float)last_page_len;
        // DEBUG: KV page indexing values
        debug_out[10] = (float)seq_id;
        debug_out[11] = (float)kv_start_page_pos;
        debug_out[12] = (float)kv_end_page_pos;
        if (num_pages > 0) {
            debug_out[13] = (float)kv_page_indices[kv_start_page_pos];  // First page index
        }
    }

    if (effective_kv_len <= 0) {
        for (int d = tid_in_tgp; d < head_dim; d += TGP_SIZE) {
            output[qo_idx * head_dim + d] = 0.0f;
        }
        return;
    }

    // --- Per-head processing: compute attention independently for each head ---
    for (int h = 0; h < num_heads; ++h) {
        // Initialize online softmax state and load query into threadgroup memory
        if (tid_in_tgp == 0) {
            m_i = -INFINITY;
            l_i = 0.0f;
        }

        int q_base = int(qo_idx) * head_dim + h * head_size;

        // Cache-aware vectorized query loading with simdgroup coordination
        if ((head_size & 3) == 0 && TGP_SIZE >= 4) {
            // Compute vector group info for cache-aware access
            VectorGroupInfo vg_info = compute_vector_group_info(tid_in_tgp, simd_lane_id, simd_group_id);
            int vectors_per_iter = vg_info.vectors_per_simdgroup * ((TGP_SIZE + SIMD_SIZE - 1) / SIMD_SIZE);

            // Process vectors in simdgroup-aligned chunks for optimal memory coalescing
            for (int vec_base = 0; vec_base < head_size / VECTOR_WIDTH; vec_base += vectors_per_iter) {
                int vec_idx = vec_base + vg_info.global_vector_id;
                if (vec_idx < head_size / VECTOR_WIDTH) {
                    int d = vec_idx * VECTOR_WIDTH;
                    // Load with cache-aligned access pattern
                    float4 q_vec = *((device const float4*)(q_input + q_base + d));
                    // Store interleaved for better subsequent gather performance
                    q_s[d + vg_info.lane_in_vector] = q_vec[vg_info.lane_in_vector];
                    acc_i[d + vg_info.lane_in_vector] = 0.0f;
                }
            }
            // Handle remaining elements
            for (int d = (head_size / VECTOR_WIDTH) * VECTOR_WIDTH + tid_in_tgp; d < head_size; d += TGP_SIZE) {
                q_s[d] = q_input[q_base + d];
                acc_i[d] = 0.0f;
            }
        } else {
            for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
                acc_i[d] = 0.0f;
                q_s[d] = q_input[q_base + d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Main Loop: Process KV Cache in Parallel Blocks ---
        for (int block_start = 0; block_start < effective_kv_len; block_start += KERNEL_BLOCK_SIZE) {
            // Load K/V for this block and head slice
            if (tid_in_tgp < KERNEL_BLOCK_SIZE) {
                int global_key_idx = block_start + tid_in_tgp;
                if (global_key_idx < effective_kv_len) {
                    // Map global key index to page and offset within page (bit ops for power-of-2)
                    int page_offset = global_key_idx >> (__builtin_ctz(page_size));
                    int in_page_offset = global_key_idx & (page_size - 1);
                    int page_idx = kv_page_indices[kv_start_page_pos + page_offset];
                    int kv_head = map_query_to_kv_head(h, num_query_heads, num_kv_heads);
                    uint k_offset = calculate_k_offset(in_page_offset, page_size, kv_head_dim, head_size, page_idx, kv_head);

                    // Debug logging for first key access
                    if (qo_idx == 0 && h == 0 && block_start == 0 && tid_in_tgp == 0 && debug_out != nullptr) {
                        debug_out[19] = (float)page_idx;          // Page ID being accessed
                        debug_out[20] = (float)k_offset;          // Calculated K offset
                        debug_out[21] = paged_kv_cache[k_offset]; // First key value accessed
                        debug_out[22] = (float)in_page_offset;    // Position within page
                        debug_out[23] = (float)page_offset;       // Which page in sequence
                    }

                    // Calculate V offset within unified KV cache
                    uint v_offset = calculate_v_offset(in_page_offset, page_size, kv_head_dim, head_size, page_idx, kv_head);

                    // Load K and V vectors with explicit unrolling for better memory bandwidth
                    int d = 0;
                    if ((head_size & 3) == 0) {
                        #pragma unroll 2
                        for (; d + 7 < head_size; d += 8) {
                            // Load 2 vectors at once to maximize memory bandwidth
                            float4 k_vec0 = *((device const float4*)(paged_kv_cache + k_offset + d));
                            float4 k_vec1 = *((device const float4*)(paged_kv_cache + k_offset + d + 4));
                            float4 v_vec0 = *((device const float4*)(paged_kv_cache + v_offset + d));
                            float4 v_vec1 = *((device const float4*)(paged_kv_cache + v_offset + d + 4));

                            // Store to threadgroup memory using vector stores
                            *((threadgroup float4*)&k_block[tid_in_tgp][d]) = k_vec0;
                            *((threadgroup float4*)&k_block[tid_in_tgp][d + 4]) = k_vec1;
                            *((threadgroup float4*)&v_block[tid_in_tgp][d]) = v_vec0;
                            *((threadgroup float4*)&v_block[tid_in_tgp][d + 4]) = v_vec1;
                        }
                        // Handle remaining vectors
                        for (; d + 3 < head_size; d += 4) {
                            float4 k_vec = *((device const float4*)(paged_kv_cache + k_offset + d));
                            float4 v_vec = *((device const float4*)(paged_kv_cache + v_offset + d));
                            *((threadgroup float4*)&k_block[tid_in_tgp][d]) = k_vec;
                            *((threadgroup float4*)&v_block[tid_in_tgp][d]) = v_vec;
                        }
                    } else {
                        // Fallback: process 4 at a time even if not aligned
                        for (; d + 3 < head_size; d += 4) {
                            k_block[tid_in_tgp][d] = paged_kv_cache[k_offset + d];
                            k_block[tid_in_tgp][d+1] = paged_kv_cache[k_offset + d + 1];
                            k_block[tid_in_tgp][d+2] = paged_kv_cache[k_offset + d + 2];
                            k_block[tid_in_tgp][d+3] = paged_kv_cache[k_offset + d + 3];

                            v_block[tid_in_tgp][d] = paged_kv_cache[v_offset + d];
                            v_block[tid_in_tgp][d+1] = paged_kv_cache[v_offset + d + 1];
                            v_block[tid_in_tgp][d+2] = paged_kv_cache[v_offset + d + 2];
                            v_block[tid_in_tgp][d+3] = paged_kv_cache[v_offset + d + 3];
                        }
                    }
                    // Handle remaining elements
                    for (; d < head_size; ++d) {
                        k_block[tid_in_tgp][d] = paged_kv_cache[k_offset + d];
                        v_block[tid_in_tgp][d] = paged_kv_cache[v_offset + d];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Compute Q·K dot product for attention score
            float score = 0.0f;
            int global_key_idx_score = block_start + tid_in_tgp;

            if (tid_in_tgp < KERNEL_BLOCK_SIZE && global_key_idx_score < effective_kv_len) {
                int d = 0;
                if ((head_size & 3) == 0) {
                    #pragma unroll 4
                    for (; d + 7 < head_size; d += 8) {
                        // Load 2 vectors from Q and K for better prefetch
                        float4 q_vec0 = *reinterpret_cast<threadgroup const float4*>(&q_s[d]);
                        float4 q_vec1 = *reinterpret_cast<threadgroup const float4*>(&q_s[d + 4]);
                        float4 k_vec0 = *reinterpret_cast<threadgroup const float4*>(&k_block[tid_in_tgp][d]);
                        float4 k_vec1 = *reinterpret_cast<threadgroup const float4*>(&k_block[tid_in_tgp][d + 4]);

                        // Vector multiply-add (FMA)
                        float4 prod0 = q_vec0 * k_vec0;
                        float4 prod1 = q_vec1 * k_vec1;
                        score += prod0.x + prod0.y + prod0.z + prod0.w;
                        score += prod1.x + prod1.y + prod1.z + prod1.w;
                    }
                    // Handle remaining vectors
                    for (; d + 3 < head_size; d += 4) {
                        float4 q_vec = *reinterpret_cast<threadgroup const float4*>(&q_s[d]);
                        float4 k_vec = *reinterpret_cast<threadgroup const float4*>(&k_block[tid_in_tgp][d]);
                        float4 prod = q_vec * k_vec;
                        score += prod.x + prod.y + prod.z + prod.w;
                    }
                } else {
                    // Fallback to manual unrolling for non-multiple-of-4 dimensions
                    for (; d < (head_size & ~3); d += 4) {
                        score += q_s[d] * k_block[tid_in_tgp][d];
                        score += q_s[d+1] * k_block[tid_in_tgp][d+1];
                        score += q_s[d+2] * k_block[tid_in_tgp][d+2];
                        score += q_s[d+3] * k_block[tid_in_tgp][d+3];
                    }
                }

                // Handle remaining dimensions
                for (; d < head_size; ++d) {
                    score += q_s[d] * k_block[tid_in_tgp][d];
                }

                score *= scale;
            } else {
                score = -INFINITY;
            }

            // Reduce to find max score in this block
            float m_j = simdgroup_max_reduction(
                (tid_in_tgp < KERNEL_BLOCK_SIZE) ? score : -INFINITY,
                simd_scratch, tid_in_tgp, simd_lane_id, simd_group_id, KERNEL_BLOCK_SIZE
            );

            // Online softmax: update running max and rescale previous accumulators
            // Thread 0 updates m_i and broadcasts old value via simd_scratch
            if (tid_in_tgp == 0) {
                float m_prev_local = m_i;
                m_i = max(m_i, m_j);
                simd_scratch[0] = m_prev_local;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float m_prev = simd_scratch[0]; // All threads read broadcasted value

            float scale_factor = fast::exp(m_prev - m_i);
            if (tid_in_tgp == 0) {
                l_i *= scale_factor;
            }
            for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
                acc_i[d] *= scale_factor;
            }

            // Compute softmax weight: exp(score - max)
            float w = (score > -INFINITY) ? fast::exp(score - m_i) : 0.0f;

            // Sum weights across block for normalization
            float l_j = simdgroup_sum_reduction(
                (tid_in_tgp < KERNEL_BLOCK_SIZE) ? w : 0.0f,
                simd_scratch, tid_in_tgp, simd_lane_id, simd_group_id, KERNEL_BLOCK_SIZE
            );

            if (tid_in_tgp == 0) {
                l_i += l_j;
            }

            // Store weights for dimension-parallel weighted sum computation
            threadgroup float w_block[KERNEL_BLOCK_SIZE];
            if (tid_in_tgp < KERNEL_BLOCK_SIZE) {
                w_block[tid_in_tgp] = w;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Accumulate weighted values: output += w * V
            int num_keys_in_block = min((int)KERNEL_BLOCK_SIZE, effective_kv_len - block_start);

            for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
                float sum_wv_d = 0.0f;

                int j = 0;
                #pragma unroll 2
                for (; j + 7 < num_keys_in_block; j += 8) {
                    // Sequential weight loads, strided V loads (hardware gather)
                    float w0 = w_block[j];
                    float w1 = w_block[j+1];
                    float w2 = w_block[j+2];
                    float w3 = w_block[j+3];
                    float w4 = w_block[j+4];
                    float w5 = w_block[j+5];
                    float w6 = w_block[j+6];
                    float w7 = w_block[j+7];
                    float v0 = v_block[j][d];
                    float v1 = v_block[j+1][d];
                    float v2 = v_block[j+2][d];
                    float v3 = v_block[j+3][d];
                    float v4 = v_block[j+4][d];
                    float v5 = v_block[j+5][d];
                    float v6 = v_block[j+6][d];
                    float v7 = v_block[j+7][d];

                    sum_wv_d += w0 * v0 + w1 * v1 + w2 * v2 + w3 * v3;
                    sum_wv_d += w4 * v4 + w5 * v5 + w6 * v6 + w7 * v7;
                }
                for (; j + 3 < num_keys_in_block; j += 4) {
                    float w0 = w_block[j];
                    float w1 = w_block[j+1];
                    float w2 = w_block[j+2];
                    float w3 = w_block[j+3];

                    float v0 = v_block[j][d];
                    float v1 = v_block[j+1][d];
                    float v2 = v_block[j+2][d];
                    float v3 = v_block[j+3][d];

                    sum_wv_d += w0 * v0 + w1 * v1 + w2 * v2 + w3 * v3;
                }

                // Handle remaining keys
                for (; j < num_keys_in_block; ++j) {
                    sum_wv_d += w_block[j] * v_block[j][d];
                }

                acc_i[d] += sum_wv_d;
            }
        }

        // Normalize and write output for this head
        int out_base = int(qo_idx) * head_dim + h * head_size;
        float inv_l_i = (l_i > 1e-9f) ? (1.0f / l_i) : 0.0f;

        if ((head_size & 3) == 0 && TGP_SIZE >= 4) {
            VectorGroupInfo vg_info = compute_vector_group_info(tid_in_tgp, simd_lane_id, simd_group_id);
            int vectors_per_iter = vg_info.vectors_per_simdgroup * ((TGP_SIZE + SIMD_SIZE - 1) / SIMD_SIZE);

            for (int vec_base = 0; vec_base < head_size / VECTOR_WIDTH; vec_base += vectors_per_iter) {
                int vec_idx = vec_base + vg_info.global_vector_id;
                if (vec_idx < head_size / VECTOR_WIDTH) {
                    int d = vec_idx * VECTOR_WIDTH;
                    float4 out_vec = float4(
                        acc_i[d + 0] * inv_l_i,
                        acc_i[d + 1] * inv_l_i,
                        acc_i[d + 2] * inv_l_i,
                        acc_i[d + 3] * inv_l_i
                    );
                    output[out_base + d + vg_info.lane_in_vector] = out_vec[vg_info.lane_in_vector];
                }
            }
            for (int d = (head_size / VECTOR_WIDTH) * VECTOR_WIDTH + tid_in_tgp; d < head_size; d += TGP_SIZE) {
                output[out_base + d] = acc_i[d] * inv_l_i;
            }
        } else {
            for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
                output[out_base + d] = acc_i[d] * inv_l_i;
            }
        }
        if (h < num_heads - 1) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    threadgroup_barrier(mem_flags::mem_device);
}

// --- One TG per (qo, head) Mapping ---
// Each threadgroup handles exactly one (query_token, head) pair instead of looping over heads
// Benefits: Better GPU occupancy for small batches, multiplies TG count by num_query_heads

