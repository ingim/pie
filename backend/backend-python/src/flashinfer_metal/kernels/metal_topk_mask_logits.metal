#include <metal_stdlib>
using namespace metal;

struct Pair { float val; uint idx; };

void heapify_down(threadgroup Pair* heap, uint n, uint root) {
    uint i = root;
    while (true) {
        uint smallest = i, l = 2 * i + 1, r = 2 * i + 2;
        if (l < n && heap[l].val < heap[smallest].val) smallest = l;
        if (r < n && heap[r].val < heap[smallest].val) smallest = r;
        if (smallest == i) break;
        Pair t = heap[i]; heap[i] = heap[smallest]; heap[smallest] = t;
        i = smallest;
    }
}
void build_min_heap(threadgroup Pair* heap, uint n) {
    int start = int(n / 2) - 1;
    for (int ii = start; ii >= 0; --ii) heapify_down(heap, n, uint(ii));
}

struct TopKMaskParams { uint32_t num_tokens, vocab_size, k; };

// ===== Float32 =====
kernel void metal_topk_mask_logits_float32(
    device float* logits [[buffer(0)]],
    constant TopKMaskParams& params [[buffer(1)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint token_idx = tg, thread_id = lid;
    if (token_idx >= params.num_tokens) return;

    device float* token_logits = logits + token_idx * params.vocab_size;

    threadgroup Pair top_k_heap[256];
    threadgroup uint top_k_idx[256];

    uint k = min(params.k, params.vocab_size);
    k = min(k, (uint)256);

    if (thread_id == 0 && k > 0) {
        // Initialize heap with first k
        for (uint i = 0; i < k; ++i) {
            float v = token_logits[i];
            top_k_heap[i].val = v;
            top_k_heap[i].idx = i;
        }
        build_min_heap(top_k_heap, k);

        // Scan remainder
        for (uint i = k; i < params.vocab_size; ++i) {
            float v = token_logits[i];
            if (v > top_k_heap[0].val) {
                top_k_heap[0].val = v;
                top_k_heap[0].idx = i;
                heapify_down(top_k_heap, k, 0);
            }
        }
        // Extract indices (order doesn’t matter)
        for (uint j = 0; j < k; ++j) top_k_idx[j] = top_k_heap[j].idx;
    }

    // All threads wait for top_k_idx to be ready
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel masking: write -inf only if NOT in top-k
    for (uint i = thread_id; i < params.vocab_size; i += tg_size) {
        bool keep = false;
        // Linear membership test: O(k), k ≤ 256
        for (uint j = 0; j < k; ++j) {
            if (i == top_k_idx[j]) { keep = true; break; }
        }
        if (!keep) token_logits[i] = -INFINITY;
    }
}

// ===== bfloat16 =====
kernel void metal_topk_mask_logits_bfloat16(
    device bfloat* logits [[buffer(0)]],
    constant TopKMaskParams& params [[buffer(1)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint token_idx = tg, thread_id = lid;
    if (token_idx >= params.num_tokens) return;

    device bfloat* token_logits = logits + token_idx * params.vocab_size;

    threadgroup Pair top_k_heap[256];
    threadgroup uint top_k_idx[256];

    uint k = min(params.k, params.vocab_size);
    k = min(k, (uint)256);

    if (thread_id == 0 && k > 0) {
        // Initialize heap with first k (convert to float for comparisons)
        for (uint i = 0; i < k; ++i) {
            float v = float(token_logits[i]);
            top_k_heap[i].val = v;
            top_k_heap[i].idx = i;
        }
        build_min_heap(top_k_heap, k);

        // Scan remainder
        for (uint i = k; i < params.vocab_size; ++i) {
            float v = float(token_logits[i]);
            if (v > top_k_heap[0].val) {
                top_k_heap[0].val = v;
                top_k_heap[0].idx = i;
                heapify_down(top_k_heap, k, 0);
            }
        }
        for (uint j = 0; j < k; ++j) top_k_idx[j] = top_k_heap[j].idx;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = thread_id; i < params.vocab_size; i += tg_size) {
        bool keep = false;
        for (uint j = 0; j < k; ++j) {
            if (i == top_k_idx[j]) { keep = true; break; }
        }
        if (!keep) token_logits[i] = bfloat(-65504.0f);  // Large negative value within bfloat16 range
    }
}