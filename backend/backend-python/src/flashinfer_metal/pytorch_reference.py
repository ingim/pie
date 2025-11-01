"""
PyTorch reference implementations for Metal operations.

These implementations serve as ground truth for validating Metal kernel implementations.
They use only PyTorch native operations and are designed to match the exact semantics
of the Metal kernels.
"""

import math
import os
from typing import Optional

import torch


def rope_reference(
    input_qk: torch.Tensor,
    position_ids: torch.Tensor,
    rope_theta: float = 10000.0,
    rope_factor: float = 1.0,
    interleaved: bool = False,
    inplace: bool = False,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 4.0,
    old_context_len: int = 8192,
) -> torch.Tensor:
    """
    PyTorch reference implementation of RoPE (Rotary Position Embedding).

    This serves as ground truth for validating the Metal kernel implementation.

    Supports both 3D and 4D inputs for batched processing via broadcasting.

    Args:
        input_qk: Input tensor
                  - [num_tokens, num_heads, head_size] (3D, single sequence), OR
                  - [batch_size, num_tokens, num_heads, head_size] (4D, batched)
        position_ids: Position IDs
                      - [num_tokens] (1D, shared across batch), OR
                      - [batch_size, num_tokens] (2D, per-sequence positions)
        rope_theta: RoPE theta parameter (default: 10000.0)
        rope_factor: RoPE scaling factor (default: 1.0)
        interleaved: Layout mode (default: False for non-interleaved)
        inplace: If True, modify input_qk in-place; otherwise return new tensor (default: False)
        low_freq_factor: Low frequency factor for LLaMA 3.1 scaling (default: 1.0)
        high_freq_factor: High frequency factor for LLaMA 3.1 scaling (default: 4.0)
        old_context_len: Original context length for LLaMA 3.1 scaling (default: 8192)

    Returns:
        Output tensor with RoPE applied (same shape as input_qk)
    """
    # Support both 3D [num_tokens, num_heads, head_size] and
    # 4D [batch, num_tokens, num_heads, head_size]
    if input_qk.ndim == 3:
        # 3D input: single sequence
        num_tokens, num_heads, head_size = input_qk.shape
        _batch_size = None  # pylint: disable=unused-variable
    elif input_qk.ndim == 4:
        # 4D input: batched sequences
        _batch_size, num_tokens, num_heads, head_size = (
            input_qk.shape
        )  # pylint: disable=unused-variable
    else:
        raise ValueError(
            f"input_qk must be 3D or 4D, got {input_qk.ndim}D tensor with shape {input_qk.shape}"
        )

    device = input_qk.device
    dtype = input_qk.dtype

    # Clone to avoid modifying input unless inplace=True
    output = input_qk if inplace else input_qk.clone()

    # Compute frequency bands: theta^(-2i/d) for i in [0, d/2)
    # inv_freq shape: [head_size // 2]
    dim_indices = torch.arange(0, head_size // 2, dtype=torch.float32, device=device)
    inv_freq_base = 1.0 / (rope_theta ** (2.0 * dim_indices / head_size))

    # Apply LLaMA 3.1-style wavelength-based selective scaling
    # Following HuggingFace's exact implementation

    # Compute wavelengths for each frequency
    wavelen = 2 * math.pi / inv_freq_base

    # Define wavelength boundaries
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    # Apply selective scaling matching HuggingFace's formula:
    # wavelen < high_freq_wavelen: no change
    # wavelen > low_freq_wavelen: DIVIDE by factor (not multiply!)
    # otherwise: smooth interpolation

    # Start with low-frequency scaling (divide by factor for long wavelengths)
    inv_freq_llama = torch.where(
        wavelen > low_freq_wavelen, inv_freq_base / rope_factor, inv_freq_base
    )

    # Apply smooth interpolation for medium frequencies
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed_inv_freq = (
        1 - smooth_factor
    ) * inv_freq_llama / rope_factor + smooth_factor * inv_freq_llama

    # Determine which frequencies are in the medium range
    is_medium_freq = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
    inv_freq = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    # Compute position-dependent frequencies: position * inv_freq
    # Support both 1D position_ids [num_tokens] and 2D [batch_size, num_tokens]
    position_ids_float = position_ids.float()
    if position_ids.ndim == 1:
        # [num_tokens] -> outer product -> [num_tokens, head_size // 2]
        freqs = torch.outer(position_ids_float, inv_freq)
    elif position_ids.ndim == 2:
        # [batch_size, num_tokens] -> einsum -> [batch_size, num_tokens, head_size // 2]
        freqs = torch.einsum("bn,d->bnd", position_ids_float, inv_freq)
    else:
        raise ValueError(f"position_ids must be 1D or 2D, got {position_ids.ndim}D")

    # Compute cos and sin
    cos_freqs = torch.cos(freqs)  # [num_tokens, head_size // 2]
    sin_freqs = torch.sin(freqs)  # [num_tokens, head_size // 2]

    # For non-interleaved layout (HuggingFace style), concatenate freqs
    # torch.cat((freqs, freqs), dim=-1) gives [f0, f1, ..., f31, f0, f1, ..., f31]
    # NOT repeat_interleave which would give [f0, f0, f1, f1, ...]
    if not interleaved:
        if os.environ.get("DEBUG_ROPE"):
            print("    [rope_reference] Using NON-INTERLEAVED mode with concatenation")

        # Concatenate freqs with itself: [c0, c1, c2, ...] -> [c0, c1, c2, ..., c0, c1, c2, ...]
        # cos/sin_freqs shape: [num_tokens, head_size//2] or [batch, num_tokens, head_size//2]
        last_dim = -1
        cos_expanded = torch.cat(
            [cos_freqs, cos_freqs], dim=last_dim
        )  # [..., num_tokens, head_size]
        sin_expanded = torch.cat(
            [sin_freqs, sin_freqs], dim=last_dim
        )  # [..., num_tokens, head_size]

        # Apply rotation using vectorized operations
        # (supports both 3D and 4D inputs)
        # output shape: [num_tokens, num_heads, head_size]
        # or [batch, num_tokens, num_heads, head_size]

        # Initialize variables to satisfy type checker
        first_half = None
        second_half = None
        cos_t = None
        sin_t = None

        if input_qk.ndim == 3:
            # 3D: [num_tokens, num_heads, head_size]
            first_half = output[:, :, : head_size // 2]
            second_half = output[:, :, head_size // 2 :]

            # Broadcast cos/sin: [num_tokens, head_size] -> [num_tokens, 1, head_size]
            cos_t = cos_expanded.unsqueeze(1)  # [num_tokens, 1, head_size]
            sin_t = sin_expanded.unsqueeze(1)

        elif input_qk.ndim == 4:
            # 4D: [batch, num_tokens, num_heads, head_size]
            first_half = output[:, :, :, : head_size // 2]
            second_half = output[:, :, :, head_size // 2 :]

            # Broadcast cos/sin
            # If position_ids is 1D, cos_expanded is [num_tokens, head_size]
            # If position_ids is 2D, cos_expanded is [batch, num_tokens, head_size]
            if position_ids.ndim == 1:
                # cos_expanded: [num_tokens, head_size] -> [1, num_tokens, 1, head_size]
                cos_t = cos_expanded.unsqueeze(0).unsqueeze(2)
                sin_t = sin_expanded.unsqueeze(0).unsqueeze(2)
            else:
                # cos_expanded: [batch, num_tokens, head_size] -> [batch, num_tokens, 1, head_size]
                cos_t = cos_expanded.unsqueeze(2)
                sin_t = sin_expanded.unsqueeze(2)

        # Apply rotation: output = input * cos + rotate_half(input) * sin
        # HuggingFace's rotate_half: [-second_half, first_half]
        # After repeat_interleave, cos_t and sin_t are full head_size with repeated values
        # We apply the SAME cos/sin to both halves (not split them!)

        # Type assertion for variables set in if/elif above
        assert first_half is not None and second_half is not None
        assert cos_t is not None and sin_t is not None

        first_half_float = first_half.float()
        second_half_float = second_half.float()

        # HuggingFace formula:
        # first_half_new = first_half * cos - second_half * sin
        # second_half_new = second_half * cos + first_half * sin
        cos_first = cos_t[..., : head_size // 2]
        cos_second = cos_t[..., head_size // 2 :]
        sin_first = sin_t[..., : head_size // 2]
        sin_second = sin_t[..., head_size // 2 :]

        new_first_half = first_half_float * cos_first - second_half_float * sin_first
        new_second_half = second_half_float * cos_second + first_half_float * sin_second

        # Write back with original dtype
        if input_qk.ndim == 3:
            output[:, :, : head_size // 2] = new_first_half.to(dtype)
            output[:, :, head_size // 2 :] = new_second_half.to(dtype)
        else:  # 4D
            output[:, :, :, : head_size // 2] = new_first_half.to(dtype)
            output[:, :, :, head_size // 2 :] = new_second_half.to(dtype)
    else:
        # Interleaved layout: process pairs
        for token_idx in range(num_tokens):
            for head_idx in range(num_heads):
                for pair_idx in range(head_size // 2):
                    # Interleaved layout: [x0, x1, x2, x3, ...] -> pairs (x0,x1), (x2,x3), ...
                    idx_real = pair_idx * 2
                    idx_imag = pair_idx * 2 + 1

                    # Extract real and imaginary parts
                    real = output[token_idx, head_idx, idx_real].float()
                    imag = output[token_idx, head_idx, idx_imag].float()

                    # Apply rotation: [cos -sin; sin cos] * [real; imag]
                    cos_val = cos_freqs[token_idx, pair_idx]
                    sin_val = sin_freqs[token_idx, pair_idx]

                    new_real = real * cos_val - imag * sin_val
                    new_imag = real * sin_val + imag * cos_val

                    # Write back with original dtype
                    output[token_idx, head_idx, idx_real] = new_real.to(dtype)
                    output[token_idx, head_idx, idx_imag] = new_imag.to(dtype)

    return output


def append_paged_kv_cache_reference(
    k_input: torch.Tensor,
    v_input: torch.Tensor,
    paged_k_cache: torch.Tensor,
    paged_v_cache: torch.Tensor,
    kv_batch_indices: torch.Tensor,
    kv_positions: torch.Tensor,
    kv_page_indices: torch.Tensor,
    kv_page_indptr: torch.Tensor,
    _kv_last_page_lens: torch.Tensor,
    _num_kv_heads: int,
    _head_size: int,
) -> None:
    """
    PyTorch reference implementation of append_paged_kv_cache.

    Modifies paged_k_cache and paged_v_cache in-place.

    Args:
        k_input: Key states to append [num_tokens, num_kv_heads * head_size]
        v_input: Value states to append [num_tokens, num_kv_heads * head_size]
        paged_k_cache: Paged K cache [max_num_pages, page_size, num_kv_heads * head_size]
        paged_v_cache: Paged V cache [max_num_pages, page_size, num_kv_heads * head_size]
        kv_batch_indices: Batch index for each token [num_tokens]
        kv_positions: Position within sequence [num_tokens]
        kv_page_indices: Page indices [max_num_pages]
        kv_page_indptr: Page indptr [batch_size + 1]
        kv_last_page_lens: Last page lengths [batch_size]
        num_kv_heads: Number of KV heads
        head_size: Size of each head
    """
    num_tokens = k_input.shape[0]
    page_size = paged_k_cache.shape[1]

    # Process each token
    for token_idx in range(num_tokens):
        batch_idx = int(kv_batch_indices[token_idx].item())
        position = int(kv_positions[token_idx].item())

        # Calculate which page and offset within page
        page_idx_within_batch = position // page_size
        offset_in_page = position % page_size

        # Get the global page index
        page_start = int(kv_page_indptr[batch_idx].item())
        global_page_idx = int(
            kv_page_indices[page_start + page_idx_within_batch].item()
        )

        # Copy K and V values to cache
        # Input: [num_tokens, num_kv_heads * head_size]
        # Cache: [max_num_pages, page_size, num_kv_heads * head_size]
        paged_k_cache[global_page_idx, offset_in_page, :] = k_input[token_idx, :]
        paged_v_cache[global_page_idx, offset_in_page, :] = v_input[token_idx, :]


def attention_reference(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_page_indices: torch.Tensor,
    kv_page_indptr: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    qo_indptr: torch.Tensor,
    custom_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    PyTorch reference implementation of paged attention.

    Implements scaled dot-product attention with paged KV cache support.

    Args:
        query: Query tensor [num_tokens, num_heads, head_dim]
        kv_cache: Paged KV cache [num_pages, 2, page_size, num_kv_heads, head_dim]
        kv_page_indices: Page indices [total_pages]
        kv_page_indptr: Page indptr [batch_size + 1]
        kv_last_page_lens: Last page lengths [batch_size]
        qo_indptr: Query indptr [batch_size + 1]
        custom_mask: Optional custom mask

    Returns:
        Output tensor [num_tokens, num_heads * head_dim] (flattened for FlashInfer compatibility)
    """
    num_tokens, num_heads, head_dim = query.shape
    _num_pages, _, page_size, num_kv_heads, _ = (
        kv_cache.shape
    )  # pylint: disable=unused-variable
    # Explicitly cast to int for type checker
    page_size = int(page_size)
    num_kv_heads = int(num_kv_heads)
    batch_size = int(qo_indptr.shape[0] - 1)
    device = query.device
    dtype = query.dtype

    # Scale for attention
    scale = 1.0 / (head_dim**0.5)

    # Output buffer
    output = torch.zeros(num_tokens, num_heads, head_dim, device=device, dtype=dtype)

    # Process each batch separately
    for batch_idx in range(batch_size):
        # Get query range for this batch
        q_start = int(qo_indptr[batch_idx].item())
        q_end = int(qo_indptr[batch_idx + 1].item())
        num_queries = q_end - q_start

        if num_queries == 0:
            continue

        # Get KV page range for this batch
        kv_page_start = int(kv_page_indptr[batch_idx].item())
        kv_page_end = int(kv_page_indptr[batch_idx + 1].item())
        num_kv_pages = kv_page_end - kv_page_start
        last_page_len = int(kv_last_page_lens[batch_idx].item())

        # Compute total KV sequence length
        if num_kv_pages == 0:
            continue
        kv_seq_len = (num_kv_pages - 1) * page_size + last_page_len

        # Gather K and V from paged cache
        # K: [num_kv_pages, page_size, num_kv_heads, head_dim]
        # V: [num_kv_pages, page_size, num_kv_heads, head_dim]
        k_pages = []
        v_pages = []

        for page_offset in range(num_kv_pages):
            global_page_idx = int(kv_page_indices[kv_page_start + page_offset].item())
            k_page = kv_cache[
                global_page_idx, 0, :, :, :
            ]  # [page_size, num_kv_heads, head_dim]
            v_page = kv_cache[global_page_idx, 1, :, :, :]
            k_pages.append(k_page)
            v_pages.append(v_page)

        # Concatenate pages: [num_kv_pages * page_size, num_kv_heads, head_dim]
        k_flat = torch.cat(k_pages, dim=0)
        v_flat = torch.cat(v_pages, dim=0)

        # Trim to actual sequence length
        k_flat = k_flat[:kv_seq_len, :, :]  # [kv_seq_len, num_kv_heads, head_dim]
        v_flat = v_flat[:kv_seq_len, :, :]

        # Get queries for this batch: [num_queries, num_heads, head_dim]
        q_batch = query[q_start:q_end, :, :]

        # Handle grouped-query attention (GQA) by repeating KV heads if needed
        if num_kv_heads != num_heads:
            # Repeat KV heads to match query heads
            head_repeat = num_heads // num_kv_heads
            k_flat = k_flat.repeat_interleave(
                head_repeat, dim=1
            )  # [kv_seq_len, num_heads, head_dim]
            v_flat = v_flat.repeat_interleave(head_repeat, dim=1)

        # Compute attention scores: Q @ K^T
        # Q: [num_queries, num_heads, head_dim]
        # K: [kv_seq_len, num_heads, head_dim]
        # Scores: [num_queries, num_heads, kv_seq_len]
        q_batch_transposed = q_batch.transpose(
            0, 1
        )  # [num_heads, num_queries, head_dim]
        k_transposed = k_flat.transpose(0, 1).transpose(
            1, 2
        )  # [num_heads, head_dim, kv_seq_len]

        scores = (
            torch.bmm(q_batch_transposed, k_transposed) * scale
        )  # [num_heads, num_queries, kv_seq_len]

        scores = scores.transpose(0, 1)  # [num_queries, num_heads, kv_seq_len]

        # Apply causal mask
        # Each query at position i can attend to KV positions [0, i]
        # The queries start at position (kv_seq_len - num_queries)
        #
        # For prefill: queries are being added, so query i is at absolute
        # position (kv_seq_len - num_queries + i)
        # For decode: typically num_queries=1, query is at position
        # (kv_seq_len - 1)
        #
        # Causal mask: upper triangular with diagonal such that:
        #   query i (at absolute pos k = kv_seq_len - num_queries + i)
        #   can see KV [0..k]
        #   This means masking KV positions [k+1..kv_seq_len-1]
        #   In the [num_queries x kv_seq_len] matrix, this is positions [i+1..kv_seq_len-1] in row i
        #   But we need to account for the query's absolute position
        #
        # Actually, simpler approach:
        # Query i is at absolute position (kv_seq_len - num_queries + i)
        # It can attend to all positions up to and including its own position
        # So we want to mask everything AFTER position
        # (kv_seq_len - num_queries + i)
        #
        # For a standard causal mask: torch.triu(ones, diagonal=1)
        # masks upper triangle excluding diagonal
        # We want: torch.triu(ones, diagonal=kv_seq_len - num_queries + 1)
        # Wait, that's what we had... let me recalculate
        #
        # Actually for decode (num_queries=1, kv_seq_len=31):
        #   Query is at position 30
        #   Should attend to [0-30], so mask nothing
        #   diagonal should be kv_seq_len - 0 = kv_seq_len (to not mask anything)
        #
        # For prefill (num_queries=30, kv_seq_len=30):
        #   Queries at positions [0-29]
        #   Query 0 attends to [0], query 1 to [0-1], etc.
        #   Standard causal: diagonal=1
        #
        # The formula should be: diagonal = kv_seq_len - num_queries + 1
        # But that gives 31-1+1=31 for decode, which masks everything!
        #
        # The bug: we're creating a [1, 31] mask and setting diagonal=31
        # torch.triu(ones[1,31], diagonal=31) masks column indices >= 31, which is nothing... wait
        #
        # Let me re-read torch.triu docs: diagonal=k means keep elements at or below diagonal k
        # diagonal=0 is main diagonal, diagonal=1 is one above, etc.
        # For [1, 31] matrix: diagonal=31 means keep nothing (all above main diagonal)
        # NO WAIT: triu keeps UPPER triangle, meaning elements ON and ABOVE diagonal
        # So diagonal=31 for a [1,31] matrix means indices [i,j] where j >= i + 31
        # For i=0: j >= 31, which is nothing for j in [0-30]
        # So it masks NOTHING, which is... wait, let me just test this
        #
        # Actually the issue is: we CREATE the mask as True (ones), then MASK where mask=True
        # So True = masked out
        # triu with diagonal=d creates True for indices where col >= row + d
        # For [1, 31] with diagonal=31: col >= 0 + 31 = col >= 31, so nothing is True
        # So nothing is masked, which means attend to everything - that's CORRECT for decode!
        #
        # So the formula is actually correct. The bug must be elsewhere.
        # Let me keep the original formula but add debug output

        causal_mask = torch.triu(
            torch.ones(num_queries, kv_seq_len, device=device, dtype=torch.bool),
            diagonal=kv_seq_len - num_queries + 1,
        )

        # Apply custom_mask if provided
        # custom_mask can be:
        #   - 1-D flattened: [total_mask_elements] - needs reshaping
        #   - 2-D: [num_tokens, max_kv_len] - can slice directly
        if custom_mask is not None and custom_mask.numel() > 0:
            if custom_mask.ndim == 1:
                # Flattened mask - reshape to [num_queries, kv_seq_len]
                # Assumes mask was originally [num_queries, kv_seq_len] and flattened
                expected_size = num_queries * kv_seq_len
                if custom_mask.numel() >= expected_size:
                    # Extract the portion for this batch and reshape
                    mask_start = (
                        q_start * kv_seq_len
                    )  # Approximate - may need adjustment
                    mask_end = mask_start + expected_size
                    batch_custom_mask = custom_mask[mask_start:mask_end].reshape(
                        num_queries, kv_seq_len
                    )
                    # Combine with causal mask (logical OR: mask if EITHER says to mask)
                    causal_mask = causal_mask | batch_custom_mask
            else:
                # 2-D mask - extract the slice for this batch
                batch_custom_mask = custom_mask[q_start:q_end, :kv_seq_len]
                # Combine with causal mask (logical OR: mask if EITHER says to mask)
                causal_mask = causal_mask | batch_custom_mask

        # Try using PyTorch's native scaled_dot_product_attention for better MPS support
        use_native_sdpa = os.environ.get("PIE_METAL_USE_NATIVE_SDPA", "0") == "1"

        if use_native_sdpa and hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        ):
            # Prepare inputs for native SDPA: (batch, heads, seq, dim)
            # We have: q_batch_transposed [heads, queries, dim], k_transposed [heads, dim, kv_seq]
            q_sdpa = q_batch_transposed  # [heads, queries, dim]
            k_sdpa = k_flat.transpose(0, 1)  # [heads, kv_seq, dim]
            v_sdpa = v_flat.transpose(0, 1)  # [heads, kv_seq, dim]

            # Create attention mask in the format SDPA expects
            # SDPA expects [batch, heads, queries, kv] or [queries, kv] with True=keep, False=mask
            attn_mask = ~causal_mask if num_queries > 1 else None

            # Call native SDPA
            # pylint: disable=not-callable  # Function existence checked by hasattr above
            attn_output_transposed = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=(attn_mask is None and num_queries > 1),
            )  # [heads, queries, dim]

            attn_output = attn_output_transposed.transpose(
                0, 1
            )  # [queries, heads, dim]
        else:
            # Original manual implementation
            # Use large negative value instead of -inf for numerical stability on MPS
            # -inf can cause NaN with softmax on MPS, especially with float32
            mask_value = -1e9 if scores.dtype == torch.float32 else -65504.0
            scores = scores.masked_fill(causal_mask.unsqueeze(1), mask_value)

            # Apply softmax
            attn_weights = torch.softmax(
                scores, dim=-1
            )  # [num_queries, num_heads, kv_seq_len]

            # Apply attention to values: attn_weights @ V
            # V: [kv_seq_len, num_heads, head_dim]
            # Output: [num_queries, num_heads, head_dim]
            attn_weights_transposed = attn_weights.transpose(
                0, 1
            )  # [num_heads, num_queries, kv_seq_len]
            v_transposed = v_flat.transpose(0, 1)  # [num_heads, kv_seq_len, head_dim]
            attn_output = torch.bmm(
                attn_weights_transposed, v_transposed
            )  # [num_heads, num_queries, head_dim]
            attn_output = attn_output.transpose(
                0, 1
            )  # [num_queries, num_heads, head_dim]

        # Write to output buffer
        output[q_start:q_end, :, :] = attn_output

    # Flatten to 2D for FlashInfer API compatibility: [num_tokens, num_heads * head_dim]
    return output.view(num_tokens, num_heads * head_dim)
