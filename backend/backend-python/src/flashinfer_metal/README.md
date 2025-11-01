# flashinfer-metal

`flashinfer-metal` provides a Metal (MPS) implementation of selected [flashinfer](https://github.com/flashinfer-ai/flashinfer) APIs for PyTorch.

## Usage

Use it as a drop-in replacement for `flashinfer`:

```python
# Before:
# import flashinfer as ops

# After:
import flashinfer_metal as ops

workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device='cpu')
wrapper = ops.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
wrapper.plan(...)
output = wrapper.run(query, kv_cache)
```


## Supported APIs

### Core
- `BatchPrefillWithPagedKVCacheWrapper` — Multi-token prefill attention  
- `BatchDecodeWithPagedKVCacheWrapper` — Single-token decode attention

### Utilities
- `apply_llama31_rope_pos_ids_inplace` — RoPE position encoding  
- `append_paged_kv_cache` — KV cache management  
- `get_seq_lens` — Sequence length calculation  
- `get_batch_indices_positions` — Batch indexing utilities  