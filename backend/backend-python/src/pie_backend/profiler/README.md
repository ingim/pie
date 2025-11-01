# Memory Profiler

Automatic memory profiling for PyTorch tensor allocations across all devices (CPU, MPS, CUDA) with operation-level tracking.

## Features

- **Automatic periodic tracking**: Captures memory snapshots every 5 seconds by default
- **Operation-level tracking**: Track which operations allocate which tensors
- **Multi-device support**: Tracks tensors on CPU, MPS (Metal), and CUDA devices
- **Operation memory map**: See memory deltas per operation with invocation counts
- **JSON export**: Saves detailed memory profiles with operation tree on shutdown
- **Thread-safe**: Safe to use in multi-threaded environments

## Usage

### Enable via Configuration

Add to your backend TOML config:

```toml
[[backend]]
enable_memory_profiling = true
```

Or via CLI:

```bash
pie config update --backend-enable-memory-profiling true
```

### What Gets Tracked

For each snapshot, the profiler records:
- Total memory usage across all devices
- Per-device breakdown (MPS, CPU, CUDA)
- Top 10 largest tensor allocations
- Tensor shapes, dtypes, and sizes
- Total tensor count

### Output

When you stop the server (Ctrl+C or SIGTERM), a JSON file is automatically saved:

```
memory_profile_YYYYMMDD_HHMMSS.json
```

### JSON Structure

```json
{
  "metadata": {
    "generated_at": "2025-01-13T10:30:00",
    "num_snapshots": 12,
    "operation_tracking_enabled": true
  },
  "snapshots": [
    {
      "timestamp": "2025-01-13T10:29:45",
      "checkpoint_name": "model_forward_pre",
      "total_mb": 2048.5,
      "total_bytes": 2147483648,
      "tensor_count": 1523,
      "device_breakdown": {
        "mps": {
          "total_mb": 2000.0,
          "count": 1500,
          "tensors": [
            {
              "size_mb": 512.0,
              "shape": [10240, 2, 16, 8, 128],
              "dtype": "torch.float16",
              "device": "mps",
              "requires_grad": false,
              "operation": "model_forward"
            }
          ]
        }
      }
    }
  ],
  "operation_allocations": {
    "model_forward": [
      {
        "memory_delta_mb": 320.5,
        "tensor_delta": 15,
        "pre_total_mb": 2048.5,
        "post_total_mb": 2369.0,
        "timestamp": "2025-01-13T10:29:45"
      }
    ]
  },
  "operation_summary": {
    "model_forward": {
      "invocation_count": 20,
      "total_memory_delta_mb": 6410.0,
      "avg_memory_delta_mb": 320.5,
      "max_memory_delta_mb": 325.2
    },
    "preprocessing": {
      "invocation_count": 20,
      "total_memory_delta_mb": 10.5,
      "avg_memory_delta_mb": 0.53,
      "max_memory_delta_mb": 1.2
    }
  }
}
```

## Programmatic API

### Basic Usage

```python
from profiler import (
    initialize_memory_tracker,
    memory_checkpoint,
    stop_memory_tracker,
)

# Initialize with custom settings
initialize_memory_tracker(output_dir="./profiles", enabled=True)

# Optional: Manual checkpoint at specific points
memory_checkpoint("after_model_load")

# Stop and save
stop_memory_tracker()
```

### Operation Tracking

Track memory allocations within specific operations:

```python
from profiler import get_memory_tracker

tracker = get_memory_tracker()

# Track memory delta for a specific operation
with tracker.track_operation("model_forward"):
    output = model.forward(input)

# Track nested operations
with tracker.track_operation("full_inference"):
    with tracker.track_operation("preprocessing"):
        processed_input = preprocess(input)

    with tracker.track_operation("forward_pass"):
        output = model.forward(processed_input)

    with tracker.track_operation("postprocessing"):
        result = postprocess(output)
```

This will generate an operation tree showing:
- Memory delta for each operation
- Number of times each operation was called
- Pre/post memory snapshots
- Which tensors were allocated during each operation

### PyTorch Operation Graph

Track low-level PyTorch operations (gemm, matmul, add, etc.) and their memory usage:

```python
from profiler import get_memory_tracker

tracker = get_memory_tracker()

# Track all PyTorch operations during forward pass
with tracker.track_pytorch_ops():
    output = model.forward(input)
```

This captures detailed information about every PyTorch operation:
- Operation name (e.g., `aten::matmul`, `aten::addmm`, `aten::linear`)
- Number of times each operation was called
- CPU and CUDA time per operation
- Memory allocations per operation
- Input tensor shapes for each operation

The output includes a `pytorch_operations` section in the JSON:

```json
{
  "pytorch_operations": [
    {
      "op_name": "aten::matmul",
      "count": 32,
      "cpu_time_us": 125000,
      "cuda_time_us": 0,
      "cpu_memory_mb": 2.5,
      "cuda_memory_mb": 0,
      "input_shapes": [[1, 512, 768], [768, 768]]
    },
    {
      "op_name": "aten::addmm",
      "count": 16,
      "cpu_time_us": 85000,
      "input_shapes": [[1, 512], [512, 2048], [2048]]
    }
  ]
}
```

This is useful for:
- Understanding which operations dominate inference time
- Identifying memory-intensive operations
- Optimizing matrix multiplication patterns
- Debugging numerical issues by tracking operation sequence

## Configuration Options

The memory tracker can be customized when initializing:

```python
initialize_memory_tracker(
    output_dir=".",      # Where to save JSON files
    enabled=True,        # Enable/disable tracking
    interval=5.0,        # Snapshot interval in seconds
)
```

## Performance Impact

- **Minimal overhead**: Snapshots use Python's garbage collector to scan tensors
- **Background thread**: Tracking runs in a daemon thread
- **Configurable interval**: Adjust snapshot frequency to balance detail vs overhead
- **Production-safe**: Disabled by default, only active when explicitly enabled

## Use Cases

1. **Debugging OOM errors**: Identify which tensors are consuming memory
2. **Memory optimization**: Find opportunities to reduce memory usage
3. **KV cache tuning**: Verify KV cache size matches expectations
4. **Model profiling**: Understand memory footprint of different model architectures

## Example: Analyzing Llama 3.2 1B Memory

Expected memory for Llama 3.2 1B with float16:
- **Model weights**: ~2 GB (1B params × 2 bytes)
- **KV cache**: Depends on `max_num_kv_pages` and `kv_page_size`
- **Activations**: Temporary during forward pass

Use the profiler to verify actual vs expected memory usage.

## Visualizing Memory Profiles

Once you've captured a memory profile JSON file, visualize it as an interactive Gantt chart.
Inside the `memory_profiler` directory, run:

```bash
python ../backend/backend-python/memory_profiler/visualize_gantt.py memory_profile_20251014_141129.json output_gantt.html
```

Then open it in your browser:

```bash
open output_gantt.html  # macOS
xdg-open output_gantt.html  # Linux
```

### What You'll See

The interactive Gantt chart shows:
- **Timeline view**: See when each operation ran and for how long
- **Parallel execution**: Operations organized in compact lanes showing concurrency
- **Color-coded operations**: Different colors for different operation types
- **Interactive tooltips**: Hover over operations to see:
  - Operation name and duration
  - Memory delta (allocation/deallocation)
  - Start and end timestamps
- **Zoom and pan**: Scroll to zoom, drag to pan across the timeline
- **Self-contained HTML**: Works offline, no external dependencies

### Finding Performance Issues

Use the Gantt chart to identify:

1. **Memory bottlenecks**: Look for operations with large memory deltas
2. **Hot paths**: Operations that run frequently (multiple instances on timeline)
3. **Sequential bottlenecks**: Long chains of operations that could be parallelized
4. **Resource contention**: Overlapping operations competing for resources
