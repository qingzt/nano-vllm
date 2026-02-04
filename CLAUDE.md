# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nano-vLLM is a lightweight vLLM implementation built from scratch in ~1,200 lines of Python code. It provides fast offline inference for large language models with performance comparable to (or exceeding) vLLM, while maintaining a highly readable codebase.

**Key Features:**
- PagedAttention for efficient KV cache management
- Prefix Caching to reuse computation for common prefixes
- Tensor Parallelism for multi-GPU inference
- CUDA Graph optimization for reduced kernel launch overhead
- Continuous Batching for dynamic request scheduling

## Development Setup

### Installation

For development (with CPU-only PyTorch by default):
```bash
uv sync
```

For GPU support, modify `pyproject.toml` and `.devcontainer/devcontainer.json`:
- In `pyproject.toml`: Uncomment `triton` and `flash-attn` dependencies, comment out the `[tool.uv.sources]` section
- In `.devcontainer/devcontainer.json`: Uncomment the `--gpus all` runArgs

### Model Download

Download model weights manually:
```bash
uvx --from huggingface-hub hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B/
```

### Running Examples

Basic usage example:
```bash
python example.py
```

Benchmark against vLLM:
```bash
python bench.py
```

## Architecture Overview

### Core Components

The codebase is organized into three main layers:

1. **Engine Layer** (`nanovllm/engine/`)
   - `llm_engine.py`: Main LLM interface, orchestrates the entire inference pipeline
   - `scheduler.py`: Manages request queues (waiting/running) and decides which sequences to process
   - `block_manager.py`: Manages KV cache blocks with PagedAttention and Prefix Caching
   - `model_runner.py`: Executes model inference, handles GPU resources and CUDA graphs
   - `sequence.py`: Core data structure representing a generation request

2. **Model Layer** (`nanovllm/models/`)
   - `qwen3.py`: Qwen3 model implementation (currently the only supported model)

3. **Layers** (`nanovllm/layers/`)
   - `attention.py`: Implements FlashAttention (prefill) and PagedAttention (decode)
   - `sampler.py`: Token sampling with Gumbel sampling
   - `linear.py`: Tensor parallel linear layers
   - `rotary_embedding.py`: RoPE positional encoding
   - `embed_head.py`: Embedding and LM head layers
   - `layernorm.py`: RMSNorm implementation
   - `activation.py`: SwiGLU activation

### Data Flow

**Initialization:**
```
LLM() → Config → ModelRunner (loads model, allocates KV cache) → Scheduler (with BlockManager)
```

**Generation Pipeline:**
```
User prompts → Tokenizer → Sequence objects → Scheduler.waiting queue
→ Scheduler.schedule() → BlockManager.allocate() → ModelRunner.run()
→ Prefill phase (FlashAttention varlen) → Sampler → append tokens
→ Decode phase (PagedAttention) → Sampler → append tokens (loop)
→ Finish condition → BlockManager.deallocate() → Return decoded text
```

### Key Concepts

**Sequence States:**
- `WAITING`: In queue, not yet allocated KV cache
- `RUNNING`: Actively generating tokens
- `FINISHED`: Reached max_tokens or EOS token

**Two-Phase Inference:**
- **Prefill**: Process all prompt tokens at once using FlashAttention varlen (variable length)
- **Decode**: Generate one token at a time using PagedAttention to read from KV cache

**Block Management:**
- KV cache is divided into fixed-size blocks (default: 256 tokens per block)
- Each sequence has a `block_table` mapping logical positions to physical block IDs
- Blocks are shared across sequences when prefixes match (Prefix Caching)
- Reference counting ensures blocks are freed when no longer needed

**Scheduling Strategy:**
1. Prefill new requests first (up to `max_num_batched_tokens` limit)
2. Batch decode for running sequences (up to `max_num_seqs` limit)
3. Preempt (pause) sequences if KV cache is full, moving them back to waiting queue

## Important Implementation Details

### Context Management

The `nanovllm/utils/context.py` module uses thread-local storage to pass attention metadata between layers without explicit parameter passing. This includes:
- `cu_seqlens_q`, `cu_seqlens_k`: Cumulative sequence lengths for FlashAttention
- `slot_mapping`: Physical KV cache positions for writing
- `block_tables`: Block mappings for PagedAttention
- `context_lens`: Sequence lengths for decode phase

Always call `reset_context()` after each inference step.

### Tensor Parallelism

When `tensor_parallel_size > 1`:
- Main process (rank 0) spawns worker processes for ranks 1..N
- Communication via shared memory (`SharedMemory` named "nanovllm")
- Main process writes method calls to shared memory, workers execute in parallel
- Uses NCCL for GPU communication, GLOO for CPU
- Linear layers split along output dimension, with AllReduce for final aggregation

### CUDA Graph Optimization

Enabled by default (disable with `enforce_eager=True`):
- Captures computation graphs for batch sizes: [1, 2, 4, 8, 16, 32, ..., 512]
- Only used in decode phase (prefill uses eager mode)
- Selects smallest graph that fits current batch size
- Reuses input/output buffers to avoid memory allocation overhead

### Prefix Caching

Automatic hash-based caching:
- Each complete block (256 tokens) is hashed using xxhash
- Hash includes previous block's hash for chain verification
- When allocating blocks, checks `hash_to_block_id` for matches
- If match found and tokens identical, reuses block and increments `ref_count`
- `seq.num_cached_tokens` tracks how many tokens skip computation in prefill

## Configuration Parameters

Key parameters in `Config` (nanovllm/config.py):
- `max_num_batched_tokens` (default: 16384): Maximum tokens in a single prefill batch
- `max_num_seqs` (default: 512): Maximum sequences processed simultaneously
- `max_model_len` (default: 4096): Maximum sequence length
- `gpu_memory_utilization` (default: 0.8): Fraction of GPU memory for KV cache
- `tensor_parallel_size` (default: 1): Number of GPUs for tensor parallelism
- `enforce_eager` (default: False): Disable CUDA graph optimization
- `kvcache_block_size` (default: 256): Tokens per KV cache block (must be multiple of 256)

## Adding New Models

To support a new model architecture:

1. Create a new file in `nanovllm/models/` (e.g., `llama.py`)
2. Implement the model class following the Qwen3 pattern:
   - Use tensor parallel layers from `nanovllm/layers/`
   - Implement `forward(input_ids, positions)` returning hidden states
   - Implement `compute_logits(hidden_states)` for the LM head
3. Update `nanovllm/utils/loader.py` to handle the new model's weight loading
4. The model should write KV cache using `slot_mapping` from context

## Testing and Benchmarking

The repository doesn't include formal unit tests. Testing is done via:
- `example.py`: Functional test with real model inference
- `bench.py`: Performance comparison against vLLM

When making changes:
1. Run `python example.py` to verify basic functionality
2. Run `python bench.py` to check performance hasn't regressed
3. Monitor GPU memory usage and throughput metrics

## Performance Considerations

- **Prefill throughput**: Limited by `max_num_batched_tokens` and memory bandwidth
- **Decode throughput**: Limited by `max_num_seqs` and compute capacity
- **Memory usage**: Primarily determined by KV cache size (controlled by `gpu_memory_utilization`)
- **CUDA graphs**: Provide ~1.5x speedup in decode phase but require warmup time
- **Tensor parallelism**: Scales well for large models but adds communication overhead

## Common Pitfalls

1. **Block size must be multiple of 256**: This is a hard requirement due to FlashAttention alignment
2. **Greedy sampling not supported**: `temperature` must be > 1e-10 (use small value like 0.01 for near-greedy)
3. **CPU mode limitations**: CUDA graphs and some optimizations only work with GPU
4. **Model path must be a directory**: The code expects a local directory with model files, not a HuggingFace model ID
5. **Shared memory cleanup**: If processes crash, manually clean up with `rm /dev/shm/nanovllm`

## Code Style Notes

- Chinese comments are used throughout the codebase for detailed explanations
- Type hints use modern Python 3.10+ syntax (e.g., `list[int]` instead of `List[int]`)
- Dataclasses are preferred for configuration objects
- The code prioritizes readability over micro-optimizations
