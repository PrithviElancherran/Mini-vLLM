# My Mini-vLLM

A from-scratch custom implementation of vLLM's core inference engine, designed for learning and experimentation. This project provides self-contained implementations of paged attention, flash attention, and the complete LLM serving pipeline—no black-box dependencies.

Built on concepts from [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm), with custom Triton kernels for all attention mechanisms.

> **New to vLLM?** Check out [HowToApproachvLLM.md](HowToApproachvLLM.md) for a step-by-step implementation guide covering layers, models, paged attention, CUDA graphs, and scheduling.

## Key Features

- **Paged Attention**: Memory-efficient KV cache management using fixed-size blocks, enabling dynamic batching and reducing memory fragmentation
- **Flash Attention**: O(N) memory implementation using online softmax for the prefill phase
- **Two-Phase Scheduling**: Separate handling of prefill (prompt processing) and decode (token generation) phases
- **Prefix Caching**: Automatic detection and reuse of duplicate prefixes across requests using xxhash
- **Tensor Parallelism**: Multi-GPU support with NCCL-based communication and sharded weight loading
- **CUDA Graph Capture**: Pre-captured execution graphs for faster decode iterations
- **Custom Triton Kernels**: GPU-optimized kernels for attention and KV cache operations

## How It Works

Mini-vLLM implements the same two-phase inference approach used by production vLLM:

### Prefill Phase
When a new prompt arrives, the engine processes all input tokens at once using flash attention. This phase is compute-bound and benefits from processing multiple tokens in parallel.

### Decode Phase
After prefilling, the engine generates tokens one at a time. Each new token attends to all previous tokens stored in the paged KV cache. This phase is memory-bound, so the engine batches multiple sequences together to maximize GPU utilization.

### Paged KV Cache
Instead of pre-allocating a contiguous KV cache per sequence, Mini-vLLM divides GPU memory into fixed-size blocks (default: 256 tokens). Sequences are assigned blocks on-demand, and blocks can be shared across sequences with identical prefixes.

```
┌─────────────────────────────────────────────────────────┐
│                    GPU Memory                           │
├──────────┬──────────┬──────────┬──────────┬────────────┤
│ Block 0  │ Block 1  │ Block 2  │ Block 3  │    ...     │
│ (Seq A)  │ (Seq A)  │ (Seq B)  │ (Shared) │            │
└──────────┴──────────┴──────────┴──────────┴────────────┘
```

## Quickstart

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run the inference demo
uv run python main.py
```

## Scripts

### `main.py` — Full Inference Demo

Demonstrates the complete LLM inference pipeline:

- Create a small version of Qwen3 with random initialization
- Processes 60 chat prompts (2 base prompts × 30 repetitions to demonstrate prefix caching)
- Processes them through the custom LLM engine with batch processing
- Uses paged attention and KV cache management for efficient inference
- Generates up to 256 tokens per prompt with temperature sampling

```bash
uv run python main.py
```

**What you'll see**: The engine processes prompts in batches, with repeated prompts benefiting from prefix cache hits. Watch GPU memory usage stay stable despite varying sequence lengths.

---

### `benchmark_prefilling.py` — Prefill Attention Comparison

Compares three attention implementations during the **prefill phase** (processing input prompts):

| Implementation | Memory Complexity | Description |
|----------------|-------------------|-------------|
| PyTorch Standard | O(N²) | Traditional attention that materializes the full N×N attention matrix |
| Naive Triton | O(N²) | Custom GPU kernel with shared memory, limited to sequences ≤128 tokens |
| Flash Attention | O(N) | Block-wise processing with online softmax—no full matrix materialization |

```bash
uv run python benchmark_prefilling.py
```

**Why it matters**: Flash attention enables processing much longer sequences without running out of GPU memory. A 4K token sequence with O(N²) attention needs 16M attention scores; flash attention needs only a few blocks.

---

### `benchmark_decoding.py` — Decode Attention Comparison

Compares three implementations during the **decode phase** (generating tokens one at a time):

| Implementation | Description |
|----------------|-------------|
| Naive PyTorch | Loop-based implementation iterating over paged KV cache blocks |
| Optimized PyTorch | Vectorized gather operations with batch masking |
| Triton Kernel | Custom GPU kernel optimized for paged memory access patterns |

```bash
uv run python benchmark_decoding.py
```

**Why it matters**: Decode is memory-bound (each new token reads the entire KV cache). The Triton kernel minimizes memory transactions by coalescing accesses across the paged block structure.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        LLMEngine                             │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐   │
│  │  Scheduler  │───▶│ ModelRunner  │───▶│ Qwen3Model    │   │
│  │             │    │              │    │               │   │
│  │ • waiting   │    │ • prefill()  │    │ • Attention   │   │
│  │ • running   │    │ • decode()   │    │ • MLP         │   │
│  │ • preempt   │    │ • CUDA graph │    │ • RoPE        │   │
│  └─────────────┘    └──────────────┘    └───────────────┘   │
│         │                  │                                 │
│         ▼                  ▼                                 │
│  ┌─────────────┐    ┌──────────────┐                        │
│  │BlockManager │    │  KV Cache    │                        │
│  │             │    │              │                        │
│  │ • allocate  │    │ (num_blocks, │                        │
│  │ • free      │    │  block_size, │                        │
│  │ • prefix    │    │  num_heads,  │                        │
│  │   caching   │    │  head_dim)   │                        │
│  └─────────────┘    └──────────────┘                        │
└──────────────────────────────────────────────────────────────┘
```

## Project Structure

```
Mini-vLLM/
├── src/myvllm/
│   ├── models/
│   │   └── qwen3.py             # Qwen3 transformer with GQA support
│   │
│   ├── engine/
│   │   ├── llm_engine.py        # Top-level API: generate(), manages workers
│   │   ├── scheduler.py         # Request batching, prefill/decode scheduling
│   │   ├── block_manager.py     # Paged memory allocation, prefix caching
│   │   ├── model_runner.py      # GPU execution, CUDA graph capture
│   │   └── sequence.py          # Request state: tokens, status, block table
│   │
│   ├── layers/
│   │   ├── attention.py         # Flash attention (prefill) + paged attention (decode)
│   │   ├── linear.py            # Column/Row parallel linear for tensor parallelism
│   │   ├── embedding_head.py    # Vocab-parallel embedding and LM head
│   │   ├── rotary_embedding.py  # RoPE positional embeddings
│   │   ├── layernorm.py         # RMSNorm
│   │   ├── activation.py        # SiLU gating
│   │   └── sampler.py           # Temperature sampling
│   │
│   ├── utils/
│   │   ├── context.py           # Global state: prefill vs decode mode
│   │   └── loader.py            # Weight loading utilities
│   │
│   └── sampling_parameters.py   # SamplingParams dataclass
│
├── main.py                      # Inference demo
├── benchmark_prefilling.py      # Prefill attention benchmarks
├── benchmark_decoding.py        # Decode attention benchmarks
└── HowToApproachvLLM.md         # Step-by-step implementation guide
```

## Configuration

Key parameters in `main.py`:

```python
config = {
    'max_num_sequences': 16,         # Max concurrent sequences in a batch
    'max_num_batched_tokens': 1024,  # Max tokens per forward pass
    'block_size': 256,               # Tokens per KV cache block
    'gpu_memory_utilization': 0.9,   # Fraction of GPU memory for KV cache
    'world_size': 1,                 # Number of GPUs (tensor parallelism)
    'enforce_eager': True,           # Disable CUDA graphs for debugging
}
```

## Requirements

- **Python**: 3.11 (specifically `>=3.11, <3.12`)
- **Hardware**: CUDA-capable GPU
- **Dependencies**: Managed by uv
  - `torch` — Core tensor operations
  - `transformers` — Tokenizer utilities
  - `xxhash` — Fast hashing for prefix caching

## Acknowledgments

This project builds on ideas from:
- [vLLM](https://github.com/vllm-project/vllm) — The production paged attention system
- [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) — Minimal vLLM implementation
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) — Memory-efficient attention algorithm
