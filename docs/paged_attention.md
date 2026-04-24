# PagedAttention & Continuous Batching — How vLLM Works Under Load

This document explains the two core mechanisms that the traffic generator is designed to stress.

---

## Why Memory Management Matters for LLM Serving

Every transformer layer computes attention over all previous tokens in a sequence. The intermediate results — the **Key** and **Value** vectors for each token — are cached to avoid recomputation on subsequent decode steps. This is the **KV cache**.

For a sequence of N tokens with H attention heads of dimension D, the KV cache size grows as:

```
KV cache size = 2 × N × H × D × bytes_per_element × num_layers
```

For Llama 3 8B (32 layers, 32 heads, head_dim=128, fp16):
- 1K context: ~512 MB
- 4K context: ~2 GB
- 8K context: ~4 GB

This is why memory management is the central bottleneck for serving many users concurrently.

---

## PagedAttention

Before PagedAttention, serving engines allocated a contiguous block of GPU memory per sequence equal to `max_seq_len × kv_size`. This caused two problems:

1. **Internal fragmentation** — a 512-token response allocated for 4096 tokens wastes 87% of its reservation
2. **External fragmentation** — gaps between fixed allocations become unusable over time

PagedAttention borrows the OS virtual memory model:

- KV cache is divided into fixed-size **blocks** (e.g. 16 tokens per block)
- Each sequence has a **block table** mapping logical block indices to physical block addresses
- Physical blocks are allocated from a **free pool on demand**, one block at a time
- When a sequence finishes, its blocks are returned to the pool immediately

```
Logical view (sequence):    [ block 0 | block 1 | block 2 | ... ]
                                  ↓         ↓         ↓
Physical memory (pool):    [ addr:0x4A | addr:0x1F | addr:0x8C | ... ]
```

Blocks for a sequence don't need to be contiguous in physical memory. The attention kernel uses the block table to gather them at compute time.

**Result:** Near-zero memory waste. vLLM achieves <4% KV cache waste vs 60–80% in prior systems.

### What the traffic generator does to PagedAttention

- Many concurrent short + long requests force the block allocator to rapidly allocate and free blocks
- When the free pool runs low, vLLM **preempts** sequences: evicts their blocks (either swaps to CPU or recomputes on re-schedule)
- The `burst` scenario is designed to trigger this — quiet baseline fills the pool normally, then a spike of new requests fights for the remaining blocks

You'll see preemption events as **p99 TTFT spikes** in the results — a request that was mid-decode gets paused and re-queued.

---

## Continuous Batching

Traditional static batching:
```
[req A ──────────────────] wait...
                           [req B ──────]
                                        [req C ──────────────]
GPU utilization: ████░░░░████░░░░████░░░░  (lots of idle gaps)
```

Continuous batching (iteration-level scheduling):
```
iteration 1: [A tok1] [B tok1] [C tok1]
iteration 2: [A tok2] [B tok2] [C tok2]
iteration 3: [A tok3] [B done] [D tok1]  ← B finished, D immediately fills its slot
iteration 4: [A tok4] [D tok2] [E tok1]  ← A still running, new arrivals slot in
GPU utilization: ████████████████████████  (near-continuous)
```

Every decode step, the scheduler:
1. Checks which sequences are running, waiting, or preempted
2. Allocates new KV blocks for running sequences
3. If blocks are unavailable, preempts lowest-priority sequences
4. Adds newly arrived requests to fill freed capacity
5. Flattens all active sequences into a single forward pass ("super sequence")

**Result:** Completed sequences are immediately replaced by new arrivals, keeping GPU utilization high regardless of variable output lengths.

### What the traffic generator does to continuous batching

- The **persona system** creates sequences with wildly different lengths: a "short" user generating 10 tokens finishes in 1–2 iterations; a "long" user generating 256 tokens runs for many iterations
- This heterogeneity is exactly what makes continuous batching valuable — the scheduler can fill the short user's vacated slot immediately with a new arrival
- With only 1 user (your Mac doing one chat at a time), continuous batching does nothing — you need concurrent users to see its effect

---

## Apple Silicon Specifics

vllm-metal uses **MLX** as the compute backend instead of CUDA. Key differences:

| | CUDA (NVIDIA) | Metal / MLX (Apple Silicon) |
|--|--|--|
| Memory | Separate GPU VRAM | Unified memory (shared CPU/GPU) |
| KV cache copy overhead | CPU→GPU memcpy for swaps | Near-zero (same physical memory) |
| Tensor ops | cuBLAS / FlashAttention | MLX Metal kernels |
| Throughput (8B model) | ~2000 tok/s (H100) | ~30–50 tok/s (M4) |

The unified memory architecture means vLLM's swap-based preemption (evicting KV blocks to CPU) is essentially free on Apple Silicon — there's no PCIe bus to cross. This makes preemption less painful here than on discrete GPU setups.
