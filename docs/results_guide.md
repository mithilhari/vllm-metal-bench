# Reading Traffic Generator Results

## Output Metrics

| Metric | What it measures | Apple Silicon M4 baseline |
|--------|-----------------|--------------------------|
| `throughput_req_s` | Requests completed per second | 0.1–0.5 req/s |
| `total_tokens_generated` | Total output tokens across all requests | — |
| `ttft_ms mean` | Average time to first token | 500–2000ms |
| `ttft_ms p99` | Worst-case time to first token | spikes = preemption |
| `e2e_latency_ms mean` | Average full request time | 5–30s at 150 max_tokens |
| `e2e_latency_ms p99` | Tail latency | 2–5× mean is normal |

## What to Look For

**TTFT p99 spike** — if p99 TTFT is 3–5× the mean, vLLM is preempting sequences under KV cache pressure. This is the PagedAttention block eviction mechanism working. More users = more preemption on 16 GB.

**Throughput plateau** — as you increase `--users`, throughput will stop growing at some point. That's your KV cache saturation point for this model + context length combination.

**E2E latency growing with users** — normal. The continuous batcher keeps GPU busy, but each individual request waits longer in the queue as concurrency increases. This is the throughput/latency tradeoff.

## Suggested Experiments

```bash
# Baseline — 1 user, measure solo performance
python traffic/vllm_traffic_gen.py --model meta-llama/Meta-Llama-3-8B-Instruct \
  --users 1 --duration 60 --max-tokens 150 --output-json results/baseline_1user.json

# Scale up — find the saturation point
for N in 2 4 6 8 10; do
  python traffic/vllm_traffic_gen.py --model meta-llama/Meta-Llama-3-8B-Instruct \
    --users $N --duration 60 --max-tokens 150 \
    --output-json results/steady_${N}users.json
done

# Burst — trigger preemption, watch p99 TTFT
python traffic/vllm_traffic_gen.py --model meta-llama/Meta-Llama-3-8B-Instruct \
  --scenario burst --users 10 --max-tokens 150 --output-json results/burst.json

# Long context — stress KV cache more per sequence
python traffic/vllm_traffic_gen.py --model meta-llama/Meta-Llama-3-8B-Instruct \
  --users 4 --max-tokens 512 --duration 90 --output-json results/long_ctx.json
```
