# vllm-metal-bench

> Run vLLM on Apple Silicon (M-series), serve a HuggingFace LLM, stress-test it with a multi-user traffic generator, and visualize results — locally on your Mac or against a GPU cloud endpoint on Parasail.

---

## What This Is

A complete end-to-end workflow for benchmarking vLLM inference performance:

1. **Install vllm-metal** — the official vLLM plugin for Apple Silicon (M1–M4)
2. **Download model weights** from HuggingFace (`meta-llama/Meta-Llama-3-8B-Instruct`)
3. **Serve the model** via `vllm serve` with an OpenAI-compatible API on `localhost:8000`
4. **Run the traffic generator** — simulates concurrent users with 4 traffic patterns to stress PagedAttention and continuous batching
5. **Deploy to Parasail** — point the same traffic generator at an H100/A100 GPU endpoint
6. **Visualize results** — interactive HTML dashboard rendering throughput, TTFT, and latency charts

---

## Repository Structure

```
vllm-metal-bench/
├── README.md
├── .gitignore
├── scripts/
│   ├── 1_install_vllm_metal.sh       # installs vllm-metal into ~/.venv-vllm-metal
│   ├── 2_download_model.sh           # downloads weights from HuggingFace
│   ├── 3_serve.sh                    # starts vllm serve on localhost:8000
│   └── run_all_scenarios.sh          # runs all 4 traffic scenarios in sequence
├── traffic/
│   └── vllm_traffic_gen.py           # multi-user traffic generator
├── parasail/
│   └── deploy_parasail.py            # deploys model to Parasail via API
├── dashboard/
│   └── index.html                    # standalone results visualization dashboard
├── docs/
│   ├── paged_attention.md            # PagedAttention and continuous batching internals
│   └── results_guide.md             # how to interpret results + experiments
└── results/
    └── .gitkeep                      # save your .json results here (gitignored)
```

---

## Part 1 — Local Setup (Apple Silicon)

### Requirements

- Apple Silicon Mac (M1–M4) — tested on **MacBook Pro M4**
- macOS arm64: verify with `uname -m`
- Python 3.10+ (3.12 recommended)
- 16 GB unified memory minimum; 32 GB recommended for 8B models
- HuggingFace account + access token (for gated models like Llama 3)

### Step 1 — Install vllm-metal

```bash
bash scripts/1_install_vllm_metal.sh
```

Installs the [official vllm-metal plugin](https://github.com/vllm-project/vllm-metal) into `~/.venv-vllm-metal`. Takes 5–15 minutes.

### Step 2 — Download model weights

```bash
export HF_TOKEN=hf_your_token_here
bash scripts/2_download_model.sh
```

Downloads `meta-llama/Meta-Llama-3-8B-Instruct` (~16 GB) to `~/models/`.

### Step 3 — Serve the model

```bash
bash scripts/3_serve.sh
```

Starts `vllm serve` on `http://localhost:8000`. Wait for `Application startup complete`.

### Step 4 — Run traffic scenarios

```bash
source ~/.venv-vllm-metal/bin/activate
pip install aiohttp

# Run all four scenarios
bash scripts/run_all_scenarios.sh

# Or run one at a time
python traffic/vllm_traffic_gen.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --scenario wave \
  --users 10 \
  --max-tokens 150 \
  --output-json results/local_wave.json
```

---

## Part 2 — Parasail GPU Cloud

[Parasail](https://parasail.io) runs vLLM on H100/A100 GPUs with a managed OpenAI-compatible API. The traffic generator works against it with only a `--base-url` change.

### Setup

```bash
echo 'export PARASAIL_API_KEY=your_key_here' >> ~/.vllm-metal-env
echo 'export HF_TOKEN=your_hf_token' >> ~/.vllm-metal-env
source ~/.vllm-metal-env
```

Get your Parasail API key at `saas.parasail.io` → Profile → API Keys.

### Deploy a model

```bash
python parasail/deploy_parasail.py
```

Or use the Parasail UI → Dedicated tab → enter HuggingFace model ID → select GPU → Deploy.

The script outputs the model ID when ONLINE, e.g. `xxxxxxxx-vllm-generative-dedicated`.

### Run traffic against Parasail

```bash
ENDPOINT=https://api.parasail.io/v1 \
MODEL=xxxxxxxx-vllm-generative-dedicated \
bash scripts/run_all_scenarios.sh
```

### Pause when done (stops billing)

```bash
curl -X POST https://api.parasail.io/api/v1/dedicated/deployments/YOUR_ID/pause \
  -H "Authorization: Bearer $PARASAIL_API_KEY"
```

---

## Traffic Generator Scenarios

| Scenario | What it does | Best for |
|----------|-------------|----------|
| `steady` | Fixed concurrency throughout | Baseline throughput |
| `burst` | Quiet → spike → quiet | KV cache preemption |
| `ramp` | Gradually adds users | Scheduler scale-up |
| `wave` | Sinusoidal concurrency | Realistic mixed load |

### Key flags

```
--model        Model name or Parasail deployment ID
--base-url     Endpoint URL (default: http://localhost:8000)
--users        Max concurrent users (default: 10)
--duration     Seconds to run (default: 60)
--max-tokens   Cap per request (default: 200 — lower is faster on Apple Silicon)
--scenario     steady | burst | ramp | wave | random
--api-key      Bearer token (auto-read from PARASAIL_API_KEY env var)
--output-json  Save results to JSON
```

---

## Results Dashboard

Open `dashboard/index.html` in any browser and drop your JSON result files onto it.

- Metric cards — throughput, tok/s, TTFT p50/p95/p99, E2E p50/p95/p99
- Throughput over time — completed requests per 10s window
- Latency percentiles — TTFT and E2E comparison across scenarios
- TTFT distribution — mean/median/p95/p99 tail analysis

---

## Apple Silicon M4 vs Parasail H100 — wave scenario results

| | M4 Mac (local) | Parasail H100 |
|--|--|--|
| Throughput | ~0.2 req/s | ~5 req/s |
| TTFT mean | ~1500 ms | ~300 ms |
| E2E mean | ~15,000 ms | ~900 ms |
| Tokens/s | ~40 | ~380 |

---

## References

- [vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal)
- [vllm-metal installation docs](https://docs.vllm.ai/projects/vllm-metal/en/latest/installation/)
- [Parasail docs](https://docs.parasail.io/parasail-docs)
- [PagedAttention paper — SOSP 2023](https://arxiv.org/abs/2309.06180)
