#!/usr/bin/env python3
"""
vLLM Traffic Generator
======================
Simulates multiple concurrent "users" hitting a vLLM serve endpoint,
designed to excite PagedAttention and continuous batching internals.

Usage:
  # Start vLLM first:
  #   vllm serve <model> --host 0.0.0.0 --port 8000

  python vllm_traffic_gen.py --model <model-name>
  python vllm_traffic_gen.py --model llama3 --users 20 --duration 120
  python vllm_traffic_gen.py --model llama3 --scenario burst
"""

import asyncio
import argparse
import json
import os
import random
import time
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import statistics

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp not found. Run: pip install aiohttp")
    sys.exit(1)

# ─────────────────────────────────────────────
# Prompt Library — varied lengths to stress the
# scheduler: short decode-heavy, long prefill-
# heavy, and mixed conversational.
# ─────────────────────────────────────────────

PROMPT_CATALOG = {
    "short": [
        "What is 42?",
        "Say hello.",
        "Name a color.",
        "What day is it?",
        "Define entropy.",
        "Recommend a book.",
        "What is Python?",
        "Tell me a joke.",
        "What is the capital of France?",
        "Name three planets.",
    ],
    "medium": [
        "Explain how TCP/IP works in 3 sentences.",
        "What are the key differences between SQL and NoSQL databases?",
        "Write a haiku about machine learning.",
        "Summarize the plot of Romeo and Juliet.",
        "How does a transformer neural network work at a high level?",
        "Compare Python and Rust for systems programming.",
        "What are the causes of the French Revolution?",
        "Describe the water cycle in detail.",
        "How does HTTPS encryption work?",
        "What is the difference between RAM and storage?",
    ],
    "long": [
        (
            "You are a senior software engineer. I need you to write a detailed technical design document "
            "for a distributed rate-limiting system that supports 10M requests per second across 50 "
            "data centers globally. Cover: architecture overview, data store choices, consistency model, "
            "failover strategy, and monitoring approach. Be thorough."
        ),
        (
            "Explain the entire history of computing from Charles Babbage's Analytical Engine through "
            "to modern GPUs and neural network accelerators. Cover key milestones, inventors, "
            "technological leaps, and how each era enabled the next. Include vacuum tubes, transistors, "
            "integrated circuits, microprocessors, and beyond."
        ),
        (
            "Write a comprehensive guide to Kubernetes for a developer who knows Docker but has never "
            "used k8s. Cover: pods, deployments, services, ingress, configmaps, secrets, persistent "
            "volumes, namespaces, RBAC, and Helm charts. Include concrete YAML examples for each concept."
        ),
        (
            "I'm building a high-frequency trading system in C++. Analyze the following design decisions "
            "and give detailed recommendations: lock-free queues vs mutexes, kernel bypass networking "
            "(DPDK/RDMA), CPU pinning and NUMA awareness, branch prediction optimization, and latency "
            "measurement methodology. Explain tradeoffs for each."
        ),
        (
            "Write a detailed essay on the philosophy of consciousness. Discuss: Descartes' mind-body "
            "dualism, Dennett's multiple drafts model, Chalmers' hard problem of consciousness, "
            "integrated information theory (Tononi), global workspace theory, and the Chinese Room "
            "argument. Compare and contrast these views and give your own assessment."
        ),
    ],
    "conversational": [
        "Can you help me debug a memory leak in Python?",
        "I'm trying to learn Rust. Where should I start?",
        "What's the best way to structure a REST API?",
        "I need to optimize a slow SQL query. What should I look for?",
        "Help me understand async/await in JavaScript.",
        "What's the difference between a process and a thread?",
        "I want to build a chatbot. What frameworks do you recommend?",
        "Explain gradient descent like I'm 12.",
        "How do I deploy a model to production safely?",
        "What's your recommendation for a CI/CD pipeline?",
    ],
}

ALL_PROMPTS = [p for prompts in PROMPT_CATALOG.values() for p in prompts]


class Scenario(Enum):
    STEADY   = "steady"    # constant arrival rate
    BURST    = "burst"     # sudden spike, then quiet
    RAMP     = "ramp"      # gradually increasing load
    WAVE     = "wave"      # sinusoidal traffic pattern
    RANDOM   = "random"    # Poisson arrivals (most realistic)


@dataclass
class RequestResult:
    user_id: int
    prompt_type: str
    prompt_len: int
    status: str             # "ok" | "error" | "timeout"
    ttft_ms: float = 0.0    # time to first token
    total_ms: float = 0.0   # end-to-end latency
    tokens_generated: int = 0
    error: str = ""


@dataclass
class Stats:
    results: list = field(default_factory=list)

    def add(self, r: RequestResult):
        self.results.append(r)

    def summary(self) -> dict:
        ok = [r for r in self.results if r.status == "ok"]
        errors = [r for r in self.results if r.status == "error"]
        timeouts = [r for r in self.results if r.status == "timeout"]

        if not ok:
            return {"error": "no successful requests"}

        ttfts = [r.ttft_ms for r in ok if r.ttft_ms > 0]
        totals = [r.total_ms for r in ok]
        tokens = [r.tokens_generated for r in ok]

        total_time = max(totals) / 1000 if totals else 1
        throughput = len(ok) / total_time if total_time > 0 else 0

        return {
            "total_requests": len(self.results),
            "successful": len(ok),
            "errors": len(errors),
            "timeouts": len(timeouts),
            "throughput_req_s": round(throughput, 2),
            "total_tokens_generated": sum(tokens),
            "ttft_ms": {
                "mean":   round(statistics.mean(ttfts), 1)   if ttfts else 0,
                "median": round(statistics.median(ttfts), 1) if ttfts else 0,
                "p95":    round(sorted(ttfts)[int(len(ttfts)*0.95)], 1) if len(ttfts) > 1 else 0,
                "p99":    round(sorted(ttfts)[int(len(ttfts)*0.99)], 1) if len(ttfts) > 1 else 0,
            },
            "e2e_latency_ms": {
                "mean":   round(statistics.mean(totals), 1),
                "median": round(statistics.median(totals), 1),
                "p95":    round(sorted(totals)[int(len(totals)*0.95)], 1) if len(totals) > 1 else 0,
                "p99":    round(sorted(totals)[int(len(totals)*0.99)], 1) if len(totals) > 1 else 0,
            },
        }


# ─────────────────────────────────────────────
# Core request function — streaming OpenAI-compat
# ─────────────────────────────────────────────

async def send_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    user_id: int,
    prompt: str,
    prompt_type: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    use_chat_api: bool,
    verbose: bool,
) -> RequestResult:

    if use_chat_api:
        endpoint = f"{base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
    else:
        endpoint = f"{base_url}/completions"
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

    t_start = time.perf_counter()
    ttft = 0.0
    tokens = 0
    first_token = False

    try:
        async with session.post(
            endpoint,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return RequestResult(
                    user_id=user_id,
                    prompt_type=prompt_type,
                    prompt_len=len(prompt),
                    status="error",
                    error=f"HTTP {resp.status}: {body[:200]}",
                )

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if not first_token:
                    ttft = (time.perf_counter() - t_start) * 1000
                    first_token = True

                # Count tokens from delta
                if use_chat_api:
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                else:
                    content = chunk.get("choices", [{}])[0].get("text", "")
                if content:
                    tokens += 1  # approx 1 token per chunk in streaming

        total_ms = (time.perf_counter() - t_start) * 1000
        if verbose:
            print(f"  [user-{user_id:03d}] {prompt_type:15s} | "
                  f"TTFT={ttft:6.0f}ms | E2E={total_ms:7.0f}ms | "
                  f"~{tokens} tokens")
        return RequestResult(
            user_id=user_id,
            prompt_type=prompt_type,
            prompt_len=len(prompt),
            status="ok",
            ttft_ms=ttft,
            total_ms=total_ms,
            tokens_generated=tokens,
        )

    except asyncio.TimeoutError:
        return RequestResult(
            user_id=user_id, prompt_type=prompt_type,
            prompt_len=len(prompt), status="timeout",
            error=f"Timed out after {timeout}s",
        )
    except Exception as e:
        return RequestResult(
            user_id=user_id, prompt_type=prompt_type,
            prompt_len=len(prompt), status="error", error=str(e),
        )


# ─────────────────────────────────────────────
# Task cancellation helper — hard-stops in-flight
# requests so scenarios exit on time
# ─────────────────────────────────────────────

async def _cancel_and_drain(tasks: list):
    """Cancel all tasks and give them a moment to clean up."""
    for t in tasks:
        if not t.done():
            t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


# ─────────────────────────────────────────────
# User simulation — each user loops sending
# requests with a think-time between them
# ─────────────────────────────────────────────

async def simulate_user(
    user_id: int,
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    stats: Stats,
    stop_event: asyncio.Event,
    args: argparse.Namespace,
    initial_delay: float = 0.0,
):
    await asyncio.sleep(initial_delay)

    # Each user has a "persona" — bias toward certain prompt types
    personas = ["short", "medium", "long", "conversational"]
    persona = personas[user_id % len(personas)]

    while not stop_event.is_set():
        # Mix: 70% persona type, 30% random from all
        if random.random() < 0.7:
            pool = PROMPT_CATALOG.get(persona, ALL_PROMPTS)
        else:
            pool = ALL_PROMPTS

        prompt = random.choice(pool)

        # Determine prompt type for labeling
        for ptype, plist in PROMPT_CATALOG.items():
            if prompt in plist:
                prompt_type = ptype
                break
        else:
            prompt_type = "mixed"

        # Vary max_tokens by prompt type — capped by --max-tokens arg
        cap = args.max_tokens
        if prompt_type == "short":
            max_tokens = min(cap, random.randint(10, 64))
        elif prompt_type == "medium":
            max_tokens = min(cap, random.randint(64, 128))
        elif prompt_type == "long":
            max_tokens = min(cap, random.randint(128, 256))
        else:
            max_tokens = min(cap, random.randint(50, 128))

        temperature = random.uniform(0.0, 1.0)

        result = await send_request(
            session=session,
            base_url=base_url,
            model=model,
            user_id=user_id,
            prompt=prompt,
            prompt_type=prompt_type,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=args.timeout,
            use_chat_api=not args.completion_api,
            verbose=args.verbose,
        )
        stats.add(result)

        # Think time between requests (simulates real user behavior)
        think_time = random.expovariate(1.0 / args.think_time)
        think_time = max(0.1, min(think_time, args.think_time * 5))
        await asyncio.sleep(think_time)


# ─────────────────────────────────────────────
# Session factory — injects auth header once
# ─────────────────────────────────────────────

def _make_session(args, multiplier: int = 2) -> aiohttp.ClientSession:
    connector = aiohttp.TCPConnector(limit=args.users * multiplier)
    headers = {}
    if getattr(args, "api_key", None):
        headers["Authorization"] = f"Bearer {args.api_key}"
    return aiohttp.ClientSession(connector=connector, headers=headers)


# ─────────────────────────────────────────────
# Scenario coordinators
# ─────────────────────────────────────────────

async def run_steady(args, stats, stop_event):
    """Constant number of concurrent users throughout."""
    async with _make_session(args, multiplier=2) as session:
        tasks = []
        for uid in range(args.users):
            delay = uid * (1.0 / args.users)  # stagger startup
            t = asyncio.create_task(
                simulate_user(uid, session, args.base_url, args.model,
                              stats, stop_event, args, initial_delay=delay)
            )
            tasks.append(t)

        await asyncio.sleep(args.duration)
        stop_event.set()
        await _cancel_and_drain(tasks)


async def run_burst(args, stats, stop_event):
    """Quiet → sudden spike → quiet. Tests preemption and KV cache pressure."""
    async with _make_session(args, multiplier=3) as session:
        print(f"[burst] Phase 1: baseline ({args.users//4} users, 20s)")
        tasks = []
        phase1_stop = asyncio.Event()
        for uid in range(max(1, args.users // 4)):
            t = asyncio.create_task(
                simulate_user(uid, session, args.base_url, args.model,
                              stats, phase1_stop, args)
            )
            tasks.append(t)
        await asyncio.sleep(20)
        phase1_stop.set()
        await _cancel_and_drain(tasks)

        print(f"[burst] Phase 2: BURST ({args.users} users, 30s)")
        tasks = []
        phase2_stop = asyncio.Event()
        for uid in range(args.users):
            t = asyncio.create_task(
                simulate_user(uid, session, args.base_url, args.model,
                              stats, phase2_stop, args, initial_delay=uid*0.1)
            )
            tasks.append(t)
        await asyncio.sleep(30)
        phase2_stop.set()
        await _cancel_and_drain(tasks)

        print(f"[burst] Phase 3: cooldown ({args.users//4} users, 20s)")
        tasks = []
        phase3_stop = asyncio.Event()
        for uid in range(max(1, args.users // 4)):
            t = asyncio.create_task(
                simulate_user(uid, session, args.base_url, args.model,
                              stats, phase3_stop, args)
            )
            tasks.append(t)
        await asyncio.sleep(20)
        stop_event.set()
        phase3_stop.set()
        await _cancel_and_drain(tasks)


async def run_ramp(args, stats, stop_event):
    """Gradually add users every N seconds. Tests continuous batching ramp-up."""
    async with _make_session(args, multiplier=2) as session:
        tasks = []
        step = max(1, args.users // 10)
        interval = args.duration / (args.users / step)

        for wave in range(0, args.users, step):
            if stop_event.is_set():
                break
            print(f"[ramp] Adding users {wave}–{wave+step-1} "
                  f"(total active: {wave+step})")
            for uid in range(wave, min(wave + step, args.users)):
                t = asyncio.create_task(
                    simulate_user(uid, session, args.base_url, args.model,
                                  stats, stop_event, args)
                )
                tasks.append(t)
            await asyncio.sleep(interval)

        remaining = args.duration - (args.users // step) * interval
        if remaining > 0:
            await asyncio.sleep(remaining)
        stop_event.set()
        await _cancel_and_drain(tasks)


async def run_wave(args, stats, stop_event):
    """Sinusoidal load — continuously varies concurrency. Most interesting pattern."""
    import math
    async with _make_session(args, multiplier=2) as session:
        active_tasks = {}
        uid_counter = 0
        period = 30.0  # seconds per wave cycle
        t_start = time.time()

        while not stop_event.is_set():
            elapsed = time.time() - t_start
            if elapsed >= args.duration:
                stop_event.set()
                break

            # Sinusoidal: ranges from 10% to 100% of max users
            frac = 0.55 + 0.45 * math.sin(2 * math.pi * elapsed / period)
            target = max(1, int(frac * args.users))

            current = len([t for t in active_tasks.values() if not t.done()])

            # Spawn new users if below target
            while current < target:
                uid = uid_counter % args.users
                uid_counter += 1
                user_stop = asyncio.Event()
                t = asyncio.create_task(
                    simulate_user(uid, session, args.base_url, args.model,
                                  stats, user_stop, args)
                )
                active_tasks[uid_counter] = t
                current += 1

            # Trim done tasks
            active_tasks = {k: v for k, v in active_tasks.items() if not v.done()}

            await asyncio.sleep(1.0)

        # Cancel all in-flight tasks — don't wait for long requests to drain
        for t in active_tasks.values():
            if not t.done():
                t.cancel()
        # Give cancellations a moment to propagate, then move on
        await asyncio.gather(*active_tasks.values(), return_exceptions=True)


# ─────────────────────────────────────────────
# Progress printer
# ─────────────────────────────────────────────

async def print_progress(stats: Stats, stop_event: asyncio.Event, interval: float = 10.0):
    t0 = time.time()
    while not stop_event.is_set():
        await asyncio.sleep(interval)
        elapsed = time.time() - t0
        ok = sum(1 for r in stats.results if r.status == "ok")
        err = sum(1 for r in stats.results if r.status != "ok")
        rps = ok / elapsed if elapsed > 0 else 0
        print(f"\n[{elapsed:5.0f}s] requests: {ok} ok / {err} err | {rps:.2f} req/s")


# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────

async def health_check(base_url: str, model: str, api_key: str = None) -> bool:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(
                f"{base_url}/models",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [m["id"] for m in data.get("data", [])]
                    print(f"✓ Server healthy. Available models: {models}")
                    if model and not any(model in m for m in models):
                        print(f"  ⚠ Warning: '{model}' not found. Using anyway.")
                    return True
                else:
                    body = await resp.text()
                    print(f"✗ Health check failed: HTTP {resp.status} — {body[:200]}")
                    return False
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        print(f"  Is the server running at {base_url}?")
        return False


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="vLLM Traffic Generator — stresses PagedAttention + continuous batching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic: 10 users for 60 seconds
  python vllm_traffic_gen.py --model llama3

  # 20 users, burst scenario, 2s think time
  python vllm_traffic_gen.py --model llama3 --users 20 --scenario burst --think-time 2

  # Ramp scenario, verbose output, completions API
  python vllm_traffic_gen.py --model llama3 --scenario ramp --verbose --completion-api

  # Long run against remote server
  python vllm_traffic_gen.py --model llama3 --base-url http://192.168.1.100:8000 --duration 300

Scenarios:
  steady   — fixed concurrency throughout (default)
  burst    — quiet → spike → quiet (stresses preemption)
  ramp     — gradually add users (tests scheduler scaling)
  wave     — sinusoidal concurrency (most realistic mix)
  random   — Poisson arrivals
        """
    )
    parser.add_argument("--model",        required=True, help="Model name (as served by vLLM)")
    parser.add_argument("--base-url",     default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--users",        type=int, default=10, help="Max concurrent users (default: 10)")
    parser.add_argument("--duration",     type=int, default=60,  help="Test duration in seconds (default: 60)")
    parser.add_argument("--think-time",   type=float, default=3.0, help="Mean think time between requests (default: 3s)")
    parser.add_argument("--timeout",      type=float, default=120.0, help="Request timeout in seconds (default: 120)")
    parser.add_argument("--scenario",     choices=[s.value for s in Scenario],
                        default=Scenario.STEADY.value, help="Traffic pattern (default: steady)")
    parser.add_argument("--max-tokens",      type=int, default=200, help="Max tokens per request (default: 200, lower = faster on Apple Silicon)")
    parser.add_argument("--completion-api", action="store_true",
                        help="Use /v1/completions instead of /v1/chat/completions")
    parser.add_argument("--verbose",      action="store_true", help="Print each request result")
    parser.add_argument("--no-health-check", action="store_true", help="Skip server health check")
    parser.add_argument("--output-json", help="Save results to JSON file")
    parser.add_argument("--api-key", default=None,
                        help="Bearer token. Falls back to PARASAIL_API_KEY or VLLM_API_KEY env vars.")
    return parser.parse_args()


async def main():
    args = parse_args()

    print("=" * 60)
    print("  vLLM Traffic Generator")
    print("=" * 60)
    print(f"  Server:    {args.base_url}")
    print(f"  Model:     {args.model}")
    print(f"  Users:     {args.users}")
    print(f"  Duration:  {args.duration}s")
    print(f"  Scenario:  {args.scenario}")
    print(f"  API:       {'completions' if args.completion_api else 'chat/completions'}")

    # Resolve API key: CLI flag → PARASAIL_API_KEY → VLLM_API_KEY → None (no auth)
    api_key = (
        args.api_key
        or os.environ.get("PARASAIL_API_KEY")
        or os.environ.get("VLLM_API_KEY")
    )
    args.api_key = api_key
    print(f"  Auth:      {'Bearer token ✓' if api_key else 'None (local)'}")
    print("=" * 60)

    if not args.no_health_check:
        ok = await health_check(args.base_url, args.model, api_key)
        if not ok:
            sys.exit(1)

    stats = Stats()
    stop_event = asyncio.Event()

    # Progress reporter
    progress_task = asyncio.create_task(
        print_progress(stats, stop_event)
    )

    t_start = time.time()
    scenario = Scenario(args.scenario)

    print(f"\n▶ Starting '{scenario.value}' scenario...")

    if scenario == Scenario.STEADY:
        await run_steady(args, stats, stop_event)
    elif scenario == Scenario.BURST:
        await run_burst(args, stats, stop_event)
    elif scenario == Scenario.RAMP:
        await run_ramp(args, stats, stop_event)
    elif scenario == Scenario.WAVE:
        await run_wave(args, stats, stop_event)
    elif scenario == Scenario.RANDOM:
        # Poisson arrivals: same as steady but with exponential inter-arrival gaps
        args.think_time = args.think_time * 0.5
        await run_steady(args, stats, stop_event)

    stop_event.set()
    await progress_task

    # ── Final Report ──────────────────────────────
    elapsed = time.time() - t_start
    summary = stats.summary()

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Duration:            {elapsed:.1f}s")
    print(f"  Total requests:      {summary.get('total_requests', 0)}")
    print(f"  Successful:          {summary.get('successful', 0)}")
    print(f"  Errors:              {summary.get('errors', 0)}")
    print(f"  Timeouts:            {summary.get('timeouts', 0)}")
    print(f"  Throughput:          {summary.get('throughput_req_s', 0)} req/s")
    print(f"  Tokens generated:    {summary.get('total_tokens_generated', 0)}")

    ttft = summary.get("ttft_ms", {})
    if ttft:
        print(f"\n  Time to First Token (ms):")
        print(f"    mean={ttft.get('mean',0)}  median={ttft.get('median',0)}"
              f"  p95={ttft.get('p95',0)}  p99={ttft.get('p99',0)}")

    e2e = summary.get("e2e_latency_ms", {})
    if e2e:
        print(f"\n  End-to-End Latency (ms):")
        print(f"    mean={e2e.get('mean',0)}  median={e2e.get('median',0)}"
              f"  p95={e2e.get('p95',0)}  p99={e2e.get('p99',0)}")

    print("=" * 60)

    if args.output_json:
        output = {
            "config": vars(args),
            "elapsed_s": round(elapsed, 2),
            "summary": summary,
            "requests": [
                {
                    "user_id": r.user_id,
                    "prompt_type": r.prompt_type,
                    "prompt_len": r.prompt_len,
                    "status": r.status,
                    "ttft_ms": round(r.ttft_ms, 1),
                    "total_ms": round(r.total_ms, 1),
                    "tokens_generated": r.tokens_generated,
                    "error": r.error,
                }
                for r in stats.results
            ],
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to: {args.output_json}")


if __name__ == "__main__":
    asyncio.run(main())
