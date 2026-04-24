#!/usr/bin/env python3
"""
deploy_parasail.py
==================
Deploys a HuggingFace model as a vLLM dedicated endpoint on Parasail.
Polls until the endpoint is ONLINE, then prints the model ID to use
with vllm_traffic_gen.py.

Usage:
  export PARASAIL_API_KEY=your_key
  export HF_TOKEN=your_hf_token          # required for gated models
  python parasail/deploy_parasail.py

  # Override model or GPU:
  MODEL=meta-llama/Meta-Llama-3-8B-Instruct GPU=A100 python parasail/deploy_parasail.py
"""

import httpx
import os
import sys
import time

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME    = os.environ.get("MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
DESIRED_GPU   = os.environ.get("GPU", "H100SXM")
DESIRED_COUNT = int(os.environ.get("GPU_COUNT", "1"))
REPLICAS      = int(os.environ.get("REPLICAS", "1"))
DEPLOY_NAME   = os.environ.get("DEPLOY_NAME", "vllm-generative-dedicated")

# Scale down after 1 hour of inactivity — avoids surprise bills
SCALE_DOWN_POLICY    = "TIMER"
SCALE_DOWN_THRESHOLD = 3_600_000  # ms

CONTROL_URL = "https://api.parasail.io/api/v1"

# ── Auth ──────────────────────────────────────────────────────────────────────
api_key  = os.environ.get("PARASAIL_API_KEY")
hf_token = os.environ.get("HF_TOKEN", "")

if not api_key:
    print("ERROR: PARASAIL_API_KEY not set.")
    print("  export PARASAIL_API_KEY=your_key")
    sys.exit(1)

client = httpx.Client(
    base_url=CONTROL_URL,
    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    timeout=30,
)

print("=" * 56)
print("  Parasail Deployment")
print("=" * 56)
print(f"  Model:   {MODEL_NAME}")
print(f"  GPU:     {DESIRED_GPU} x{DESIRED_COUNT}")
print(f"  Name:    {DEPLOY_NAME}")
print("=" * 56)

# ── Step 1: Check model compatibility ────────────────────────────────────────
print("\n[1/4] Checking model compatibility...")
resp = client.get("/dedicated/support", params={
    "modelName": MODEL_NAME,
    "engine": "VLLM",
    "modelAccessKey": hf_token,
})
if resp.status_code != 200:
    print(f"ERROR: {resp.status_code} — {resp.text}")
    sys.exit(1)

data = resp.json()
if not data.get("supported"):
    print(f"ERROR: Model not supported — {data.get('errorMessage','unknown reason')}")
    sys.exit(1)
print("  ✓ Model is supported")

# ── Step 2: Get hardware options ──────────────────────────────────────────────
print("\n[2/4] Fetching available hardware...")
resp = client.get("/dedicated/devices", params={
    "engineName": "VLLM",
    "modelName": MODEL_NAME,
    "modelAccessKey": hf_token,
})
if resp.status_code != 200:
    print(f"ERROR: {resp.status_code} — {resp.text}")
    sys.exit(1)

devices = resp.json()
matched = False
for d in devices:
    is_match = (d.get("device") == DESIRED_GPU and d.get("count") == DESIRED_COUNT)
    d["selected"] = is_match
    if is_match:
        matched = True
        cost = d.get("cost", "?")
        print(f"  ✓ Selected: {DESIRED_GPU} x{DESIRED_COUNT}  ~${cost}/hr")

if not matched:
    available = [(d.get("device"), d.get("count")) for d in devices if d.get("available")]
    print(f"ERROR: {DESIRED_GPU} x{DESIRED_COUNT} not found or unavailable.")
    print(f"  Available options: {available}")
    print(f"  Set GPU= and GPU_COUNT= env vars to one of the above.")
    sys.exit(1)

# ── Step 3: Create deployment ─────────────────────────────────────────────────
print("\n[3/4] Creating deployment...")
payload = {
    "deploymentName": DEPLOY_NAME,
    "modelName": MODEL_NAME,
    "engine": "VLLM",
    "deviceConfigs": devices,
    "replicas": REPLICAS,
    "scaleDownPolicy": SCALE_DOWN_POLICY,
    "scaleDownThreshold": SCALE_DOWN_THRESHOLD,
}
if hf_token:
    payload["modelAccessKey"] = hf_token

resp = client.post("/dedicated/deployments", json=payload)
if resp.status_code != 200:
    print(f"ERROR: {resp.status_code} — {resp.text}")
    sys.exit(1)

deployment_id = resp.json()["id"]
print(f"  ✓ Deployment created: ID={deployment_id}")

# ── Step 4: Poll until ONLINE ─────────────────────────────────────────────────
print("\n[4/4] Waiting for endpoint to come ONLINE (this takes 5–15 min)...")
poll_client = httpx.Client(
    base_url=CONTROL_URL,
    headers={"Authorization": f"Bearer {api_key}"},
    timeout=10,
)

start = time.time()
while True:
    try:
        status_resp = poll_client.get(f"/dedicated/deployments/{deployment_id}")
        status = status_resp.json().get("status", "UNKNOWN")
        elapsed = int(time.time() - start)
        print(f"  [{elapsed:3d}s] status: {status}")

        if status == "ONLINE":
            model_id = status_resp.json().get("modelId", DEPLOY_NAME)
            break
        elif status in ("ERROR", "FAILED"):
            print(f"ERROR: Deployment failed with status {status}")
            sys.exit(1)
    except Exception as e:
        print(f"  (poll error: {e})")

    time.sleep(15)

print("\n" + "=" * 56)
print("  ✓ Deployment ONLINE")
print("=" * 56)
print(f"\nModel ID for traffic generator:\n  {model_id}\n")
print("Run the traffic generator:")
print(f"""
  python traffic/vllm_traffic_gen.py \\
    --model {model_id} \\
    --base-url https://api.parasail.io/v1 \\
    --scenario wave \\
    --users 20 \\
    --max-tokens 256 \\
    --output-json results/parasail_wave.json
""")
