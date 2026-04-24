#!/usr/bin/env bash
# =============================================================================
# 3_serve.sh
#
# Starts vllm serve using the vllm-metal backend on Apple Silicon.
# Exposes an OpenAI-compatible API on http://localhost:8000
#
# Usage:
#   bash scripts/3_serve.sh
#
#   # Override model or port:
#   MODEL_PATH=~/models/Qwen2.5-7B-Instruct PORT=8001 bash scripts/3_serve.sh
# =============================================================================

set -euo pipefail

# ── Config — override via env vars ───────────────────────────────────────────
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
MODEL_DIR_NAME=$(echo "$MODEL_NAME" | cut -d'/' -f2)
MODEL_PATH="${MODEL_PATH:-${HOME}/models/${MODEL_DIR_NAME}}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"     # reduce if you hit OOM on 16GB RAM
VENV_DIR="${HOME}/.venv-vllm-metal"

echo "=============================================="
echo "  vllm serve  (vllm-metal / Apple Silicon)"
echo "=============================================="
echo "  Model:        $MODEL_NAME"
echo "  Weights path: $MODEL_PATH"
echo "  Endpoint:     http://localhost:$PORT"
echo "  Max ctx len:  $MAX_MODEL_LEN tokens"
echo "=============================================="

# ── Guards ────────────────────────────────────────────────────────────────────
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "ERROR: vllm-metal venv not found."
  echo "       Run: bash scripts/1_install_vllm_metal.sh"
  exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: Model directory not found: $MODEL_PATH"
  echo "       Run: bash scripts/2_download_model.sh"
  exit 1
fi

# ── Activate venv ────────────────────────────────────────────────────────────
source "$VENV_DIR/bin/activate"
echo "✓ Activated: $VENV_DIR"
echo "✓ vLLM:      $(vllm --version 2>/dev/null || echo 'unknown')"
echo ""

# ── Memory guidance ───────────────────────────────────────────────────────────
TOTAL_MEM_GB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
echo "System unified memory: ${TOTAL_MEM_GB} GB"

if [[ "$TOTAL_MEM_GB" -lt 24 ]]; then
  echo "⚠  16 GB detected — using conservative MAX_MODEL_LEN=$MAX_MODEL_LEN"
  echo "   If you hit OOM, reduce with: MAX_MODEL_LEN=2048 bash scripts/3_serve.sh"
fi
echo ""
echo "Starting vllm serve..."
echo "Wait for: 'Application startup complete' before running the traffic generator."
echo "Press Ctrl+C to stop."
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
exec vllm serve "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --disable-log-requests
