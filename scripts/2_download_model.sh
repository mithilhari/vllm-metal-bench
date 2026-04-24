#!/usr/bin/env bash
# =============================================================================
# 2_download_model.sh
#
# Downloads model weights from HuggingFace into ~/models/
#
# Usage:
#   export HF_TOKEN=hf_your_token_here
#   bash scripts/2_download_model.sh
#
#   # Or override the model:
#   MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct bash scripts/2_download_model.sh
# =============================================================================

set -euo pipefail

# ── Config — override via env vars ───────────────────────────────────────────
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
MODELS_DIR="${MODELS_DIR:-${HOME}/models}"
VENV_DIR="${HOME}/.venv-vllm-metal"

# Derive local directory name from model (replace / with -)
MODEL_DIR_NAME=$(echo "$MODEL" | cut -d'/' -f2)
LOCAL_MODEL_PATH="${MODELS_DIR}/${MODEL_DIR_NAME}"

echo "=============================================="
echo "  Model Downloader"
echo "=============================================="
echo "  Model:      $MODEL"
echo "  Saving to:  $LOCAL_MODEL_PATH"
echo "=============================================="

# ── Guard: HF token ──────────────────────────────────────────────────────────
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo ""
  echo "ERROR: HF_TOKEN is not set."
  echo ""
  echo "For public models (e.g. Qwen, Mistral), you may not need a token."
  echo "For gated models (e.g. Llama 3), you need a HuggingFace token with"
  echo "access granted at: https://huggingface.co/$MODEL"
  echo ""
  echo "Set it with:"
  echo "  export HF_TOKEN=hf_your_token_here"
  echo "  bash scripts/2_download_model.sh"
  echo ""
  read -r -p "Continue without token? (only works for ungated models) [y/N] " REPLY
  if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi

# ── Activate venv ────────────────────────────────────────────────────────────
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "ERROR: vllm-metal venv not found at $VENV_DIR"
  echo "       Run: bash scripts/1_install_vllm_metal.sh"
  exit 1
fi

source "$VENV_DIR/bin/activate"
echo "✓ Activated venv: $VENV_DIR"

# ── Install huggingface_hub if missing ───────────────────────────────────────
if ! python3 -c "import huggingface_hub" &>/dev/null; then
  echo "Installing huggingface_hub..."
  pip install -q huggingface-hub
fi

# ── Check if already downloaded ──────────────────────────────────────────────
if [[ -d "$LOCAL_MODEL_PATH" ]]; then
  echo ""
  echo "Model directory already exists: $LOCAL_MODEL_PATH"
  read -r -p "Re-download? [y/N] " REPLY
  if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
    echo "Skipping download. Model path: $LOCAL_MODEL_PATH"
    echo ""
    echo "To serve:  bash scripts/3_serve.sh"
    exit 0
  fi
fi

mkdir -p "$MODELS_DIR"

# ── Download ──────────────────────────────────────────────────────────────────
echo ""
echo "Downloading $MODEL ..."
echo "(This may take a while — 8B model is ~16 GB)"
echo ""

HF_ARGS=(
  "download"
  "$MODEL"
  "--local-dir" "$LOCAL_MODEL_PATH"
  "--local-dir-use-symlinks" "False"
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_ARGS+=("--token" "$HF_TOKEN")
fi

python3 -c "
from huggingface_hub import snapshot_download
import os, sys

token = os.environ.get('HF_TOKEN') or None
model = sys.argv[1]
local_dir = sys.argv[2]

print(f'Downloading {model} to {local_dir} ...')
snapshot_download(
    repo_id=model,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    token=token,
    ignore_patterns=['*.msgpack', '*.h5', 'flax_model*', 'tf_model*'],
)
print('Download complete.')
" "$MODEL" "$LOCAL_MODEL_PATH"

# ── Checksum the safetensors ──────────────────────────────────────────────────
echo ""
echo "Computing checksums (for future verification)..."
CHECKSUM_FILE="${LOCAL_MODEL_PATH}/CHECKSUMS.sha256"

find "$LOCAL_MODEL_PATH" -type f \( \
  -name "*.safetensors" \
  -o -name "*.json" \
  -o -name "tokenizer.model" \
  -o -name "tokenizer.model.v3" \
\) | sort | xargs shasum -a 256 > "$CHECKSUM_FILE"

echo "✓ Checksums saved to: $CHECKSUM_FILE"
echo "  Files checksummed: $(wc -l < "$CHECKSUM_FILE" | tr -d ' ')"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
MODEL_SIZE=$(du -sh "$LOCAL_MODEL_PATH" | cut -f1)
echo "=============================================="
echo "  ✓ Download complete"
echo "=============================================="
echo "  Path:  $LOCAL_MODEL_PATH"
echo "  Size:  $MODEL_SIZE"
echo ""
echo "Next step:"
echo "  bash scripts/3_serve.sh"
