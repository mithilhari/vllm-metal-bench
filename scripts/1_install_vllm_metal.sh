#!/usr/bin/env bash
# =============================================================================
# 1_install_vllm_metal.sh
#
# Installs vllm-metal — the official vLLM plugin for Apple Silicon.
# Installs into ~/.venv-vllm-metal (self-contained, does not touch system Python).
#
# Source: https://github.com/vllm-project/vllm-metal
# Docs:   https://docs.vllm.ai/projects/vllm-metal/en/latest/installation/
# =============================================================================

set -euo pipefail

# ── Guard: Apple Silicon only ────────────────────────────────────────────────
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
  echo "ERROR: vllm-metal requires Apple Silicon (arm64)."
  echo "       Detected architecture: $ARCH"
  exit 1
fi

OS=$(uname -s)
if [[ "$OS" != "Darwin" ]]; then
  echo "ERROR: This script is for macOS only. Detected: $OS"
  exit 1
fi

echo "=============================================="
echo "  vllm-metal installer"
echo "  Platform: $OS / $ARCH"
echo "=============================================="

# ── Check Xcode CLI tools ────────────────────────────────────────────────────
if ! xcode-select -p &>/dev/null; then
  echo ""
  echo "Xcode Command Line Tools not found. Installing..."
  echo "A dialog may appear — click Install and wait for it to complete."
  echo "Then re-run this script."
  xcode-select --install
  exit 0
fi
echo "✓ Xcode CLI tools found: $(xcode-select -p)"

# ── Check Python ─────────────────────────────────────────────────────────────
PYTHON_BIN=""
for cmd in python3.12 python3.11 python3.10 python3; do
  if command -v "$cmd" &>/dev/null; then
    VERSION=$("$cmd" --version 2>&1 | awk '{print $2}')
    MAJOR=$(echo "$VERSION" | cut -d. -f1)
    MINOR=$(echo "$VERSION" | cut -d. -f2)
    if [[ "$MAJOR" -eq 3 && "$MINOR" -ge 10 ]]; then
      PYTHON_BIN="$cmd"
      echo "✓ Python found: $cmd ($VERSION)"
      break
    fi
  fi
done

if [[ -z "$PYTHON_BIN" ]]; then
  echo ""
  echo "ERROR: Python 3.10+ not found."
  echo "       Install via: brew install python@3.12"
  exit 1
fi

# ── Check if already installed ───────────────────────────────────────────────
VENV_DIR="${HOME}/.venv-vllm-metal"
if [[ -d "$VENV_DIR" ]]; then
  echo ""
  echo "Found existing installation at: $VENV_DIR"
  read -r -p "Reinstall? This will delete the existing venv. [y/N] " REPLY
  if [[ "$REPLY" =~ ^[Yy]$ ]]; then
    echo "Removing $VENV_DIR ..."
    rm -rf "$VENV_DIR"
  else
    echo "Skipping install. To activate: source $VENV_DIR/bin/activate"
    exit 0
  fi
fi

# ── Run official install script ──────────────────────────────────────────────
echo ""
echo "Running official vllm-metal install script..."
echo "(This builds vLLM from source — may take 5–15 minutes)"
echo ""

curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash

# ── Verify ───────────────────────────────────────────────────────────────────
echo ""
if [[ -f "$VENV_DIR/bin/activate" ]]; then
  echo "=============================================="
  echo "  ✓ vllm-metal installed successfully"
  echo "=============================================="
  echo ""
  echo "To activate the environment:"
  echo "  source ~/.venv-vllm-metal/bin/activate"
  echo ""
  echo "To verify vllm is available:"
  echo "  source ~/.venv-vllm-metal/bin/activate && vllm --version"
  echo ""
  echo "Next step:"
  echo "  bash scripts/2_download_model.sh"
else
  echo "ERROR: Install appears to have failed — $VENV_DIR not found."
  echo ""
  echo "Try the manual reinstall:"
  echo "  rm -rf ~/.venv-vllm-metal && curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash"
  exit 1
fi
