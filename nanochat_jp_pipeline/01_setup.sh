#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/00_env.sh"

mkdir -p "$NANOCHAT_BASE_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# OSレベルのビルド要件
apt-get update
apt-get install -y python3.10-dev build-essential

# ---------------------------------------------

# uv setup
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv
uv sync --extra gpu
# ---------------------------------------------

# --- Rust toolchain for maturin (persist on /workspace) ---
if ! command -v rustc >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"
# ---------------------------------------------

# Rust tokenizer をビルド（nanochat README/解説で一般的な手順）
# ※これで nanochat.tokenizer から Rust 実装を呼べるようになる
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
