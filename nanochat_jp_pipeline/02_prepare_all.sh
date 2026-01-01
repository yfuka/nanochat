#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/00_env.sh"

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "== prepare base_data parquet =="
uv run python "$PIPE_DIR/02_prepare_data.py" --config "$PIPE_DIR/02_data_config.yaml" --stage base

echo "== prepare mid jsonl =="
uv run python "$PIPE_DIR/02_prepare_data.py" --config "$PIPE_DIR/02_data_config.yaml" --stage mid

echo "== prepare sft jsonl =="
uv run python "$PIPE_DIR/02_prepare_data.py" --config "$PIPE_DIR/02_data_config.yaml" --stage sft

echo "== done =="
echo "mid/sft は identity_conversations.jsonl を symlink で切り替えて使います（公式I/F準拠）"
