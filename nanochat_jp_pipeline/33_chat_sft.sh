#!/usr/bin/env bash
set -euo pipefail
source /workspace/nanochat_jp_pipeline/00_env.sh

cd "$NANOCHAT_REPO_DIR"
source .venv/bin/activate

uv run python -m scripts.chat_cli -i sft --model-tag "$MODEL_TAG"
