#!/usr/bin/env bash
set -euo pipefail
source /workspace/nanochat_jp_pipeline/00_env.sh

cd "$NANOCHAT_REPO_DIR"
source .venv/bin/activate

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_loss
