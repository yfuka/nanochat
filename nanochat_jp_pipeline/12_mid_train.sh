#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/00_env.sh"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-65536}"

bash "$PIPE_DIR/03_link_identity.sh" mid

uv run torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.mid_train -- \
  --run="$WANDB_RUN" \
  --device_batch_size="$DEVICE_BATCH_SIZE" \
  --total_batch_size="$TOTAL_BATCH_SIZE" \
  --model_tag="$MODEL_TAG"

if [[ -n "${HF_TOKEN:-}" ]]; then
  uv run python "$PIPE_DIR/20_hf_push.py" --repo_type=model --repo_id "$HF_REPO_MID" \
    --local_path "$NANOCHAT_BASE_DIR/mid_checkpoints/$MODEL_TAG" --commit_message "mid_ckpt" --multi_commits
else
  echo "[mid] HF_TOKEN is empty; skip push"
fi
