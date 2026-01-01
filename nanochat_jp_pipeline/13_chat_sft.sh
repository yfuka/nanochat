#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/00_env.sh"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

bash "$PIPE_DIR/03_link_identity.sh" sft
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
TARGET_EXAMPLES_PER_STOP="${TARGET_EXAMPLES_PER_STOP:-1024}"

uv run torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- \
  --run="$WANDB_RUN" \
  --device_batch_size="$DEVICE_BATCH_SIZE" \
  --target_examples_per_step="$TARGET_EXAMPLES_PER_STOP" \
  --model_tag="$MODEL_TAG"

if [[ -n "${HF_TOKEN:-}" ]]; then
  uv run python "$PIPE_DIR/20_hf_push.py" --repo_type=model --repo_id "$HF_REPO_SFT" \
    --local_path "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$MODEL_TAG" --commit_message "sft_ckpt" --multi_commits
else
  echo "[sft] HF_TOKEN is empty; skip push"
fi
