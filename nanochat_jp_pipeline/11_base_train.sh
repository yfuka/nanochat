#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/00_env.sh"

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

# まずは小さく（必要なら env で上書き）
DEPTH="${DEPTH:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-64}"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
  --depth="$DEPTH" \
  --max_seq_len="$MAX_SEQ_LEN" \
  --device_batch_size="$DEVICE_BATCH_SIZE" \
  --run="$WANDB_RUN" \
  --model_tag="$MODEL_TAG"

if [[ -n "${HF_TOKEN:-}" ]]; then
  uv run python "$PIPE_DIR/20_hf_push.py" --repo_type=model --repo_id "$HF_REPO_BASE" \
    --local_path "$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG" --commit_message "base_ckpt" --multi_commits
else
  echo "[base] HF_TOKEN is empty; skip push"
fi
