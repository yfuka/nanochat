#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/00_env.sh"

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

DEPTH="${DEPTH:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-524288}"
NUM_ITERATIONS="${NUM_ITERATIONS:-45000}"

export WANDB_RUN="base-train-d${DEPTH}-msl${MAX_SEQ_LEN}-dbs${DEVICE_BATCH_SIZE}-tbs${TOTAL_BATCH_SIZE}-ni${NUM_ITERATIONS}"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
  --depth="$DEPTH" \
  --max_seq_len="$MAX_SEQ_LEN" \
  --device_batch_size="$DEVICE_BATCH_SIZE" \
  --total_batch_size="$TOTAL_BATCH_SIZE" \
  --run="$WANDB_RUN" \
  --model_tag="$MODEL_TAG" \
  --num_iterations="$NUM_ITERATIONS"

if [[ -n "${HF_TOKEN:-}" ]]; then
  uv run python "$PIPE_DIR/20_hf_push.py" --repo_type=model --repo_id "$HF_REPO_BASE" \
    --local_path "$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG" --commit_message "base_ckpt" --multi_commits
else
  echo "[base] HF_TOKEN is empty; skip push"
fi
