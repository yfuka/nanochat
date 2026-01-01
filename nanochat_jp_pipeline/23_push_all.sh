#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/00_env.sh"

uv run python "$PIPE_DIR/21_hf_pull.py" --repo_type=model --repo_id "$HF_REPO_TOKENIZER" \
  --local_dir "$NANOCHAT_BASE_DIR/tokenizer"

uv run python "$PIPE_DIR/21_hf_pull.py" --repo_type=model --repo_id "$HF_REPO_BASE" \
  --local_dir "$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"

uv run python "$PIPE_DIR/21_hf_pull.py" --repo_type=model --repo_id "$HF_REPO_MID" \
  --local_dir "$NANOCHAT_BASE_DIR/mid_checkpoints/$MODEL_TAG"

uv run python "$PIPE_DIR/21_hf_pull.py" --repo_type=model --repo_id "$HF_REPO_SFT" \
  --local_dir "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$MODEL_TAG"

uv run python "$PIPE_DIR/21_hf_pull.py" --repo_type=dataset --repo_id "$HF_REPO_DATA" \
  --local_dir "$NANOCHAT_BASE_DIR/_pulled_data"
