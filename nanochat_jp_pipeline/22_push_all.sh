#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/00_env.sh"

# data（mid/sft jsonlだけ）: dataset repo
uv run python "$PIPE_DIR/20_hf_push.py" --repo_type=dataset --repo_id "$HF_REPO_DATA" \
  --local_path "$NANOCHAT_BASE_DIR/mid_data/jp_mid.jsonl" --commit_message "mid_jsonl"

uv run python "$PIPE_DIR/20_hf_push.py" --repo_type=dataset --repo_id "$HF_REPO_DATA" \
  --local_path "$NANOCHAT_BASE_DIR/sft_data/jp_sft.jsonl" --commit_message "sft_jsonl"

# tokenizer
uv run python "$PIPE_DIR/20_hf_push.py" --repo_type=model --repo_id "$HF_REPO_TOKENIZER" \
  --local_path "$NANOCHAT_BASE_DIR/tokenizer" --commit_message "tokenizer" --multi_commits

# checkpoints
uv run python "$PIPE_DIR/20_hf_push.py" --repo_type=model --repo_id "$HF_REPO_BASE" \
  --local_path "$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG" --commit_message "base_ckpt" --multi_commits

uv run python "$PIPE_DIR/20_hf_push.py" --repo_type=model --repo_id "$HF_REPO_MID" \
  --local_path "$NANOCHAT_BASE_DIR/mid_checkpoints/$MODEL_TAG" --commit_message "mid_ckpt" --multi_commits

uv run python "$PIPE_DIR/20_hf_push.py" --repo_type=model --repo_id "$HF_REPO_SFT" \
  --local_path "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$MODEL_TAG" --commit_message "sft_ckpt" --multi_commits
