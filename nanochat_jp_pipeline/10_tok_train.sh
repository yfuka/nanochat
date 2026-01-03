#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/00_env.sh"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TOK_MAX_CHARS="${TOK_MAX_CHARS:-400000000}"  # 小さめでE2E
uv run python -m scripts.tok_train --max_chars="$TOK_MAX_CHARS"
uv run python -m scripts.tok_eval || true

if [[ -n "${HF_TOKEN:-}" ]]; then
  uv run python "$PIPE_DIR/20_hf_push.py" --repo_type=model --repo_id "$HF_REPO_TOKENIZER" \
    --local_path "$NANOCHAT_BASE_DIR/tokenizer" --commit_message "tokenizer" --multi_commits
else
  echo "[tok] HF_TOKEN is empty; skip push"
fi
