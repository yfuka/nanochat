#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/00_env.sh"

MODE="${1:-mid}"  # mid | sft
TARGET="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"

if [ "$MODE" = "mid" ]; then
  SRC="$NANOCHAT_BASE_DIR/mid_data/jp_mid.jsonl"
elif [ "$MODE" = "sft" ]; then
  SRC="$NANOCHAT_BASE_DIR/sft_data/jp_sft.jsonl"
else
  echo "usage: $0 (mid|sft)"
  exit 2
fi

ln -sf "$SRC" "$TARGET"
echo "[link] $TARGET -> $SRC"
