#!/usr/bin/env bash
set -euo pipefail

# 1) 永続キャッシュ（ここが消えないことが最重要）
export NANOCHAT_BASE_DIR=/workspace/nanochat_cache
export UV_CACHE_DIR=/workspace/.uv_cache

# 2) 作業ディレクトリ
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PIPE_DIR="${PIPE_DIR:-$SCRIPT_DIR}"

# 3) 超小さいモデルから（まずE2E確認）
export MODEL_TAG=jp_d8
export NPROC_PER_NODE=1
export WANDB_RUN=dummy

# 4) Hugging Face の保存先（あなたのアカウント名に合わせて変更）
export HF_TOKEN="${HF_TOKEN:-<your_token>}"
export HF_NAMESPACE="${HF_NAMESPACE:-<your_hf_username>}"
export HF_REPO_PREFIX="${HF_REPO_PREFIX:-nanochat-jp}"

export HF_REPO_TOKENIZER="${HF_NAMESPACE}/${HF_REPO_PREFIX}-${MODEL_TAG}-tokenizer}"
export HF_REPO_DATA="${HF_NAMESPACE}/${HF_REPO_PREFIX}-${MODEL_TAG}-data}"
export HF_REPO_BASE="${HF_NAMESPACE}/${HF_REPO_PREFIX}-${MODEL_TAG}-base}"
export HF_REPO_MID="${HF_NAMESPACE}/${HF_REPO_PREFIX}-${MODEL_TAG}-mid}"
export HF_REPO_SFT="${HF_NAMESPACE}/${HF_REPO_PREFIX}-${MODEL_TAG}-sft}"

# 5) データ
# ---- tokenizer ----
export TOK_MAX_CHARS="${TOK_MAX_CHARS:-200000000}"  # まず小さく
export SEED="${SEED:-42}"
