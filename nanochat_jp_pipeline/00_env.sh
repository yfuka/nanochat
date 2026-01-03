#!/usr/bin/env bash
set -euo pipefail

# 1) 永続キャッシュ（ここが消えないことが最重要）
export NANOCHAT_BASE_DIR=/workspace/nanochat_cache
export UV_CACHE_DIR=/workspace/.uv_cache

# 2) 作業ディレクトリ
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PIPE_DIR="${PIPE_DIR:-$SCRIPT_DIR}"

# 3) Hugging Face の保存先（あなたのアカウント名に合わせて変更）
export HF_TOKEN="${HF_TOKEN:-<YOUR_HF_TOKEN>}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_NAMESPACE="${HF_NAMESPACE:-<YOUR_HF_USERNAME>}"
export HF_REPO_PREFIX="${HF_REPO_PREFIX:-nanochat-jp}"

export HF_REPO_TOKENIZER="${HF_NAMESPACE}/${HF_REPO_PREFIX}-tokenizer"
export HF_REPO_DATA="${HF_NAMESPACE}/${HF_REPO_PREFIX}-${MODEL_TAG}-data"
export HF_REPO_BASE="${HF_NAMESPACE}/${HF_REPO_PREFIX}-${MODEL_TAG}-base"
export HF_REPO_MID="${HF_NAMESPACE}/${HF_REPO_PREFIX}-${MODEL_TAG}-mid"
export HF_REPO_SFT="${HF_NAMESPACE}/${HF_REPO_PREFIX}-${MODEL_TAG}-sft"

# 4) Weights & Biases の設定
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# 5) 学習
export MODEL_TAG="d8"
export TOK_MAX_CHARS="${TOK_MAX_CHARS:-400000000}"
export SEED="${SEED:-42}"

# 6) 環境依存
export OMP_NUM_THREADS=1
export NPROC_PER_NODE=8
# export NCCL_P2P_LEVEL=NVL
# export NCCL_P2P_DISABLE=1