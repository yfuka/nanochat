#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/00_env.sh"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TOKENIZER_MODE="sample"

uv run python -m  analyze.analyze_data --base-dir "$NANOCHAT_BASE_DIR" --tokenizer-mode "$TOKENIZER_MODE" --sample-rows 240000

# root@f41490c3b307:/workspace/nanochat# bash ./nanochat_jp_pipeline/41_analyze_data.sh 
# scan parquet metadata: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:02<00:00, 235.67files/s]
# scan jp_mid.jsonl: 165174rows [00:02, 81291.40rows/s]
# scan jp_sft.jsonl: 132893rows [00:03, 39120.05rows/s]
# {
#   "NANOCHAT_BASE_DIR": "/workspace/nanochat_cache",
#   "base_data": {
#     "exists": true,
#     "path": "/workspace/nanochat_cache/base_data",
#     "num_files": 608,
#     "total_file_bytes": 98404332847,
#     "total_file_bytes_h": "91.6GB",
#     "total_rows": 24243597,
#     "sampled_rows": 240000,
#     "text_len_chars_quantiles": {
#       "p50": 1956,
#       "p90": 5086,
#       "p95": 6833,
#       "p99": 12958
#     },
#     "token_stats": {
#       "mode": "sample",
#       "available": true,
#       "rows_tokenized": 240000,
#       "tokens": 249413129,
#       "avg_tokens_per_row": 1039.2213708333334
#     }
#   },
#   "mid_data": {
#     "exists": true,
#     "path": "/workspace/nanochat_cache/mid_data",
#     "files": {
#       "jp_mid.jsonl": {
#         "exists": true,
#         "path": "/workspace/nanochat_cache/mid_data/jp_mid.jsonl",
#         "conversations": 165174,
#         "messages": 543306,
#         "user_messages": 189066,
#         "assistant_messages": 189066,
#         "total_chars": 115422116,
#         "avg_chars_per_conv": 698.7910688122828,
#         "turns_top10": [
#           [
#             3,
#             142448
#           ],
#           [
#             5,
#             21560
#           ],
#           [
#             7,
#             1166
#           ]
#         ]
#       }
#     }
#   },
#   "sft_data": {
#     "exists": true,
#     "path": "/workspace/nanochat_cache/sft_data",
#     "files": {
#       "jp_sft.jsonl": {
#         "exists": true,
#         "path": "/workspace/nanochat_cache/sft_data/jp_sft.jsonl",
#         "conversations": 132893,
#         "messages": 479375,
#         "user_messages": 173241,
#         "assistant_messages": 173241,
#         "total_chars": 293763327,
#         "avg_chars_per_conv": 2210.525212012672,
#         "turns_top10": [
#           [
#             3,
#             92545
#           ],
#           [
#             5,
#             40348
#           ]
#         ]
#       }
#     }
#   }
# }