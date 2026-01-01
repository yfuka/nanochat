#!/usr/bin/env python3
"""
トークナイザ関連の補助処理。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


def load_tokenizer(tokenizer_dir: Path) -> Tuple[Optional[Any], Optional[str]]:
    """トークナイザをロードする。

    Args:
        tokenizer_dir: tokenizer ファイルがあるディレクトリ。

    Returns:
        (tokenizer, error_message) のタプル。成功時は error_message が None。
    """
    try:
        from nanochat.tokenizer import RustBPETokenizer, HuggingFaceTokenizer
    except Exception as e:
        return None, f"tokenizer import failed: {e}"
    if (tokenizer_dir / "tokenizer.pkl").exists():
        return RustBPETokenizer.from_directory(str(tokenizer_dir)), None
    if (tokenizer_dir / "tokenizer.json").exists():
        return HuggingFaceTokenizer.from_directory(str(tokenizer_dir)), None
    return None, "tokenizer files not found"


def count_tokens(text_iter: Iterator[str], tokenizer: Any, batch_size: int) -> Dict[str, Any]:
    """テキスト列のトークン数を集計する。

    Args:
        text_iter: テキストのイテレータ。
        tokenizer: nanochat の tokenizer。
        batch_size: まとめてエンコードする行数。

    Returns:
        トークン集計結果。
    """
    total_tokens = 0
    total_rows = 0
    bos = tokenizer.get_bos_token_id() if hasattr(tokenizer, "get_bos_token_id") else None
    batch: List[str] = []

    def flush(batch_texts: List[str]) -> None:
        nonlocal total_tokens, total_rows
        if not batch_texts:
            return
        if bos is None:
            ids_list = tokenizer.encode(batch_texts)
        else:
            ids_list = tokenizer.encode(batch_texts, prepend=bos)
        total_tokens += sum(len(ids) for ids in ids_list)
        total_rows += len(batch_texts)

    for text in text_iter:
        batch.append(text)
        if len(batch) >= batch_size:
            flush(batch)
            batch = []
    flush(batch)

    return {
        "rows_tokenized": total_rows,
        "tokens": total_tokens,
        "avg_tokens_per_row": (total_tokens / max(total_rows, 1)),
    }
