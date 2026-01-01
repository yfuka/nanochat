#!/usr/bin/env python3
"""
base_data の統計処理。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pyarrow.parquet as pq
from tqdm import tqdm

from .analyze_data_tokenizer import count_tokens, load_tokenizer


def human_bytes(n: int) -> str:
    """バイト数を人間が読みやすい形式に変換する。

    Args:
        n: バイト数。

    Returns:
        人間が読みやすい表現（例: 12.3MB）。
    """
    value = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024:
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}PB"


def iter_parquet_texts(files: List[Path], max_rows: int) -> Iterator[str]:
    """Parquet 群から text カラムを順に取り出す。

    Args:
        files: Parquet ファイル一覧。
        max_rows: 最大行数。0 以下は制限なし。

    Yields:
        text カラムの文字列。
    """
    remaining = None if max_rows <= 0 else max_rows
    for fp in files:
        pf = pq.ParquetFile(fp)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx, columns=["text"])
            col = rg.column("text").to_pylist()
            for t in col:
                if remaining is not None and remaining <= 0:
                    return
                if isinstance(t, str):
                    yield t
                    if remaining is not None:
                        remaining -= 1


def sample_text_lengths(files: List[Path], sample_rows_total: int) -> List[int]:
    """先頭行をサンプルして文字数分布を取る。

    Args:
        files: Parquet ファイル一覧。
        sample_rows_total: サンプルする最大行数。

    Returns:
        文字数の配列。
    """
    lens: List[int] = []
    remaining = sample_rows_total
    for fp in files:
        if remaining <= 0:
            break
        pf = pq.ParquetFile(fp)
        if pf.num_row_groups <= 0:
            continue
        rg = pf.read_row_group(0, columns=["text"])
        col = rg.column("text").to_pylist()
        for t in col:
            if remaining <= 0:
                break
            if isinstance(t, str):
                lens.append(len(t))
                remaining -= 1
    return lens


def analyze_base(
    base_dir: Path,
    sample_rows_total: int,
    tokenizer_dir: Path,
    tokenizer_mode: str,
    tokenizer_batch_size: int,
) -> Dict[str, Any]:
    """base_data の統計を集計する。

    Args:
        base_dir: base_data のディレクトリ。
        sample_rows_total: 文字数分布用のサンプル行数。
        tokenizer_dir: tokenizer ディレクトリ。
        tokenizer_mode: "none" / "sample" / "full"。
        tokenizer_batch_size: トークナイズのバッチサイズ。

    Returns:
        base_data の統計結果。
    """
    if not base_dir.exists():
        return {"exists": False}

    files = sorted(base_dir.glob("*.parquet"))
    total_bytes = sum(f.stat().st_size for f in files)
    total_rows = 0
    for f in tqdm(files, desc="scan parquet metadata", unit="files"):
        try:
            pf = pq.ParquetFile(f)
            total_rows += pf.metadata.num_rows
        except Exception:
            pass

    lens = sorted(sample_text_lengths(files, sample_rows_total))
    def q(p: float) -> Optional[int]:
        if not lens:
            return None
        idx = int(p * (len(lens) - 1))
        return lens[idx]

    token_stats: Dict[str, Any] = {"mode": tokenizer_mode}
    if tokenizer_mode != "none":
        tokenizer, err = load_tokenizer(tokenizer_dir)
        if tokenizer is None:
            token_stats.update({"available": False, "reason": err})
        else:
            max_rows = 0 if tokenizer_mode == "full" else sample_rows_total
            token_stats.update(
                {
                    "available": True,
                    **count_tokens(
                        iter_parquet_texts(files, max_rows=max_rows),
                        tokenizer,
                        tokenizer_batch_size,
                    ),
                }
            )

    return {
        "exists": True,
        "path": str(base_dir),
        "num_files": len(files),
        "total_file_bytes": total_bytes,
        "total_file_bytes_h": human_bytes(total_bytes),
        "total_rows": total_rows,
        "sampled_rows": len(lens),
        "text_len_chars_quantiles": {
            "p50": q(0.50),
            "p90": q(0.90),
            "p95": q(0.95),
            "p99": q(0.99),
        },
        "token_stats": token_stats,
    }
