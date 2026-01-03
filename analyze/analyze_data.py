#!/usr/bin/env python3
"""
nanochat 互換データの簡易統計ツール。

Usage:
  python analyze_data.py --base-dir /path/to/.cache/nanochat
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .analyze_data_base import analyze_base
from .analyze_data_chat import analyze_chat_dir


def build_report(
    root: Path,
    sample_rows: int,
    tokenizer_mode: str,
    tokenizer_dir: Path,
    tokenizer_batch_size: int,
) -> Dict[str, Any]:
    """全体レポートを構築する。

    Args:
        root: NANOCHAT_BASE_DIR。
        sample_rows: サンプル行数。
        tokenizer_mode: "none" / "sample" / "full"。
        tokenizer_dir: tokenizer ディレクトリ。
        tokenizer_batch_size: トークナイズのバッチサイズ。

    Returns:
        JSON で出力するレポート。
    """
    return {
        "NANOCHAT_BASE_DIR": str(root),
        "base_data": analyze_base(
            root / "base_data",
            sample_rows,
            tokenizer_dir,
            tokenizer_mode,
            tokenizer_batch_size,
        ),
        "mid_data": analyze_chat_dir(root / "mid_data"),
        "sft_data": analyze_chat_dir(root / "sft_data"),
    }


def parse_args() -> argparse.Namespace:
    """CLI 引数を解析する。

    Returns:
        引数オブジェクト。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True, help="NANOCHAT_BASE_DIR (e.g. /data/.cache/nanochat)")
    ap.add_argument("--sample-rows", type=int, default=40000, help="sample rows for text length stats")
    ap.add_argument(
        "--tokenizer-mode",
        choices=["none", "sample", "full"],
        default="none",
        help="tokenize none/sample/full rows for token count stats",
    )
    ap.add_argument("--tokenizer-dir", default="", help="tokenizer directory override (default: BASE_DIR/tokenizer)")
    ap.add_argument("--tokenizer-batch-size", type=int, default=128, help="batch size for tokenization")
    ap.add_argument("--out", default="", help="write JSON report to this path (optional)")
    return ap.parse_args()


def main() -> None:
    """エントリポイント。"""
    args = parse_args()
    root = Path(args.base_dir)
    tokenizer_dir = Path(args.tokenizer_dir) if args.tokenizer_dir else root / "tokenizer"

    report = build_report(
        root=root,
        sample_rows=args.sample_rows,
        tokenizer_mode=args.tokenizer_mode,
        tokenizer_dir=tokenizer_dir,
        tokenizer_batch_size=args.tokenizer_batch_size,
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] wrote {outp}")


if __name__ == "__main__":
    main()
