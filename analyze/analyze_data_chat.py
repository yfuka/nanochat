#!/usr/bin/env python3
"""
チャットデータの統計処理。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator

from tqdm import tqdm


def iter_jsonl(path: Path) -> Iterator[Any]:
    """JSONL を1行ずつ読み込む。

    Args:
        path: JSONL ファイルのパス。

    Yields:
        JSON オブジェクト。
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def scan_chat_file(fp: Path) -> Dict[str, Any]:
    """チャット形式 JSONL の統計を集計する。

    Args:
        fp: JSONL ファイルのパス。

    Returns:
        統計結果。
    """
    if not fp.exists():
        return {"exists": False}

    n_conv = 0
    n_msgs = 0
    n_user = 0
    n_assistant = 0
    total_chars = 0
    turns_hist = Counter()

    for obj in tqdm(iter_jsonl(fp), desc=f"scan {fp.name}", unit="rows"):
        n_conv += 1
        if not isinstance(obj, list):
            continue
        turns_hist[len(obj)] += 1
        n_msgs += len(obj)
        for m in obj:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                n_user += 1
            if role == "assistant":
                n_assistant += 1
            if isinstance(content, str):
                total_chars += len(content)

    return {
        "exists": True,
        "path": str(fp),
        "conversations": n_conv,
        "messages": n_msgs,
        "user_messages": n_user,
        "assistant_messages": n_assistant,
        "total_chars": total_chars,
        "avg_chars_per_conv": (total_chars / max(n_conv, 1)),
        "turns_top10": turns_hist.most_common(10),
    }


def analyze_chat_dir(dir_path: Path) -> Dict[str, Any]:
    """チャットデータディレクトリを解析する。

    Args:
        dir_path: ディレクトリパス。

    Returns:
        統計結果。
    """
    if not dir_path.exists():
        return {"exists": False}

    train_path = dir_path / "train.jsonl"
    val_path = dir_path / "val.jsonl"
    if train_path.exists() or val_path.exists():
        return {
            "exists": True,
            "path": str(dir_path),
            "train": scan_chat_file(train_path),
            "val": scan_chat_file(val_path),
        }

    files = sorted(dir_path.glob("*.jsonl"))
    return {
        "exists": True,
        "path": str(dir_path),
        "files": {f.name: scan_chat_file(f) for f in files},
    }
