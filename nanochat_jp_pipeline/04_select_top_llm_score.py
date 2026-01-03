#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Any, Optional

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


def default_repo_id() -> Optional[str]:
    namespace = os.environ.get("HF_NAMESPACE")
    if not namespace:
        return None
    prefix = os.environ.get("HF_REPO_PREFIX", "nanochat-jp")
    return f"{namespace}/{prefix}-abeja-cc-ja-edu-0.01"


def to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kajuma/ABEJA-CC-JA-edu")
    ap.add_argument("--config", default="10%")
    ap.add_argument("--split", default="train")
    ap.add_argument("--score_field", default="llm_score")
    ap.add_argument("--repo_id", default=default_repo_id())
    ap.add_argument("--config_name", default=None)
    ap.add_argument("--local_dir", default=None)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    if load_dataset is None:
        raise RuntimeError("datasets が import できません。pip install datasets を実行してください。")
    if not args.repo_id:
        raise RuntimeError("--repo_id が必要です (または HF_NAMESPACE を設定してください)。")

    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=False)

    score_col = "_llm_score_float"
    ds = ds.map(lambda x: {score_col: to_float(x.get(args.score_field))})
    ds = ds.filter(lambda x: x[score_col] is not None)

    total = len(ds)
    if total == 0:
        raise RuntimeError("score が有効な行が見つかりませんでした。")

    top_ratio = 0.1
    top_n = max(1, int(total * top_ratio))
    ds = ds.sort(score_col, reverse=True)
    ds_top = ds.select(range(top_n)).remove_columns([score_col])

    if args.local_dir:
        os.makedirs(args.local_dir, exist_ok=True)
        ds_top.save_to_disk(args.local_dir)

    config_name = args.config_name or args.config
    ds_top.push_to_hub(
        args.repo_id,
        config_name=config_name,
        split=args.split,
        private=args.private,
    )
    print("[done]", args.repo_id, "top", top_n, "of", total)


if __name__ == "__main__":
    main()
