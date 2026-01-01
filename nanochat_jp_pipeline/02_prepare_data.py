#!/usr/bin/env python3
"""
Compact data prep for nanochat speedrun I/F.

Usage:
  python 02_prepare_data.py --config 02_data_config.yaml --stage base
  python 02_prepare_data.py --config 02_data_config.yaml --stage mid
  python 02_prepare_data.py --config 02_data_config.yaml --stage sft
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from string import Template
from typing import Any, Dict, Iterator, List, Optional

import yaml
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

ROLE_MAP = {
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "bot": "assistant",
    "gpt": "assistant",
    "system": "system",
}

DEFAULT_SYSTEM_PROMPTS = [
    "あなたは親切で丁寧なアシスタントAIです。名前は優です。",
    "あなたは親切なAIアシスタントで、名前は優と名乗ります。",
    "あなたは思いやりのあるアシスタントAIで、名前は優です。",
]


def norm_text(value: Any) -> Optional[str]:
    """値を正規化して文字列またはNoneにする。

    Args:
        value: 正規化する入力値。

    Returns:
        空でない場合は正規化済み文字列、空ならNone。
    """
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.replace("\u0000", "").strip()
    return value if value else None


def expand_vars(obj: Any, vars_dict: Dict[str, str]) -> Any:
    if isinstance(obj, str):
        return Template(obj).safe_substitute(vars_dict)
    if isinstance(obj, list):
        return [expand_vars(x, vars_dict) for x in obj]
    if isinstance(obj, dict):
        return {k: expand_vars(v, vars_dict) for k, v in obj.items()}
    return obj


def deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    if isinstance(base, list) and isinstance(override, list):
        return base + override
    return override


def load_yaml_with_includes(path: str, stack: Optional[List[str]] = None) -> Dict[str, Any]:
    stack = stack or []
    abs_path = str(Path(path).resolve())
    if abs_path in stack:
        cycle = " -> ".join(stack + [abs_path])
        raise ValueError(f"Config include cycle detected: {cycle}")
    stack.append(abs_path)
    with open(abs_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {abs_path}")
    includes = raw.pop("include", None) or raw.pop("includes", None)
    merged: Dict[str, Any] = {}
    if includes:
        if isinstance(includes, str):
            includes = [includes]
        if not isinstance(includes, list):
            raise ValueError("include/includes must be a string or list of strings")
        for inc in includes:
            inc_path = inc
            if not os.path.isabs(inc_path):
                inc_path = os.path.join(os.path.dirname(abs_path), inc_path)
            inc_cfg = load_yaml_with_includes(inc_path, stack)
            merged = deep_merge(merged, inc_cfg)
    stack.pop()
    return deep_merge(merged, raw)


def read_config(path: str) -> Dict[str, Any]:
    # Supports include/includes with deep-merge; lists are concatenated.
    raw = load_yaml_with_includes(path)
    base_vars = {k: str(v) for k, v in os.environ.items()}
    raw_base_dir = raw.get("nanochat_base_dir", "")
    expanded_base_dir = Template(str(raw_base_dir)).safe_substitute(base_vars)
    base_vars["nanochat_base_dir"] = expanded_base_dir
    return expand_vars(raw, base_vars)


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_text_files(paths: List[str]) -> Iterator[Dict[str, Any]]:
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip("\n")
                if line:
                    yield {"text": line}


def iter_parquet_dir(path: str) -> Iterator[Dict[str, Any]]:
    ds = pa.dataset.dataset(path, format="parquet")
    for batch in ds.to_batches():
        tbl = pa.Table.from_batches([batch])
        cols = tbl.column_names
        text_col = "text" if "text" in cols else cols[0]
        texts = tbl[text_col].to_pylist()
        for i, t in enumerate(texts):
            row = {c: tbl[c][i].as_py() for c in cols}
            row["text"] = t
            yield row


def iter_hf_dataset(name: str, config: Optional[str], split: str, streaming: bool) -> Iterator[Dict[str, Any]]:
    if load_dataset is None:
        raise RuntimeError("datasets が import できません。pip install datasets を実行してください。")
    ds = load_dataset(name, config, split=split, streaming=streaming)
    for ex in ds:
        yield dict(ex)


def build_input_iterator(inp_cfg: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    t = inp_cfg["type"]
    if t == "jsonl":
        it = iter_jsonl(inp_cfg["path"])
    if t == "parquet_dir":
        it = iter_parquet_dir(inp_cfg["path"])
    if t == "text":
        paths = inp_cfg.get("paths") or [inp_cfg["path"]]
        it = iter_text_files(paths)
    if t == "hf":
        it = iter_hf_dataset(
            inp_cfg["name"],
            inp_cfg.get("config"),
            inp_cfg.get("split", "train"),
            bool(inp_cfg.get("streaming", True)),
        )
    if t not in {"jsonl", "parquet_dir", "text", "hf"}:
        raise ValueError(f"Unknown input type: {t}")

    max_rows = int(inp_cfg.get("max_rows", 0) or 0)
    if max_rows > 0:
        def limited_iter() -> Iterator[Dict[str, Any]]:
            count = 0
            for rec in it:
                yield rec
                count += 1
                if count >= max_rows:
                    break
        return limited_iter()

    return it


def normalize_role(role: Any) -> Optional[str]:
    if not isinstance(role, str):
        return None
    return ROLE_MAP.get(role.strip().lower())


def normalize_content(content: Any) -> Optional[str]:
    text = norm_text(content)
    return text if text else None


def to_messages(
    record: Dict[str, Any],
    mapping: Dict[str, str],
    system_prompt: Optional[str],
) -> Optional[List[Dict[str, str]]]:
    mf = mapping.get("messages_field", "messages")
    if mf in record and isinstance(record[mf], list):
        msgs: List[Dict[str, str]] = []
        for msg in record[mf]:
            if not isinstance(msg, dict):
                continue
            role = normalize_role(msg.get("role"))
            if role not in ("user", "assistant"):
                continue
            content = normalize_content(msg.get("content"))
            if not content:
                continue
            msgs.append({"role": role, "content": content})
        # enforce user/assistant alternation starting from user
        fixed: List[Dict[str, str]] = []
        expect = "user"
        for msg in msgs:
            if msg["role"] != expect:
                continue
            fixed.append(msg)
            expect = "assistant" if expect == "user" else "user"
        if len(fixed) < 2:
            return None
        if system_prompt:
            fixed = [{"role": "system", "content": system_prompt}] + fixed
        return fixed

    inst_f = mapping.get("instruction_field", "instruction")
    inp_f = mapping.get("input_field", "input")
    out_f = mapping.get("output_field", "output")

    instruction = normalize_content(record.get(inst_f))
    output = normalize_content(record.get(out_f))
    input_text = normalize_content(record.get(inp_f))
    if not output:
        return None

    if instruction:
        user_text = instruction
        if input_text:
            user_text = f"{instruction}\n\n{input_text}"
    elif input_text:
        user_text = input_text
    else:
        return None
    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": output},
    ]
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages
    return messages


def resolve_system_prompt(
    inp_cfg: Dict[str, Any],
    stage_prompt: Optional[str],
    rng: random.Random,
) -> str:
    prompt = inp_cfg.get("system_prompt", stage_prompt)
    prompt = normalize_content(prompt)
    if not prompt:
        prompt = rng.choice(DEFAULT_SYSTEM_PROMPTS)
    return prompt


def write_parquet_shards(
    inputs: List[Dict[str, Any]],
    out_dir: str,
    shard_rows: int,
    val_rows: int,
    seed: int,
) -> None:
    import random

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    shard_id = 0
    train_buf: List[str] = []
    val_buf: List[str] = []
    total_seen = 0
    total_kept = 0

    def flush_train() -> None:
        nonlocal shard_id, train_buf
        if not train_buf:
            return
        table = pa.Table.from_arrays([pa.array(train_buf, pa.string())], names=["text"])
        out_path = Path(out_dir) / f"shard_{shard_id:05d}.parquet"
        pq.write_table(table, out_path)
        shard_id += 1
        train_buf = []

    def add_train(text: str) -> None:
        train_buf.append(text)
        if len(train_buf) >= shard_rows:
            flush_train()

    for inp in inputs:
        it = build_input_iterator(inp)
        text_field = inp.get("text_field", "text")
        source_name = inp.get("name", inp.get("path", "input"))
        for rec in tqdm(it, desc=f"base: {source_name}", unit="rows"):
            total_seen += 1
            text = norm_text(rec.get(text_field))
            if not text:
                continue
            total_kept += 1
            if val_rows > 0:
                if len(val_buf) < val_rows:
                    val_buf.append(text)
                    continue
                # reservoir sampling to keep val unbiased
                j = rng.randint(1, total_kept)
                if j <= val_rows:
                    replaced = val_buf[j - 1]
                    val_buf[j - 1] = text
                    add_train(replaced)
                else:
                    add_train(text)
            else:
                add_train(text)

    flush_train()
    if val_rows > 0:
        table = pa.Table.from_arrays([pa.array(val_buf, pa.string())], names=["text"])
        pq.write_table(table, Path(out_dir) / "zz_val.parquet")

    print(f"[base] seen={total_seen:,} kept={total_kept:,} shards={shard_id:,} out={out_dir}")


def write_chat_jsonl(
    inputs: List[Dict[str, Any]],
    out_dir: str,
    out_file: str,
    mapping_default: Dict[str, str],
    system_prompt: Optional[str],
) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / out_file
    kept = 0
    dropped = 0
    rng = random.Random(int(os.environ.get("SEED", "42")))

    with open(out_path, "w", encoding="utf-8") as f:
        for inp in inputs:
            it = build_input_iterator(inp)
            source_name = inp.get("name", inp.get("path", "input"))
            mapping = dict(mapping_default)
            mapping.update(inp.get("mapping", {}) or {})
            per_input_system_prompt = resolve_system_prompt(inp, system_prompt, rng)
            for rec in tqdm(it, desc=f"chat: {source_name}", unit="rows"):
                msgs = to_messages(rec, mapping=mapping, system_prompt=per_input_system_prompt)
                if not msgs:
                    dropped += 1
                    continue
                f.write(json.dumps(msgs, ensure_ascii=False) + "\n")
                kept += 1

    print(f"[chat] kept={kept:,} dropped={dropped:,} out={out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=["base", "mid", "sft"])
    args = ap.parse_args()

    cfg = read_config(args.config)
    stage_cfg = cfg["stages"][args.stage]

    if args.stage == "base":
        write_parquet_shards(
            inputs=stage_cfg["inputs"],
            out_dir=stage_cfg["out_dir"],
            shard_rows=int(stage_cfg.get("shard_rows", 20000)),
            val_rows=int(stage_cfg.get("val_rows", 2000)),
            seed=int(stage_cfg.get("seed", 42)),
        )
    else:
        write_chat_jsonl(
            inputs=stage_cfg["inputs"],
            out_dir=stage_cfg["out_dir"],
            out_file=stage_cfg.get("out_file", f"jp_{args.stage}.jsonl"),
            mapping_default=stage_cfg.get("mapping", {}) or {},
            system_prompt=stage_cfg.get("system_prompt"),
        )


if __name__ == "__main__":
    main()
