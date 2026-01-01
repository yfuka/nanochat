#!/usr/bin/env python3
import os, argparse
from huggingface_hub import snapshot_download

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--local_dir", required=True)
    ap.add_argument("--repo_type", default="model", choices=["model","dataset","space"])
    args = ap.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)
    # snapshot_downloadで復元 :contentReference[oaicite:30]{index=30}
    snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        local_dir=args.local_dir,
        local_dir_use_symlinks=False,  # 直下に実体化
    )
    print("[pull] done:", args.repo_id, "->", args.local_dir)

if __name__ == "__main__":
    main()
