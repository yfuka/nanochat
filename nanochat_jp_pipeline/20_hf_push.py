#!/usr/bin/env python3
import os, argparse
from huggingface_hub import HfApi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--local_path", required=True)
    ap.add_argument("--repo_type", default="model", choices=["model","dataset","space"])
    ap.add_argument("--commit_message", default="upload")
    ap.add_argument("--multi_commits", action="store_true")
    args = ap.parse_args()

    # HF_TOKEN envでOK :contentReference[oaicite:28]{index=28}
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, exist_ok=True)

    if os.path.isdir(args.local_path):
        upload_kwargs = dict(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            folder_path=args.local_path,
            path_in_repo=".",
            commit_message=args.commit_message,
        )
        if args.multi_commits:
            upload_kwargs["multi_commits"] = True  # newer huggingface_hub only
        try:
            api.upload_folder(**upload_kwargs)
        except TypeError as exc:
            if "multi_commits" not in str(exc):
                raise
            upload_kwargs.pop("multi_commits", None)
            print("[push] multi_commits unsupported in this huggingface_hub; retrying without it.")
            api.upload_folder(**upload_kwargs)
    else:
        # 単一ファイルも扱えるように（dataset repoにjsonlだけ上げる等）
        api.upload_file(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            path_or_fileobj=args.local_path,
            path_in_repo=os.path.basename(args.local_path),
            commit_message=args.commit_message,
        )

    print("[push] done:", args.repo_id, "<-", args.local_path)

if __name__ == "__main__":
    main()
