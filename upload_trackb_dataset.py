#!/usr/bin/env python3
"""Upload Track B SFT dataset files to Hugging Face Hub."""

import argparse
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


DEFAULT_REPO_ID = os.getenv("HF_REPO_ID", "archit11/track_b_sft")
DEFAULT_DATA_DIRS = ("data", "data_trackb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HF dataset repo id (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing sft_trackb_train.json and sft_trackb_test.json",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repo as private (ignored if repo already exists)",
    )
    return parser.parse_args()


def row_count(path: Path) -> int:
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
        return len(content) if isinstance(content, list) else 1
    except Exception:
        return -1


def resolve_data_dir(cli_data_dir: str | None) -> Path | None:
    if cli_data_dir:
        data_dir = Path(cli_data_dir)
        return data_dir if data_dir.exists() else None

    for candidate in DEFAULT_DATA_DIRS:
        data_dir = Path(candidate)
        if data_dir.exists():
            return data_dir
    return None


def hf_error_message(err: HfHubHTTPError) -> str:
    status = getattr(err.response, "status_code", None)
    details = str(err).strip()
    if status == 401:
        return (
            "Unauthorized (401). Check HF_TOKEN permissions and account.\n"
            "Token needs write access to datasets."
        )
    if status == 404:
        return (
            "Repository not found (404). Ensure repo id is correct and accessible.\n"
            "This script creates missing repos, so 404 often means auth scope issue."
        )
    return details


def main() -> int:
    args = parse_args()
    token = os.getenv("HF_TOKEN", "").strip().strip('"').strip("'")
    if not token:
        print("Error: HF_TOKEN not found in environment or .env")
        print("Hint: export HF_TOKEN=hf_xxx or run `huggingface-cli login`.")
        return 1

    data_dir = resolve_data_dir(args.data_dir)
    if data_dir is None:
        print("Error: data directory not found.")
        print("Tried: --data-dir value and defaults: data, data_trackb")
        return 1

    train_path = data_dir / "sft_trackb_train.json"
    test_path = data_dir / "sft_trackb_test.json"
    card_path = data_dir / "sft_trackb_data_card.json"
    if not train_path.exists() or not test_path.exists():
        print(f"Loading data from {data_dir} ...")
        print("Error: sft_trackb_train.json or sft_trackb_test.json not found!")
        return 1

    print(f"Loading data from {data_dir} ...")
    train_rows = row_count(train_path)
    test_rows = row_count(test_path)
    if train_rows >= 0:
        print(f"  Train: {train_rows} rows")
    if test_rows >= 0:
        print(f"  Test:  {test_rows} rows")

    print(f"Pushing to {args.repo_id} ...")
    api = HfApi(token=token)

    try:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
            token=token,
        )
    except HfHubHTTPError as err:
        print(f"Upload failed while creating repo: {hf_error_message(err)}")
        return 1

    uploads = [
        (train_path, "train.json"),
        (test_path, "test.json"),
    ]
    if card_path.exists():
        uploads.append((card_path, "data_card.json"))

    try:
        for local_path, remote_path in uploads:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                repo_id=args.repo_id,
                repo_type="dataset",
                token=token,
            )
            print(f"  Uploaded {remote_path}")
    except HfHubHTTPError as err:
        print(f"Upload failed: {hf_error_message(err)}")
        return 1

    print(f"Done! Check https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
