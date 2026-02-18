"""Upload the embedding dataset to HuggingFace Hub."""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

from huggingface_hub import HfApi

token = os.getenv("HF_TOKEN", "").strip().strip('"')
hf_repo = "archit11/code-embedding-dataset"

if not token:
    print("[ERROR] HF_TOKEN not found in .env")
    sys.exit(1)

print(f"[HF] Token: {token[:12]}...")
print(f"[HF] Repo : {hf_repo}")

api = HfApi(token=token)
api.create_repo(repo_id=hf_repo, repo_type="dataset", exist_ok=True)
print(f"[HF] Repo created/verified")

files = [
    "data/embedding_dataset/all.jsonl",
    "data/embedding_dataset/train.jsonl",
    "data/embedding_dataset/test.jsonl",
    "data/embedding_dataset/README.md",
]

for f in files:
    p = Path(f)
    if p.exists():
        print(f"[HF] Uploading {f} ({p.stat().st_size} bytes)...")
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=p.name,
            repo_id=hf_repo,
            repo_type="dataset",
            token=token,
        )
        print(f"[HF] ✓ {p.name}")
    else:
        print(f"[WARN] Not found: {f}")

print(f"\n[HF] Done! https://huggingface.co/datasets/{hf_repo}")
