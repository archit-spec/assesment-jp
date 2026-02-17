"""
Upload Dataset to Hugging Face
==============================
Uploads the local code corpus (data/code_corpus) to the Hugging Face Hub.
Requires 'huggingface_hub' and 'datasets' libraries.
Run 'huggingface-cli login' first if not authenticated.
"""

import os
from pathlib import Path
from datasets import Dataset
from huggingface_hub import HfApi, create_repo

# ── Config ──────────────────────────────────────────────────────────
REPO_ID = "archit11/verl-code-corpus"
CORPUS_DIR = "data/code_corpus"
IS_PRIVATE = False

def load_files_list(corpus_dir: str):
    """Load all files into a dictionary for Dataset.from_dict."""
    path = Path(corpus_dir)
    files = sorted(path.glob("*.py"))
    print(f"[INFO] Found {len(files)} files to upload.")
    
    texts = []
    names = []
    
    for f in files:
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
            texts.append(content)
            names.append(f.name)
        except Exception as e:
            print(f"[WARN] Skipping {f.name}: {e}")
            
    return {"text": texts, "file_name": names}

def main():
    print(f"[INFO] Preparing dataset for {REPO_ID}...")
    
    # 1. Load data into memory (avoids generator caching)
    data = load_files_list(CORPUS_DIR)
    ds = Dataset.from_dict(data)
    print(f"[INFO] Created dataset with {len(ds)} rows.")
    
    # 2. Create repo if it doesn't exist
    api = HfApi()
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=IS_PRIVATE, exist_ok=True)
        print(f"[INFO] Repo {REPO_ID} is ready.")
    except Exception as e:
        print(f"[ERROR] Failed to create repo: {e}")
        print("Tip: Run 'huggingface-cli login' in your terminal.")
        return

    # 3. Push to hub
    print("[INFO] Pushing to Hugging Face Hub...")
    ds.push_to_hub(REPO_ID)
    print("\n[SUCCESS] Dataset uploaded!")
    print(f"Propagate this ID to other scripts: {REPO_ID}")

if __name__ == "__main__":
    main()
