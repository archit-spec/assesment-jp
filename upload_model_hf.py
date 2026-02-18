"""
Upload Track B fine-tuned model and dataset to Hugging Face.

Usage:
    python upload_model_hf.py
    python upload_model_hf.py --model-dir results/track_b_sft --model-repo archit11/track_b_sft_model
"""
import argparse, os, json
from huggingface_hub import HfApi

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="results/track_b_sft")
    p.add_argument("--model-repo", default="archit11/track_b_sft_model")
    p.add_argument("--dataset-dir", default="data")
    p.add_argument("--dataset-repo", default="archit11/track_b_sft")
    p.add_argument("--skip-model", action="store_true")
    p.add_argument("--skip-dataset", action="store_true")
    return p.parse_args()

def upload_model(api: HfApi, model_dir: str, repo_id: str):
    print(f"\n[MODEL] Uploading {model_dir} → {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        repo_type="model",
    )
    # Write a model card
    card = f"""---
base_model: Qwen/Qwen2.5-Coder-1.5B
tags:
  - lora
  - sft
  - code
  - python
  - instruction-tuning
license: apache-2.0
---

# Track B SFT – Qwen2.5-Coder-1.5B + LoRA

Fine-tuned on ~250 synthetic coding instruction pairs generated from the [verl](https://github.com/volcengine/verl) corpus.

## Results

| Metric | Baseline | Post-SFT | Δ |
|--------|----------|----------|---|
| pass@1 | 0.565 | **0.804** | +0.239 |
| pass@3 | 0.783 | 0.848 | +0.065 |

## Training

- **Base model:** `Qwen/Qwen2.5-Coder-1.5B`
- **Method:** LoRA (r=16, alpha=32)
- **Data:** `archit11/track_b_sft` (~257 train examples)
- **Epochs:** 3, **LR:** 2e-4, **Hardware:** T4 GPU

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")
model = PeftModel.from_pretrained(base, "{repo_id}").merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```
"""
    with open("/tmp/README.md", "w") as f:
        f.write(card)
    api.upload_file(
        path_or_fileobj="/tmp/README.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"  ✓ Model uploaded → https://huggingface.co/{repo_id}")

def upload_dataset(api: HfApi, data_dir: str, repo_id: str):
    print(f"\n[DATASET] Uploading dataset → {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)
    for fname in ["sft_trackb_train.json", "sft_trackb_test.json", "sft_trackb_data_card.json"]:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=fname,
                repo_id=repo_id,
                repo_type="dataset",
            )
            print(f"  ✓ {fname}")
    print(f"  ✓ Dataset uploaded → https://huggingface.co/datasets/{repo_id}")

def main():
    args = parse_args()
    api = HfApi()

    if not args.skip_model:
        upload_model(api, args.model_dir, args.model_repo)

    if not args.skip_dataset:
        upload_dataset(api, args.dataset_dir, args.dataset_repo)

    print("\nAll uploads complete!")

if __name__ == "__main__":
    main()
