"""
Fix archit11/assesment_embeddings_new:
- Remove train_hard_negatives.json (incompatible schema causing load errors)
- Keep train.jsonl / test.jsonl with consistent schema
- Add a clean README with proper dataset card
"""

from datasets import load_dataset
from huggingface_hub import HfApi, login

REPO_ID = "archit11/assesment_embeddings_new"

def main():
    api = HfApi()

    # Load clean splits
    print("Loading train split...")
    train_ds = load_dataset(REPO_ID, data_files="train.jsonl", split="train")
    print(f"  train: {len(train_ds)} rows | cols: {train_ds.column_names}")

    print("Loading test split...")
    test_ds = load_dataset(REPO_ID, data_files="test.jsonl", split="train")
    print(f"  test:  {len(test_ds)} rows | cols: {test_ds.column_names}")

    # Delete the conflicting file
    print("\nDeleting train_hard_negatives.json from hub...")
    try:
        api.delete_file(
            path_in_repo="train_hard_negatives.json",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="Remove train_hard_negatives.json (incompatible schema)",
        )
        print("  Deleted.")
    except Exception as e:
        print(f"  Could not delete (may not exist): {e}")

    # Also delete all.jsonl to avoid confusion (it's just train+test combined)
    print("Deleting all.jsonl from hub...")
    try:
        api.delete_file(
            path_in_repo="all.jsonl",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="Remove all.jsonl (redundant)",
        )
        print("  Deleted.")
    except Exception as e:
        print(f"  Could not delete: {e}")

    # Push clean splits via push_to_hub
    print("\nPushing fixed train/test splits...")
    from datasets import DatasetDict
    ds_dict = DatasetDict({"train": train_ds, "test": test_ds})
    ds_dict.push_to_hub(REPO_ID, commit_message="Fix dataset: consistent schema, proper train/test splits")
    print("Pushed.")

    # Update README
    readme = """---
license: mit
language:
- en
task_categories:
- text-retrieval
- feature-extraction
task_ids:
- document-retrieval
tags:
- code
- embedding
- code-search
- retrieval
- hyperswitch
size_categories:
- n<1K
---

# Code-to-Doc Embedding Dataset

AI-generated code documentation pairs for training code embedding / retrieval models on the Hyperswitch codebase.

## Dataset Description

Each record contains a code anchor (real production code) paired with:
- **anchor**: Raw source code snippet with file path header
- **positive**: Rich natural-language documentation of what the code does
- **queries**: 4 natural-language search queries a developer might use to find this code
- **label**: Short semantic label (3–8 words)

Designed for training bi-encoder embedding models with `MultipleNegativesRankingLoss`.

## Splits

| Split | Rows |
|-------|------|
| train | """ + str(len(train_ds)) + """ |
| test  | """ + str(len(test_ds)) + """ |

## Columns

| Column | Type | Description |
|--------|------|-------------|
| anchor | string | Source code with path header |
| positive | string | Natural-language documentation |
| queries | list[string] | 4 developer search queries |
| label | string | Semantic label |
| repo | string | Source repository |
| language | string | Programming language |
| filename | string | Source filename |
| symbol | string | Function/class name |
| unit_type | string | function / class / method |
| num_lines | int | Lines of code |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("archit11/assesment_embeddings_new")
train_ds = ds["train"]
test_ds = ds["test"]
```

## Fine-tuned Model

See [`archit11/assesment_qwen3_embedding_06b_e3`](https://huggingface.co/archit11/assesment_qwen3_embedding_06b_e3) — Qwen3-Embedding-0.6B fine-tuned on this dataset.

| Metric | Baseline | Fine-Tuned | Δ |
|--------|----------|------------|---|
| MRR@10 | 0.8840 | 0.9617 | +0.0777 ↑ |
| nDCG@10 | 0.9093 | 0.9710 | +0.0617 ↑ |
| Recall@10 | 0.9870 | 1.0000 | +0.0130 ↑ |
"""

    print("\nUpdating README.md...")
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Update README: point to fine-tuned model, add usage and metrics",
    )
    print("Done.")
    print(f"\nDataset fixed: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
