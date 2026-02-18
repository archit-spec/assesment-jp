---
language:
- en
license: mit
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
- retrieval
- hyperswitch
size_categories:
- n<1K
---

# Code-to-Doc Embedding Dataset

AI-generated code documentation pairs for training code embedding / retrieval models.

## Dataset Description

Each record contains a **code anchor** (real production code) paired with:
- **positive**: A rich natural-language documentation of what the code does
- **queries**: 4 natural-language search queries a developer might use to find this code
- **label**: A short semantic label (3-8 words)

This dataset is designed for training **bi-encoder** embedding models (e.g., with InfoNCE / contrastive loss)
where `anchor` = code, `positive` = documentation, and `queries` can serve as additional positives.

## Sources

| Repo | Language | Records |
|------|----------|---------|
| juspay/hyperswitch | Rust | 250 |


**Total**: 250 records (212 train / 38 test)

## Schema

```json
{
  "anchor":    "<code snippet, up to 3000 chars>",
  "positive":  "<150-300 word natural language documentation>",
  "queries":   ["query 1", "query 2", "query 3", "query 4"],
  "label":     "short semantic label",
  "repo":      "owner/repo",
  "language":  "Python | Rust",
  "filename":  "source_filename.py",
  "num_lines": 42,
  "split":     "train | test"
}
```

## Generation

- **Model**: Provider-specific (`qwen/qwen3.5` on OpenRouter, `GLM-5` on Modal)
- **Method**: LLM-generated documentation + query variants per file
- **Temperature**: 0.3 (documentation), deterministic
- **Code truncation**: 5000 chars max input, 3000 chars max anchor

## Usage

```python
from datasets import load_dataset

ds = load_dataset("archit11/assesment_embeddings")

# For contrastive training (anchor=code, positive=doc)
for example in ds["train"]:
    code = example["anchor"]
    doc  = example["positive"]
    queries = example["queries"]  # additional positives

# For retrieval evaluation
for example in ds["test"]:
    query = example["queries"][0]
    code  = example["anchor"]
```

## Training Tips

- Use `anchor` as the **code encoder** input and `positive` as the **text encoder** input
- `queries` can be used as **hard positives** or for query augmentation
- For hard negatives: sample other records from the same `language` or `repo`
