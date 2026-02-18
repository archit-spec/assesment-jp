#!/usr/bin/env python3
"""Package and upload Track A artifacts (model + dataset) to Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload Track A model and dataset to HF Hub")
    parser.add_argument(
        "--metrics-file",
        default="results/track_a_verl_metrics_lr1e4_fileholdout_bs8_clip05.json",
        help="Metrics JSON from the completed training run.",
    )
    parser.add_argument(
        "--model-dir",
        default="results/track_a_verl_lr1e4_fileholdout_bs8_clip05/model",
        help="Directory containing adapter/tokenizer artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/hf_verl_dataset_package",
        help="Local folder to write dataset package files before upload.",
    )
    parser.add_argument("--dataset-repo-id", default=None, help="HF dataset repo id (username/name).")
    parser.add_argument("--model-repo-id", default=None, help="HF model repo id (username/name).")
    parser.add_argument("--private", action="store_true", help="Create private HF repos.")
    return parser.parse_args()


def load_metrics(metrics_file: Path) -> Dict:
    with metrics_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_split_rows(metrics: Dict) -> Dict[str, List[Dict[str, str]]]:
    corpus_dir = Path(metrics["corpus_dir"])
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus dir not found: {corpus_dir}")

    exts = set(metrics.get("extensions") or [])
    all_files = sorted(
        p for p in corpus_dir.glob("*") if p.is_file() and (not exts or p.suffix in exts)
    )

    val_names = set(metrics.get("val_file_names") or [])
    test_names = set(metrics.get("test_file_names") or [])

    rows: Dict[str, List[Dict[str, str]]] = {"train": [], "validation": [], "test": []}
    for file_path in all_files:
        file_name = file_path.name
        if file_name in val_names:
            split = "validation"
        elif file_name in test_names:
            split = "test"
        else:
            split = "train"

        text = file_path.read_text(encoding="utf-8", errors="ignore")
        rows[split].append({"file_name": file_name, "text": text})

    return rows


def write_jsonl(rows: List[Dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_dataset_card(
    readme_path: Path,
    dataset_repo_id: str,
    model_repo_id: str,
    metrics: Dict,
    split_rows: Dict[str, List[Dict[str, str]]],
) -> None:
    repo_name = str(metrics.get("repo_name", "repository"))
    exts = metrics.get("extensions", [])
    language_tag = "python" if ".py" in exts else "rust" if ".rs" in exts else "code"
    seq_curr = metrics.get("sequence_curriculum", [metrics.get("block_size")])
    baseline_val = metrics.get("baseline_validation_perplexity")
    baseline_test = metrics.get("baseline_test_perplexity")
    baseline_primary = metrics.get("baseline_perplexity")
    post_val = metrics.get("post_training_validation_perplexity")
    post_test = metrics.get("post_training_test_perplexity")
    post_primary = metrics.get("post_training_perplexity")

    eval_lines = []
    if baseline_val is not None and baseline_test is not None:
        eval_lines.append(
            f"- Baseline perplexity (val/test): {baseline_val:.4f} / {baseline_test:.4f}"
        )
    elif baseline_primary is not None:
        eval_lines.append(f"- Baseline perplexity: {baseline_primary:.4f}")

    if post_val is not None and post_test is not None:
        eval_lines.append(
            f"- Post-training perplexity (val/test): {post_val:.4f} / {post_test:.4f}"
        )
    elif post_primary is not None:
        eval_lines.append(f"- Post-training perplexity: {post_primary:.4f}")

    eval_text = "\n".join(eval_lines) if eval_lines else "- Perplexity metrics not available in source JSON."

    content = f"""---
language:
- en
license: apache-2.0
task_categories:
- text-generation
- fill-mask
tags:
- code
- {language_tag}
- {repo_name}
- repo-specific-finetuning
pretty_name: {repo_name} Code Corpus (Track A Split)
size_categories:
- n<1K
---

# {dataset_repo_id}

Repository-specific code corpus extracted from `{repo_name}` and split by file for training/evaluation.

## What is in this dataset

- Source corpus: `{metrics['corpus_dir']}`
- Total files: {metrics['corpus_files']}
- Train files: {len(split_rows['train'])}
- Validation files: {len(split_rows['validation'])}
- Test files: {len(split_rows['test'])}
- File type filter: {", ".join(metrics.get("extensions", []))}
- Split mode: `{metrics.get('split_mode', 'file')}` (file-level holdout)

Each row has:

- `file_name`: flattened source file name
- `text`: full file contents

## Training context

This dataset was used for extended pretraining of:

- Model repo: `https://huggingface.co/{model_repo_id}`
- Base model: `{metrics['model']}`
- Sequence curriculum: {seq_curr}
- Learning rate: {metrics.get('learning_rate', 'n/a')}
- Batch size: {metrics['batch_size']}

Evaluation from this run:

{eval_text}

## Load with datasets

```python
from datasets import load_dataset

ds = load_dataset("{dataset_repo_id}")
print(ds)
print(ds["train"][0]["file_name"])
```
"""
    readme_path.write_text(content, encoding="utf-8")


def write_model_card(readme_path: Path, model_repo_id: str, dataset_repo_id: str, metrics: Dict) -> None:
    repo_name = str(metrics.get("repo_name", "repository"))
    exts = metrics.get("extensions", [])
    language_tag = "python" if ".py" in exts else "rust" if ".rs" in exts else "code"
    seq_curr = metrics.get("sequence_curriculum", [metrics.get("block_size")])
    baseline_val = metrics.get("baseline_validation_perplexity")
    baseline_test = metrics.get("baseline_test_perplexity")
    baseline_primary = metrics.get("baseline_perplexity")
    post_val = metrics.get("post_training_validation_perplexity")
    post_test = metrics.get("post_training_test_perplexity")
    post_primary = metrics.get("post_training_perplexity")
    ppl_reduction = metrics.get("perplexity_reduction")
    improve_pct = metrics.get("improvement_percent")

    eval_lines = []
    if baseline_val is not None:
        eval_lines.append(f"- Baseline perplexity (validation): {baseline_val:.4f}")
    if baseline_test is not None:
        eval_lines.append(f"- Baseline perplexity (test): {baseline_test:.4f}")
    if baseline_primary is not None:
        eval_lines.append(f"- Baseline perplexity (primary): {baseline_primary:.4f}")
    if post_val is not None:
        eval_lines.append(f"- Post-training perplexity (validation): {post_val:.4f}")
    if post_test is not None:
        eval_lines.append(f"- Post-training perplexity (test): {post_test:.4f}")
    if post_primary is not None:
        eval_lines.append(f"- Post-training perplexity (primary): {post_primary:.4f}")
    if ppl_reduction is not None and improve_pct is not None:
        eval_lines.append(f"- Perplexity reduction: {ppl_reduction:.4f} ({improve_pct:.2f}%)")
    eval_text = "\n".join(eval_lines) if eval_lines else "- Metrics not available in source JSON."

    content = f"""---
base_model: Qwen/Qwen2.5-Coder-3B
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
tags:
- lora
- peft
- code
- {language_tag}
- {repo_name}
---

# {model_repo_id}

LoRA adapter trained for repository-specific extended pretraining on `{repo_name}` source code.

## Model details

- Base model: `Qwen/Qwen2.5-Coder-3B`
- Fine-tuning method: LoRA (`r={metrics['lora_r']}`)
- Training corpus: `https://huggingface.co/datasets/{dataset_repo_id}`
- Split strategy: file-level train/validation/test split
- Sequence curriculum: {seq_curr}
- Effective learning rate: {metrics.get('learning_rate', 'n/a')}
- Batch size: {metrics['batch_size']}
- Gradient accumulation: {metrics['grad_accum_steps']}

## Evaluation summary

{eval_text}

## Usage

This repo stores adapter weights and tokenizer artifacts. Load it with PEFT on top of the base model.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = "Qwen/Qwen2.5-Coder-3B"
adapter = "{model_repo_id}"

tokenizer = AutoTokenizer.from_pretrained(adapter)
model = AutoModelForCausalLM.from_pretrained(base, trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter)
```
"""
    readme_path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    load_dotenv(".env")

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing. Set it in .env or environment.")

    metrics_file = Path(args.metrics_file)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    metrics = load_metrics(metrics_file)
    split_rows = build_split_rows(metrics)

    api = HfApi(token=token)
    username = api.whoami()["name"]

    repo_slug = str(metrics.get("repo_name", "repo")).lower().replace("_", "-")
    dataset_repo_id = args.dataset_repo_id or f"{username}/{repo_slug}-code-corpus-track-a"
    model_repo_id = args.model_repo_id or f"{username}/qwen2.5-coder-3b-{repo_slug}-track-a-lora"

    print(f"[HF] User: {username}")
    print(f"[HF] Dataset repo: {dataset_repo_id}")
    print(f"[HF] Model repo: {model_repo_id}")

    # Write local packaged files.
    split_manifest = {
        "train": [row["file_name"] for row in split_rows["train"]],
        "validation": [row["file_name"] for row in split_rows["validation"]],
        "test": [row["file_name"] for row in split_rows["test"]],
        "metrics_file": str(metrics_file),
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8"
    )
    (output_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    write_jsonl(split_rows["train"], output_dir / "train.jsonl")
    write_jsonl(split_rows["validation"], output_dir / "validation.jsonl")
    write_jsonl(split_rows["test"], output_dir / "test.jsonl")

    write_dataset_card(
        output_dir / "README.md",
        dataset_repo_id=dataset_repo_id,
        model_repo_id=model_repo_id,
        metrics=metrics,
        split_rows=split_rows,
    )
    write_model_card(
        model_dir / "README.md",
        model_repo_id=model_repo_id,
        dataset_repo_id=dataset_repo_id,
        metrics=metrics,
    )

    print(
        "[LOCAL] Split sizes:",
        {k: len(v) for k, v in split_rows.items()},
    )

    # Upload dataset repo.
    api.create_repo(repo_id=dataset_repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    ds_splits = {}
    for split in ["train", "validation", "test"]:
        if split_rows[split]:
            ds_splits[split] = Dataset.from_list(split_rows[split])
    ds = DatasetDict(ds_splits)
    ds.push_to_hub(dataset_repo_id, token=token, private=args.private, commit_message="Upload split dataset")

    for file_name in ["README.md", "split_manifest.json", "training_metrics.json", "train.jsonl", "validation.jsonl", "test.jsonl"]:
        api.upload_file(
            path_or_fileobj=str(output_dir / file_name),
            path_in_repo=file_name,
            repo_id=dataset_repo_id,
            repo_type="dataset",
        )

    # Upload model repo.
    api.create_repo(repo_id=model_repo_id, private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=model_repo_id,
        repo_type="model",
        commit_message="Upload Track A Verl LoRA adapter and tokenizer",
    )

    print(f"[DONE] Dataset: https://huggingface.co/datasets/{dataset_repo_id}")
    print(f"[DONE] Model:   https://huggingface.co/{model_repo_id}")


if __name__ == "__main__":
    main()
