import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import List, Tuple

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Default config (overridable through CLI)
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Coder-3B"
DEFAULT_CORPUS_DIR = "data/code_corpus"
DEFAULT_METRICS_FILE = "results/track_a_baseline.json"
DEFAULT_BLOCK_SIZE = 512
DEFAULT_VAL_RATIO = 0.10
DEFAULT_RANDOM_SEED = 42
DEFAULT_BATCH_SIZE = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure baseline perplexity on HyperSwitch Rust corpus."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--metrics-file", default=DEFAULT_METRICS_FILE)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer only from local Hugging Face cache or local path.",
    )
    return parser.parse_args()


def load_code_files(corpus_dir: str, comment_prefix: str) -> List[Tuple[str, str]]:
    """Load all files recursively as (relative_path, prefixed_content)."""
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    files: List[Tuple[str, str]] = []
    print(f"[INFO] Scanning files under: {corpus_path}")

    for code_file in sorted(corpus_path.glob("*")):
        if code_file.is_file():
            try:
                content = code_file.read_text(encoding="utf-8", errors="ignore")
                prefixed = f"{comment_prefix} FILE: {code_file.name}\n{content}"
                files.append((code_file.name, prefixed))
            except Exception as exc:
                print(f"[WARN] Failed to read {code_file}: {exc}")
                continue

    print(f"[INFO] Loaded {len(files)} files.")
    return files


def tokenize_samples(files: List[Tuple[str, str]], tokenizer, block_size: int) -> Dataset:
    """Tokenize each file individually as a semantic sample."""
    print(f"[INFO] Tokenizing {len(files)} samples individually...")
    tokenized_samples = []
    for _, content in files:
        tokens = tokenizer(
            content,
            truncation=True,
            max_length=block_size,
            padding=False,
            add_special_tokens=True
        )["input_ids"]
        tokenized_samples.append({"input_ids": tokens})

    print(f"[INFO] Created {len(tokenized_samples)} semantic samples.")
    return Dataset.from_list(tokenized_samples)


@torch.no_grad()
def compute_perplexity(model, dataset: Dataset, batch_size: int = 4) -> float:
    """Compute token-weighted perplexity using semantic chunks."""
    model.eval()
    device = next(model.parameters()).device
    
    # Use DataCollator for padding
    from transformers import DataCollatorForLanguageModeling
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME, local_files_only=True)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    total_loss = 0.0
    total_tokens = 0
    print(f"[INFO] Computing perplexity on {len(dataset)} semantic samples...")

    for i in range(0, len(dataset), batch_size):
        batch_samples = [dataset[i+j] for j in range(min(batch_size, len(dataset)-i))]
        batch = collator(batch_samples).to(device)
        
        outputs = model(**batch)
        
        labels = batch["labels"]
        valid_tokens = int((labels != -100).sum().item())
        if valid_tokens == 0:
            continue

        total_loss += float(outputs.loss.item()) * valid_tokens
        total_tokens += valid_tokens

    if total_tokens == 0:
        raise ValueError("No valid evaluation tokens found; cannot compute perplexity.")

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[INFO] Device: {device}")

    print(f"[INFO] Loading tokenizer/model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    ).to(device)

    comment_prefix = "//" if "hyperswitch" in args.corpus_dir else "#"
    all_files = load_code_files(args.corpus_dir, comment_prefix)
    if not all_files:
        raise ValueError(f"No files found in: {args.corpus_dir}")

    random.shuffle(all_files)
    val_count = max(1, int(len(all_files) * args.val_ratio))
    val_files = all_files[:val_count]
    train_count = len(all_files) - val_count

    print(f"[INFO] File split: {train_count} train files, {val_count} validation files")
    print(f"[INFO] Validation files: {[name for name, _ in val_files]}")

    val_dataset = tokenize_samples(val_files, tokenizer, args.block_size)

    baseline_ppl = compute_perplexity(model, val_dataset, args.batch_size)

    print("\n" + "=" * 50)
    print(f"BASELINE PERPLEXITY: {baseline_ppl:.4f}")
    print("=" * 50)

    os.makedirs(Path(args.metrics_file).parent, exist_ok=True)
    metrics = {
        "track": "A – Extended Pre-Training",
        "model": args.model_name,
        "language": "rust",
        "corpus_dir": args.corpus_dir,
        "corpus_files": len(all_files),
        "train_files": train_count,
        "val_files": val_count,
        "val_file_names": [name for name, _ in val_files],
        "block_size": args.block_size,
        "val_chunks": len(val_dataset),
        "baseline_perplexity": round(baseline_ppl, 4),
    }
    with open(args.metrics_file, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"[INFO] Baseline metrics saved to: {args.metrics_file}")


if __name__ == "__main__":
    main()
