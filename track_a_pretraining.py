"""
Track A - Extended Pre-Training (Repo-Specific)
===============================================
Runs repo-specific LoRA continued pre-training and reports
baseline vs post-training perplexity.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import List, Tuple

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Coder-3B"
DEFAULT_BLOCK_SIZE = 512
DEFAULT_VAL_RATIO = 0.10
DEFAULT_RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-name", default="repo")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument(
        "--extensions",
        default=".rs,.py",
        help="Comma-separated file extensions to include (e.g. .rs or .py).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics-file", required=True)

    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument(
        "--sequence-curriculum",
        default="",
        help="Comma-separated sequence lengths for staged training (e.g. 512,1024,1536).",
    )
    parser.add_argument("--eval-block-size", type=int, default=0, help="0 means final curriculum block size.")
    parser.add_argument("--split-mode", choices=["file", "chunk"], default="file")
    parser.add_argument("--eval-split", choices=["val", "test"], default="val")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)

    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=0, help="0 means same as --batch-size")
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    parser.add_argument("--max-files", type=int, default=0, help="0 means all files.")
    parser.add_argument("--max-train-chunks", type=int, default=0, help="0 means all chunks.")
    parser.add_argument("--max-val-chunks", type=int, default=0, help="0 means all chunks.")
    parser.add_argument("--max-test-chunks", type=int, default=0, help="0 means all chunks.")

    parser.add_argument(
        "--report-to",
        default="none",
        help="Comma-separated integrations for Trainer reporting (e.g. wandb,tensorboard). Use 'none' to disable.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--max-vram-gb",
        type=float,
        default=0.0,
        help="Best-effort per-process CUDA memory cap. 0 disables capping.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce VRAM usage.",
    )
    return parser.parse_args()


def parse_sequence_curriculum(block_size: int, curriculum_raw: str) -> List[int]:
    if not curriculum_raw.strip():
        return [block_size]

    values: List[int] = []
    for part in curriculum_raw.split(","):
        item = part.strip()
        if not item:
            continue
        size = int(item)
        if size <= 0:
            raise ValueError(f"Invalid sequence length in curriculum: {size}")
        values.append(size)

    if not values:
        return [block_size]

    # Preserve order, remove duplicates.
    deduped: List[int] = []
    seen = set()
    for val in values:
        if val in seen:
            continue
        seen.add(val)
        deduped.append(val)
    return deduped


def parse_report_to(report_to_raw: str) -> List[str]:
    raw = report_to_raw.strip().lower()
    if not raw or raw == "none":
        return []
    return [item.strip() for item in report_to_raw.split(",") if item.strip()]


def path_prefix_for_file(file_name: str) -> str:
    if file_name.endswith(".py"):
        return "#"
    return "//"


def load_code_files(corpus_dir: str, extensions: List[str], max_files: int) -> List[Tuple[str, str]]:
    files: List[Tuple[str, str]] = []
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    for ext in extensions:
        pattern = f"*{ext}" if not ext.startswith(".") else f"*{ext}"
        for code_file in sorted(corpus_path.rglob(pattern)):
            if not code_file.is_file():
                continue
            rel_name = str(code_file.relative_to(corpus_path))
            content = code_file.read_text(encoding="utf-8", errors="ignore")
            if not content.strip():
                continue
            prefix = path_prefix_for_file(rel_name)
            files.append((rel_name, f"{prefix} FILE: {rel_name}\n{content}"))

    # dedupe while preserving order
    seen = set()
    deduped: List[Tuple[str, str]] = []
    for name, content in files:
        if name in seen:
            continue
        seen.add(name)
        deduped.append((name, content))

    if max_files > 0:
        deduped = deduped[:max_files]

    print(f"[INFO] Loaded {len(deduped)} files from {corpus_dir}")
    return deduped


def tokenize_and_chunk(text: str, tokenizer, block_size: int) -> Dataset:
    tokenized = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    input_ids = tokenized["input_ids"]
    print(f"[INFO] Total tokens: {len(input_ids):,}")

    if len(input_ids) < block_size:
        raise ValueError(
            f"Not enough tokens ({len(input_ids)}) for block size {block_size}. "
            "Increase data or lower --block-size."
        )

    chunks = []
    for i in range(0, len(input_ids) - block_size + 1, block_size):
        chunk = input_ids[i : i + block_size]
        chunks.append({"input_ids": chunk})

    print(f"[INFO] Created {len(chunks)} chunks of {block_size} tokens")
    return Dataset.from_list(chunks)


def maybe_trim_dataset(dataset: Dataset, limit: int, seed: int, name: str) -> Dataset:
    if limit <= 0 or len(dataset) <= limit:
        return dataset
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = sorted(indices[:limit])
    trimmed = dataset.select(indices)
    print(f"[INFO] Trimmed {name} dataset from {len(dataset)} to {len(trimmed)} chunks")
    return trimmed


def split_counts(total: int, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    if total < 3:
        raise ValueError(f"Need at least 3 items to create train/val/test splits, got {total}.")

    val_count = max(1, int(total * val_ratio))
    test_count = max(1, int(total * test_ratio))

    while val_count + test_count >= total:
        if val_count >= test_count and val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            raise ValueError(
                f"Invalid split with total={total}, val_ratio={val_ratio}, test_ratio={test_ratio}."
            )

    train_count = total - val_count - test_count
    return train_count, val_count, test_count


def split_files(
    files: List[Tuple[str, str]], val_ratio: float, test_ratio: float
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    train_count, val_count, test_count = split_counts(len(files), val_ratio, test_ratio)

    test_files = files[:test_count]
    val_files = files[test_count : test_count + val_count]
    train_files = files[test_count + val_count :]

    assert len(train_files) == train_count
    return train_files, val_files, test_files


def split_chunk_dataset(
    dataset: Dataset, val_ratio: float, test_ratio: float, seed: int, name: str
) -> Tuple[Dataset, Dataset, Dataset]:
    train_count, val_count, test_count = split_counts(len(dataset), val_ratio, test_ratio)

    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    test_idx = sorted(indices[:test_count])
    val_idx = sorted(indices[test_count : test_count + val_count])
    train_idx = sorted(indices[test_count + val_count :])

    train_ds = dataset.select(train_idx)
    val_ds = dataset.select(val_idx)
    test_ds = dataset.select(test_idx)
    print(
        f"[INFO] {name} split chunks - train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}"
    )

    assert len(train_ds) == train_count
    return train_ds, val_ds, test_ds


@torch.no_grad()
def compute_perplexity(model, dataset: Dataset, batch_size: int = 4) -> float:
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    total_tokens = 0
    for i in range(0, len(dataset), batch_size):
        batch_items = dataset[i : i + batch_size]
        input_ids = torch.tensor(batch_items["input_ids"], device=device)
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        num_tokens = input_ids.numel()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    if total_tokens == 0:
        raise ValueError("No evaluation tokens available for perplexity.")

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(Path(args.metrics_file).parent, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    if device == "cuda" and args.max_vram_gb > 0:
        try:
            dev_idx = torch.cuda.current_device()
            total_gb = torch.cuda.get_device_properties(dev_idx).total_memory / (1024**3)
            fraction = min(1.0, max(0.01, args.max_vram_gb / total_gb))
            torch.cuda.set_per_process_memory_fraction(fraction, dev_idx)
            print(
                f"[INFO] CUDA memory cap enabled: target={args.max_vram_gb:.2f} GiB, "
                f"device_total={total_gb:.2f} GiB, fraction={fraction:.4f}"
            )
        except Exception as exc:
            print(f"[WARN] Failed to set CUDA memory cap: {exc}")

    extensions = [x.strip() for x in args.extensions.split(",") if x.strip()]
    sequence_curriculum = parse_sequence_curriculum(args.block_size, args.sequence_curriculum)
    final_block_size = sequence_curriculum[-1]
    print(f"[INFO] Extensions: {extensions}")
    print(f"[INFO] Sequence curriculum: {sequence_curriculum}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    lora_target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_target_modules,
    )
    model = get_peft_model(model, lora_config)
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            if hasattr(model, "config"):
                model.config.use_cache = False
            print("[INFO] Gradient checkpointing: enabled")
        except Exception as exc:
            print(f"[WARN] Could not enable gradient checkpointing: {exc}")
    model.print_trainable_parameters()
    model.to(device)

    all_files = load_code_files(args.corpus_dir, extensions=extensions, max_files=args.max_files)
    if not all_files:
        raise ValueError("No files loaded from corpus. Check --corpus-dir and --extensions.")

    random.shuffle(all_files)
    eval_block_size = args.eval_block_size if args.eval_block_size > 0 else final_block_size
    eval_batch_size = args.eval_batch_size if args.eval_batch_size > 0 else max(1, args.batch_size)
    report_to = parse_report_to(args.report_to)
    print(f"[INFO] Split mode: {args.split_mode}")
    print(f"[INFO] Eval split: {args.eval_split}")
    print(f"[INFO] Eval block size: {eval_block_size}")
    print(f"[INFO] Report to: {report_to if report_to else ['none']}")

    train_files: List[Tuple[str, str]] = []
    val_files: List[Tuple[str, str]] = []
    test_files: List[Tuple[str, str]] = []
    train_text = ""
    val_text = ""
    eval_source_text = ""

    if args.split_mode == "file":
        train_files, val_files, test_files = split_files(
            all_files, val_ratio=args.val_ratio, test_ratio=args.test_ratio
        )
        train_text = "\n\n".join(content for _, content in train_files)
        val_text = "\n\n".join(content for _, content in val_files)
        test_text = "\n\n".join(content for _, content in test_files)

        eval_val_dataset = tokenize_and_chunk(val_text, tokenizer, eval_block_size)
        eval_test_dataset = tokenize_and_chunk(test_text, tokenizer, eval_block_size)
    else:
        eval_source_text = "\n\n".join(content for _, content in all_files)
        eval_full_dataset = tokenize_and_chunk(eval_source_text, tokenizer, eval_block_size)
        _, eval_val_dataset, eval_test_dataset = split_chunk_dataset(
            eval_full_dataset,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed + 1,
            name="eval",
        )
        train_files = all_files

    eval_val_dataset = maybe_trim_dataset(eval_val_dataset, args.max_val_chunks, args.seed + 1, "val_eval")
    eval_test_dataset = maybe_trim_dataset(
        eval_test_dataset, args.max_test_chunks, args.seed + 2, "test_eval"
    )

    print("[INFO] Computing baseline perplexity...")
    baseline_val_ppl = compute_perplexity(model, eval_val_dataset, batch_size=eval_batch_size)
    baseline_test_ppl = compute_perplexity(model, eval_test_dataset, batch_size=eval_batch_size)
    baseline_ppl = baseline_val_ppl if args.eval_split == "val" else baseline_test_ppl
    print(
        f"[RESULT] Baseline perplexity ({args.eval_split}): {baseline_ppl:.4f} "
        f"[val={baseline_val_ppl:.4f}, test={baseline_test_ppl:.4f}]"
    )
    stage_summaries = []
    stage_epochs = args.epochs / len(sequence_curriculum)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    for stage_index, stage_block in enumerate(sequence_curriculum, start=1):
        print(
            f"[INFO] Stage {stage_index}/{len(sequence_curriculum)} "
            f"- block_size={stage_block}, epochs={stage_epochs:.4f}"
        )

        if args.split_mode == "file":
            train_dataset = tokenize_and_chunk(train_text, tokenizer, stage_block)
            val_dataset = tokenize_and_chunk(val_text, tokenizer, stage_block)
            test_chunks = 0
        else:
            stage_full_dataset = tokenize_and_chunk(eval_source_text, tokenizer, stage_block)
            train_dataset, val_dataset, test_dataset = split_chunk_dataset(
                stage_full_dataset,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed + 1000 + stage_index,
                name=f"stage_{stage_index}",
            )
            test_chunks = len(test_dataset)

        train_dataset = maybe_trim_dataset(
            train_dataset, args.max_train_chunks, args.seed + stage_index, f"train_stage_{stage_index}"
        )
        val_dataset = maybe_trim_dataset(
            val_dataset, args.max_val_chunks, args.seed + 100 + stage_index, f"val_stage_{stage_index}"
        )
        if args.split_mode != "file":
            test_dataset = maybe_trim_dataset(
                test_dataset, args.max_test_chunks, args.seed + 200 + stage_index, f"test_stage_{stage_index}"
            )
            test_chunks = len(test_dataset)
        print(
            f"[INFO] Stage {stage_index} chunks - train: {len(train_dataset)}, "
            f"val: {len(val_dataset)}, test: {test_chunks}"
        )

        training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir, f"stage_{stage_index}_bs{stage_block}"),
            num_train_epochs=stage_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=args.grad_accum_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            fp16=device == "cuda",
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=10,
            report_to=report_to,
            seed=args.seed + stage_index,
            dataloader_pin_memory=(device == "cuda"),
            do_train=True,
            do_eval=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )

        stage_result = trainer.train()
        stage_eval = trainer.evaluate(eval_dataset=val_dataset)
        stage_eval_loss = float(stage_eval.get("eval_loss", float("nan")))
        print(
            f"[INFO] Stage {stage_index} complete. "
            f"Training loss: {stage_result.training_loss:.4f}, "
            f"Validation loss: {stage_eval_loss:.4f}"
        )
        stage_summaries.append(
            {
                "stage": stage_index,
                "block_size": stage_block,
                "epochs": round(stage_epochs, 6),
                "train_chunks": len(train_dataset),
                "val_chunks": len(val_dataset),
                "test_chunks": test_chunks,
                "training_loss": round(stage_result.training_loss, 4),
                "validation_loss": round(stage_eval_loss, 4),
            }
        )

    mean_stage_loss = (
        sum(x["training_loss"] for x in stage_summaries) / len(stage_summaries)
        if stage_summaries
        else 0.0
    )

    print("[INFO] Computing post-training perplexity...")
    post_val_ppl = compute_perplexity(model, eval_val_dataset, batch_size=eval_batch_size)
    post_test_ppl = compute_perplexity(model, eval_test_dataset, batch_size=eval_batch_size)
    post_ppl = post_val_ppl if args.eval_split == "val" else post_test_ppl
    print(
        f"[RESULT] Post-training perplexity ({args.eval_split}): {post_ppl:.4f} "
        f"[val={post_val_ppl:.4f}, test={post_test_ppl:.4f}]"
    )

    ppl_reduction = baseline_ppl - post_ppl
    improvement_pct = (ppl_reduction / baseline_ppl) * 100 if baseline_ppl > 0 else 0.0

    metrics = {
        "track": "A - Extended Pre-Training",
        "repo_name": args.repo_name,
        "model": args.model_name,
        "corpus_dir": args.corpus_dir,
        "extensions": extensions,
        "split_mode": args.split_mode,
        "eval_split": args.eval_split,
        "eval_block_size": eval_block_size,
        "corpus_files": len(all_files),
        "train_files": len(train_files),
        "val_files": len(val_files),
        "test_files": len(test_files),
        "val_file_names": [name for name, _ in val_files],
        "test_file_names": [name for name, _ in test_files],
        "block_size": final_block_size,
        "sequence_curriculum": sequence_curriculum,
        "train_chunks": stage_summaries[-1]["train_chunks"] if stage_summaries else 0,
        "val_chunks": len(eval_val_dataset),
        "test_chunks": len(eval_test_dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": eval_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "baseline_validation_perplexity": round(baseline_val_ppl, 4),
        "baseline_test_perplexity": round(baseline_test_ppl, 4),
        "baseline_perplexity": round(baseline_ppl, 4),
        "post_training_validation_perplexity": round(post_val_ppl, 4),
        "post_training_test_perplexity": round(post_test_ppl, 4),
        "post_training_perplexity": round(post_ppl, 4),
        "perplexity_reduction": round(ppl_reduction, 4),
        "improvement_percent": round(improvement_pct, 4),
        "training_loss": round(mean_stage_loss, 4),
        "stage_summaries": stage_summaries,
    }

    with open(args.metrics_file, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    model_dir = os.path.join(args.output_dir, "model")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    print("\n" + "=" * 70)
    print("TRACK A - REPO-SPECIFIC RESULTS")
    print("=" * 70)
    print(f"  Repo:               {args.repo_name}")
    print(f"  Model:              {args.model_name}")
    print(f"  Corpus files:       {len(all_files)}")
    print(f"  Split mode:         {args.split_mode}")
    print(f"  Eval split:         {args.eval_split}")
    print(f"  Curriculum:         {sequence_curriculum}")
    print(f"  Train chunks:       {metrics['train_chunks']}")
    print(f"  Val/Test chunks:    {len(eval_val_dataset)}/{len(eval_test_dataset)}")
    print(f"  Learning rate:      {args.learning_rate}")
    print(f"  Baseline val PPL:   {baseline_val_ppl:.4f}")
    print(f"  Post val PPL:       {post_val_ppl:.4f}")
    print(f"  Baseline test PPL:  {baseline_test_ppl:.4f}")
    print(f"  Post test PPL:      {post_test_ppl:.4f}")
    print(f"  Primary PPL:        {baseline_ppl:.4f} -> {post_ppl:.4f}")
    print(f"  Reduction:          {ppl_reduction:.4f} ({improvement_pct:.2f}%)")
    print(f"  Metrics:            {args.metrics_file}")
    print(f"  Model dir:          {model_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
