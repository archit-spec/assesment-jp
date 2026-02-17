"""
Track A – Extended Pre-Training
================================
Continue training Qwen/Qwen2.5-Coder-3B on the verl Python code corpus.
Measures baseline vs post-training perplexity.

Designed for Google Colab T4 (16GB VRAM).
"""

import os
import json
import math
import random
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# ── Config ──────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-Coder-3B"
CORPUS_DIR = "data/code_corpus"
OUTPUT_DIR = "results/track_a"
METRICS_FILE = "results/track_a_metrics.json"

BLOCK_SIZE = 512
VAL_RATIO = 0.10
RANDOM_SEED = 42

# Training hyperparameters (Colab T4 friendly)
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8        # effective batch size = 16
LEARNING_RATE = 2e-4         # higher LR for LoRA
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
FP16 = True

# LoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ── Data Loading ────────────────────────────────────────────────────

def load_code_files(corpus_dir: str) -> list:
    """Load all code files as a list of (filename, content) tuples. Supports local dir or HF dataset."""
    files = []
    
    # Check if it's a local directory
    if os.path.exists(corpus_dir):
        corpus_path = Path(corpus_dir)
        print(f"[INFO] Loading from local directory: {corpus_path}")
        for py_file in sorted(corpus_path.glob("*.py")):
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            files.append((py_file.name, f"# FILE: {py_file.name}\n{content}"))
    else:
        # Assume it's a Hugging Face dataset
        print(f"[INFO] Loading from Hugging Face: {corpus_dir}")
        try:
            from datasets import load_dataset
            ds = load_dataset(corpus_dir, split="train")
            for item in ds:
                content = item["text"]
                filename = item.get("file_name", "unknown.py")
                files.append((filename, f"# FILE: {filename}\n{content}"))
        except Exception as e:
            print(f"[ERROR] Failed to load dataset {corpus_dir}: {e}")
            return []

    print(f"[INFO] Loaded {len(files)} files.")
    return files


def tokenize_and_chunk(text: str, tokenizer, block_size: int) -> Dataset:
    """Tokenize text and create chunks of block_size for causal LM training."""
    # Tokenize the entire corpus
    tokenized = tokenizer(text, return_attention_mask=False)
    input_ids = tokenized["input_ids"]
    print(f"[INFO] Total tokens: {len(input_ids):,}")

    # Create non-overlapping chunks
    chunks = []
    for i in range(0, len(input_ids) - block_size, block_size):
        chunk = input_ids[i : i + block_size]
        chunks.append({"input_ids": chunk})

    print(f"[INFO] Created {len(chunks)} chunks of {block_size} tokens each.")
    return Dataset.from_list(chunks)


# ── Perplexity Evaluation ──────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(model, dataset, tokenizer, batch_size: int = 8) -> float:
    """Compute perplexity on a dataset."""
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(dataset), batch_size):
        batch_items = dataset[i : i + batch_size]
        input_ids = torch.tensor(batch_items["input_ids"]).to(device)
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        # loss is averaged over tokens in batch, multiply back
        num_tokens = input_ids.numel()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity


# ── Main ────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # 1. Load tokenizer and model
    print("[INFO] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        trust_remote_code=True,
    )

    # 2. Apply LoRA
    print("[INFO] Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.to(device)

    # 3. Load corpus and split at FILE level (not chunk level)
    print("[INFO] Loading code corpus...")
    all_files = load_code_files(CORPUS_DIR)
    random.shuffle(all_files)

    val_count = max(1, int(len(all_files) * VAL_RATIO))
    val_files = all_files[:val_count]
    train_files = all_files[val_count:]
    print(f"[INFO] File-level split: {len(train_files)} train files, {len(val_files)} val files")

    # Tokenize and chunk each split separately
    train_text = "\n\n".join(content for _, content in train_files)
    val_text = "\n\n".join(content for _, content in val_files)
    print(f"[INFO] Train text: {len(train_text):,} chars, Val text: {len(val_text):,} chars")

    train_dataset = tokenize_and_chunk(train_text, tokenizer, BLOCK_SIZE)
    val_dataset = tokenize_and_chunk(val_text, tokenizer, BLOCK_SIZE)
    print(f"[INFO] Train: {len(train_dataset)} chunks, Val: {len(val_dataset)} chunks")
    print(f"[INFO] Val files: {[name for name, _ in val_files]}")

    # 4. Compute baseline perplexity on held-out files
    print("[INFO] Computing baseline perplexity on held-out files...")
    baseline_ppl = compute_perplexity(model, val_dataset, tokenizer)
    print(f"[RESULT] Baseline perplexity: {baseline_ppl:.2f}")

    # 5. Training
    print("[INFO] Starting training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        fp16=FP16 and device == "cuda",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
        seed=RANDOM_SEED,
        dataloader_pin_memory=True,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal LM — no masked language modeling
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    print(f"[INFO] Training complete. Loss: {train_result.training_loss:.4f}")

    # 6. Compute post-training perplexity on same held-out files
    print("[INFO] Computing post-training perplexity on held-out files...")
    post_ppl = compute_perplexity(model, val_dataset, tokenizer)
    print(f"[RESULT] Post-training perplexity: {post_ppl:.2f}")

    # 7. Compute improvement
    ppl_reduction = baseline_ppl - post_ppl
    ppl_improvement_pct = (ppl_reduction / baseline_ppl) * 100

    # 8. Save results
    metrics = {
        "track": "A – Extended Pre-Training",
        "model": MODEL_NAME,
        "corpus_files": len(all_files),
        "train_files": len(train_files),
        "val_files": len(val_files),
        "val_file_names": [name for name, _ in val_files],
        "block_size": BLOCK_SIZE,
        "train_chunks": len(train_dataset),
        "val_chunks": len(val_dataset),
        "epochs": NUM_EPOCHS,
        "lora_r": LORA_R,
        "max_grad_norm": MAX_GRAD_NORM,
        "baseline_perplexity": round(baseline_ppl, 2),
        "post_training_perplexity": round(post_ppl, 2),
        "perplexity_reduction": round(ppl_reduction, 2),
        "improvement_percent": round(ppl_improvement_pct, 2),
        "training_loss": round(train_result.training_loss, 4),
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    # 10. Print summary
    print("\n" + "=" * 60)
    print("TRACK A – EXTENDED PRE-TRAINING RESULTS")
    print("=" * 60)
    print(f"  Model:              {MODEL_NAME}")
    print(f"  Corpus files:       {metrics['corpus_files']}")
    print(f"  Training chunks:    {metrics['train_chunks']}")
    print(f"  Epochs:             {NUM_EPOCHS}")
    print(f"  LoRA rank:          {LORA_R}")
    print(f"  Baseline PPL:       {baseline_ppl:.2f}")
    print(f"  Post-training PPL:  {post_ppl:.2f}")
    print(f"  Reduction:          {ppl_reduction:.2f} ({ppl_improvement_pct:.1f}%)")
    print("=" * 60)
    print(f"  Metrics saved to {METRICS_FILE}")

    # Save the model
    model.save_pretrained(os.path.join(OUTPUT_DIR, "model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "model"))
    print(f"  Model saved to {OUTPUT_DIR}/model")


if __name__ == "__main__":
    main()
