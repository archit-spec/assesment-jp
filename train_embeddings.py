"""
Fine-tune Qwen3-Embedding-0.6B on archit11/assesment_embeddings
using SentenceTransformerTrainer + MultipleNegativesRankingLoss.
"""

import gc
import math
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.training_args import BatchSamplers

BASE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DATASET_ID = "archit11/assesment_embeddings"
OUTPUT_DIR = "output_track_c"
HF_REPO = "archit11/assesment_qwen3_embedding_06b_e3"  # set to None to skip push

EPOCHS = 3
BATCH_SIZE = 2           # T4-safe; effective batch = 4 with grad accum
GRAD_ACCUM = 2
LR = 2e-5
WARMUP_RATIO = 0.03
MAX_SEQ_LENGTH = 512     # Qwen3-Embedding works well at 512 for code retrieval


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def compute_retrieval_metrics(model, test_ds, k=10):
    anchors = list(test_ds["anchor"])
    queries_flat = []
    query_to_anchor = []
    for i, row in enumerate(test_ds):
        qs = row.get("queries", [])
        if not qs:
            continue
        for q in qs:
            queries_flat.append(q)
            query_to_anchor.append(i)

    print(f"  Encoding {len(anchors)} anchors and {len(queries_flat)} queries...")
    # Use small batch size and keep on CPU to avoid OOM during eval
    anchor_embs = model.encode(anchors, batch_size=8, show_progress_bar=True, convert_to_numpy=True)
    query_embs = model.encode(queries_flat, batch_size=8, show_progress_bar=True, convert_to_numpy=True)

    anchor_embs = anchor_embs / np.linalg.norm(anchor_embs, axis=1, keepdims=True)
    query_embs = query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True)

    sim = query_embs @ anchor_embs.T  # (Q, N)

    mrr, ndcg, recall = [], [], []
    for i, correct in enumerate(query_to_anchor):
        ranked = np.argsort(-sim[i])[:k]
        rank = None
        for r, idx in enumerate(ranked):
            if idx == correct:
                rank = r + 1
                break
        mrr.append(1.0 / rank if rank else 0.0)
        recall.append(1.0 if correct in ranked else 0.0)
        dcg = sum(1.0 / math.log2(r + 2) for r, idx in enumerate(ranked) if idx == correct)
        ndcg.append(dcg / 1.0)  # iDCG = 1 (single relevant doc)

    # Free eval tensors before returning
    del anchor_embs, query_embs, sim
    free_memory()

    return {
        f"MRR@{k}": round(float(np.mean(mrr)), 4),
        f"nDCG@{k}": round(float(np.mean(ndcg)), 4),
        f"Recall@{k}": round(float(np.mean(recall)), 4),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    is_cuda = device == "cuda"
    print(f"Device: {device}")

    # Load dataset
    print(f"Loading dataset {DATASET_ID}...")
    ds = load_dataset(DATASET_ID)
    train_ds = ds["train"].select_columns(["anchor", "positive"])
    test_ds = ds["test"] if "test" in ds else ds["validation"]
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    # Load model
    print(f"Loading {BASE_MODEL}...")
    model = SentenceTransformer(
        BASE_MODEL,
        device=device,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.float32},
    )
    model.max_seq_length = MAX_SEQ_LENGTH

    # Baseline eval
    print("\nBaseline evaluation...")
    baseline = compute_retrieval_metrics(model, test_ds)
    print(f"Baseline: {baseline}")

    # Free memory before training starts
    free_memory()

    # Train
    loss_fn = losses.MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        fp16=is_cuda,           # fp16 on GPU, off on MPS/CPU
        bf16=False,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        lr_scheduler_type="cosine",
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        label_names=[],
        dataloader_pin_memory=False,  # saves GPU memory on Colab
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_ds,
        loss=loss_fn,
        args=args,
    )

    print("\nStarting training...")
    trainer.train()

    model.save_pretrained(f"{OUTPUT_DIR}/final")
    print(f"Model saved to {OUTPUT_DIR}/final")

    # Post-training eval
    print("\nPost-training evaluation...")
    finetuned = compute_retrieval_metrics(model, test_ds)
    print(f"Fine-tuned: {finetuned}")

    # Summary
    print("\n" + "=" * 55)
    print(f"  {'Metric':<15} {'Baseline':>10} {'Fine-Tuned':>12} {'Δ':>8}")
    print("  " + "-" * 47)
    for k in baseline:
        b, f = baseline[k], finetuned[k]
        print(f"  {k:<15} {b:>10.4f} {f:>12.4f} {f-b:>+8.4f}")
    print("=" * 55)

    # Push to Hub
    if HF_REPO:
        print(f"\nPushing to {HF_REPO}...")
        model.push_to_hub(HF_REPO)
        print("Done.")


if __name__ == "__main__":
    main()
