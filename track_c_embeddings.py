"""
Track C – Embedding Fine-Tuning
=================================
Fine-tune BAAI/bge-small-en-v1.5 on text→code retrieval pairs.
Evaluates with MRR@10, nDCG@10, Recall@10.

Designed for Google Colab T4 (16GB VRAM).
"""

import os
import json
import random
import math
import numpy as np
from typing import List, Dict, Tuple

import torch
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
)
from torch.utils.data import DataLoader

# ── Config ──────────────────────────────────────────────────────────
MODEL_NAME = "BAAI/bge-small-en-v1.5"
RETRIEVAL_TRAIN_FILE = "data/retrieval_train.json"
RETRIEVAL_TEST_FILE = "data/retrieval_test.json"
OUTPUT_DIR = "results/track_c"
METRICS_FILE = "results/track_c_metrics.json"

RANDOM_SEED = 42

# Training hyperparameters
NUM_EPOCHS = 4
BATCH_SIZE = 32
WARMUP_RATIO = 0.1
LEARNING_RATE = 2e-5


# ── Data Loading ────────────────────────────────────────────────────

def load_retrieval_data(filepath: str) -> List[Dict]:
    """Load retrieval data."""
    with open(filepath) as f:
        return json.load(f)


def create_training_examples(data: List[Dict]) -> List[InputExample]:
    """Convert data to SentenceTransformer InputExamples (query, positive_code)."""
    examples = []
    for item in data:
        query = item["query"]
        code = item["code"]
        # For BGE models, prepend instruction for queries
        query_with_instruction = f"Represent this sentence for searching relevant code: {query}"
        examples.append(InputExample(texts=[query_with_instruction, code]))
    return examples


# ── Retrieval Evaluation ───────────────────────────────────────────

def compute_retrieval_metrics(
    model: SentenceTransformer,
    test_data: List[Dict],
    k: int = 10,
) -> Dict:
    """
    Compute MRR@k, nDCG@k, Recall@k.
    For each query, the correct answer is its paired code snippet.
    We rank ALL code snippets in the test set.
    """
    queries = []
    codes = []
    query_to_code_idx = {}  # maps query index to its correct code index

    # Build unique code corpus from test data
    code_set = {}  # code -> index
    for item in test_data:
        code = item["code"]
        if code not in code_set:
            code_set[code] = len(codes)
            codes.append(code)

    # Map queries to their correct code
    for i, item in enumerate(test_data):
        query = f"Represent this sentence for searching relevant code: {item['query']}"
        queries.append(query)
        query_to_code_idx[i] = code_set[item["code"]]

    # Encode
    print(f"  Encoding {len(queries)} queries and {len(codes)} code snippets...")
    query_embeddings = model.encode(queries, show_progress_bar=True, convert_to_numpy=True)
    code_embeddings = model.encode(codes, show_progress_bar=True, convert_to_numpy=True)

    # Normalize for cosine similarity
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    code_embeddings = code_embeddings / np.linalg.norm(code_embeddings, axis=1, keepdims=True)

    # Compute similarity matrix
    similarity = np.dot(query_embeddings, code_embeddings.T)  # (num_queries, num_codes)

    # Compute metrics
    mrr_scores = []
    ndcg_scores = []
    recall_scores = []

    for i in range(len(queries)):
        correct_idx = query_to_code_idx[i]
        scores = similarity[i]

        # Get top-k indices
        top_k_indices = np.argsort(scores)[::-1][:k]

        # MRR@k
        rank = None
        for r, idx in enumerate(top_k_indices):
            if idx == correct_idx:
                rank = r + 1
                break
        mrr_scores.append(1.0 / rank if rank else 0.0)

        # Recall@k
        recall_scores.append(1.0 if correct_idx in top_k_indices else 0.0)

        # nDCG@k
        dcg = 0.0
        for r, idx in enumerate(top_k_indices):
            rel = 1.0 if idx == correct_idx else 0.0
            dcg += rel / math.log2(r + 2)  # r+2 because rank is 1-indexed
        idcg = 1.0 / math.log2(2)  # only 1 relevant doc
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        f"MRR@{k}": round(float(np.mean(mrr_scores)), 4),
        f"nDCG@{k}": round(float(np.mean(ndcg_scores)), 4),
        f"Recall@{k}": round(float(np.mean(recall_scores)), 4),
    }


def run_error_analysis(
    model: SentenceTransformer,
    test_data: List[Dict],
    k: int = 10,
    num_failures: int = 20,
) -> List[Dict]:
    """Analyze failure cases where the correct code is NOT in top-k."""
    queries = []
    codes = []
    code_set = {}

    for item in test_data:
        code = item["code"]
        if code not in code_set:
            code_set[code] = len(codes)
            codes.append(code)

    query_to_code_idx = {}
    for i, item in enumerate(test_data):
        query = f"Represent this sentence for searching relevant code: {item['query']}"
        queries.append(query)
        query_to_code_idx[i] = code_set[item["code"]]

    query_embeddings = model.encode(queries, convert_to_numpy=True)
    code_embeddings = model.encode(codes, convert_to_numpy=True)

    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    code_embeddings = code_embeddings / np.linalg.norm(code_embeddings, axis=1, keepdims=True)

    similarity = np.dot(query_embeddings, code_embeddings.T)

    failures = []
    for i in range(len(queries)):
        correct_idx = query_to_code_idx[i]
        top_k = np.argsort(similarity[i])[::-1][:k]

        if correct_idx not in top_k:
            # Find actual rank
            all_ranked = np.argsort(similarity[i])[::-1]
            actual_rank = int(np.where(all_ranked == correct_idx)[0][0]) + 1

            failures.append({
                "query": test_data[i]["query"],
                "function_name": test_data[i].get("function_name", "unknown"),
                "correct_rank": actual_rank,
                "correct_similarity": float(similarity[i][correct_idx]),
                "top_1_similarity": float(similarity[i][top_k[0]]),
                "code_preview": test_data[i]["code"][:150],
            })

        if len(failures) >= num_failures:
            break

    return failures


# ── Main ────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # 1. Load data
    print("[INFO] Loading retrieval data...")
    train_data = load_retrieval_data(RETRIEVAL_TRAIN_FILE)
    test_data = load_retrieval_data(RETRIEVAL_TEST_FILE)
    print(f"[INFO] Train: {len(train_data)} pairs, Test: {len(test_data)} pairs")

    # 2. Load baseline model
    print("[INFO] Loading baseline model...")
    baseline_model = SentenceTransformer(MODEL_NAME, device=device)

    # 3. Baseline evaluation
    print("\n[INFO] Baseline evaluation...")
    baseline_metrics = compute_retrieval_metrics(baseline_model, test_data)
    print(f"[RESULT] Baseline: {baseline_metrics}")

    # 4. Prepare training data
    print("\n[INFO] Preparing training data...")
    train_examples = create_training_examples(train_data)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

    # 5. Fine-tune
    print("[INFO] Starting fine-tuning...")
    ft_model = SentenceTransformer(MODEL_NAME, device=device)

    # Use MultipleNegativesRankingLoss (in-batch negatives)
    train_loss = losses.MultipleNegativesRankingLoss(ft_model)

    warmup_steps = int(len(train_dataloader) * NUM_EPOCHS * WARMUP_RATIO)

    ft_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=NUM_EPOCHS,
        warmup_steps=warmup_steps,
        output_path=os.path.join(OUTPUT_DIR, "model"),
        show_progress_bar=True,
        optimizer_params={"lr": LEARNING_RATE},
    )

    # 6. Post-training evaluation
    print("\n[INFO] Post-training evaluation...")
    ft_model = SentenceTransformer(os.path.join(OUTPUT_DIR, "model"), device=device)
    post_metrics = compute_retrieval_metrics(ft_model, test_data)
    print(f"[RESULT] Fine-tuned: {post_metrics}")

    # 7. Error analysis
    print("\n[INFO] Running error analysis...")
    failures = run_error_analysis(ft_model, test_data)
    print(f"[INFO] Found {len(failures)} failure cases.")

    # 8. Save metrics
    metrics = {
        "track": "C – Embedding Fine-Tuning",
        "model": MODEL_NAME,
        "train_pairs": len(train_data),
        "test_pairs": len(test_data),
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "baseline": baseline_metrics,
        "fine_tuned": post_metrics,
        "improvement": {
            k: round(post_metrics[k] - baseline_metrics[k], 4)
            for k in baseline_metrics
        },
        "error_analysis": failures,
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    # 9. Print summary
    print("\n" + "=" * 60)
    print("TRACK C – EMBEDDING FINE-TUNING RESULTS")
    print("=" * 60)
    print(f"  Model:          {MODEL_NAME}")
    print(f"  Train pairs:    {len(train_data)}")
    print(f"  Epochs:         {NUM_EPOCHS}")
    print()
    print(f"  {'Metric':<15} {'Baseline':>10} {'Fine-tuned':>12} {'Delta':>10}")
    print(f"  {'-'*47}")
    for key in baseline_metrics:
        b = baseline_metrics[key]
        p = post_metrics[key]
        d = p - b
        print(f"  {key:<15} {b:>10.4f} {p:>12.4f} {d:>+10.4f}")
    print("=" * 60)
    print(f"  Error cases: {len(failures)}")
    print(f"  Metrics saved to {METRICS_FILE}")


if __name__ == "__main__":
    main()
