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
import argparse
import numpy as np
from typing import Any, List, Dict, Tuple
from pathlib import Path

import torch
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


# ── Dependency Loading ───────────────────────────────────────────────

def import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: sentence-transformers. Install with: "
            "pip install sentence-transformers"
        ) from exc
    return SentenceTransformer, InputExample, losses


def load_sentence_transformer(
    SentenceTransformer,
    model_name_or_path: str,
    device: str,
    trust_remote_code: bool = False,
):
    if trust_remote_code:
        # Compatibility shim for some remote model files expecting older transformers layout.
        try:
            import transformers.modeling_utils as modeling_utils
            if not hasattr(modeling_utils, "Conv1D"):
                from transformers.pytorch_utils import Conv1D
                modeling_utils.Conv1D = Conv1D
        except Exception:
            pass

    kwargs = {"device": device}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    try:
        return SentenceTransformer(model_name_or_path, **kwargs)
    except TypeError:
        # Some sentence-transformers versions only pass this via model_kwargs.
        if trust_remote_code:
            kwargs.pop("trust_remote_code", None)
            kwargs["model_kwargs"] = {"trust_remote_code": True}
        return SentenceTransformer(model_name_or_path, **kwargs)


# ── Data Loading ────────────────────────────────────────────────────

def format_query_for_model(query: str, model_name: str) -> str:
    m = model_name.lower()
    q = query.strip()
    if "e5" in m:
        return f"query: {q}"
    if "bge" in m:
        return f"Represent this sentence for searching relevant code: {q}"
    return q


def format_code_for_model(code: str, model_name: str) -> str:
    m = model_name.lower()
    c = code.strip()
    if "e5" in m:
        return f"passage: {c}"
    return c

def load_retrieval_data(filepath: str) -> List[Dict]:
    """Load retrieval data from JSON or JSONL and normalize to {query, code, ...} rows."""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Retrieval file not found: {filepath}")

    rows: List[Dict] = []

    if p.suffix == ".jsonl":
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "query" in item and "code" in item:
                rows.append(item)
                continue
            if "queries" in item and "anchor" in item:
                for q in item.get("queries", []):
                    if isinstance(q, str) and q.strip():
                        rows.append(
                            {
                                "query": q.strip(),
                                "code": item.get("anchor", ""),
                                "function_name": item.get("symbol") or item.get("filename"),
                                "filename": item.get("filename"),
                                "path": item.get("path"),
                                "label": item.get("label"),
                                "language": item.get("language"),
                                "repo": item.get("repo"),
                            }
                        )
        return rows

    data = json.loads(p.read_text())
    if not isinstance(data, list):
        return rows
    for item in data:
        if not isinstance(item, dict):
            continue
        if "query" in item and "code" in item:
            rows.append(item)
            continue
        if "queries" in item and "anchor" in item:
            for q in item.get("queries", []):
                if isinstance(q, str) and q.strip():
                    rows.append(
                        {
                            "query": q.strip(),
                            "code": item.get("anchor", ""),
                            "function_name": item.get("symbol") or item.get("filename"),
                            "filename": item.get("filename"),
                            "path": item.get("path"),
                            "label": item.get("label"),
                            "language": item.get("language"),
                            "repo": item.get("repo"),
                        }
                    )
    return rows


def create_training_examples(data: List[Dict], model_name: str) -> List[Any]:
    """Convert data to SentenceTransformer InputExamples (query, positive_code)."""
    _, InputExample, _ = import_sentence_transformers()
    examples = []
    for item in data:
        query = item["query"]
        code = item["code"]
        query_text = format_query_for_model(query, model_name)
        code_text = format_code_for_model(code, model_name)
        examples.append(InputExample(texts=[query_text, code_text]))
    return examples


# ── Retrieval Evaluation ───────────────────────────────────────────

def compute_retrieval_metrics(
    model: Any,
    test_data: List[Dict],
    model_name: str,
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
            codes.append(format_code_for_model(code, model_name))

    # Map queries to their correct code
    for i, item in enumerate(test_data):
        query = format_query_for_model(item["query"], model_name)
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
    acc1_scores = []

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
        acc1_scores.append(1.0 if top_k_indices[0] == correct_idx else 0.0)

        # nDCG@k
        dcg = 0.0
        for r, idx in enumerate(top_k_indices):
            rel = 1.0 if idx == correct_idx else 0.0
            dcg += rel / math.log2(r + 2)  # r+2 because rank is 1-indexed
        idcg = 1.0 / math.log2(2)  # only 1 relevant doc
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        "Accuracy@1": round(float(np.mean(acc1_scores)), 4),
        f"MRR@{k}": round(float(np.mean(mrr_scores)), 4),
        f"nDCG@{k}": round(float(np.mean(ndcg_scores)), 4),
        f"Recall@{k}": round(float(np.mean(recall_scores)), 4),
    }


def run_error_analysis(
    model: Any,
    test_data: List[Dict],
    model_name: str,
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
            codes.append(format_code_for_model(code, model_name))

    query_to_code_idx = {}
    for i, item in enumerate(test_data):
        query = format_query_for_model(item["query"], model_name)
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


# ── CLI ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track C embedding fine-tuning")
    p.add_argument("--model-name", default=MODEL_NAME)
    p.add_argument("--train-file", default=RETRIEVAL_TRAIN_FILE)
    p.add_argument("--test-file", default=RETRIEVAL_TEST_FILE)
    p.add_argument("--val-file", default=None, help="Optional validation retrieval JSON")
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--metrics-file", default=METRICS_FILE)
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO)
    p.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--max-seq-length", type=int, default=0, help="0 = model default")
    return p.parse_args()


# ── Main ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    SentenceTransformer, _, losses = import_sentence_transformers()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_parent = Path(args.metrics_file).parent
    if str(metrics_parent) not in ("", "."):
        os.makedirs(metrics_parent, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # 1. Load data
    print("[INFO] Loading retrieval data...")
    train_data = load_retrieval_data(args.train_file)
    test_data = load_retrieval_data(args.test_file)
    val_data = load_retrieval_data(args.val_file) if args.val_file else None
    if val_data is not None:
        print(
            f"[INFO] Train: {len(train_data)} pairs, "
            f"Val: {len(val_data)} pairs, Test: {len(test_data)} pairs"
        )
    else:
        print(f"[INFO] Train: {len(train_data)} pairs, Test: {len(test_data)} pairs")

    # 2. Load baseline model
    print("[INFO] Loading baseline model...")
    baseline_model = load_sentence_transformer(
        SentenceTransformer,
        args.model_name,
        device=device,
        trust_remote_code=args.trust_remote_code,
    )
    if args.max_seq_length and args.max_seq_length > 0:
        baseline_model.max_seq_length = args.max_seq_length

    # 3. Baseline evaluation
    print("\n[INFO] Baseline evaluation...")
    baseline_metrics_test = compute_retrieval_metrics(baseline_model, test_data, args.model_name)
    print(f"[RESULT] Baseline (test): {baseline_metrics_test}")
    baseline_metrics_val = None
    if val_data is not None:
        baseline_metrics_val = compute_retrieval_metrics(baseline_model, val_data, args.model_name)
        print(f"[RESULT] Baseline (val):  {baseline_metrics_val}")

    # 4. Prepare training data
    print("\n[INFO] Preparing training data...")
    train_examples = create_training_examples(train_data, args.model_name)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)

    # 5. Fine-tune
    print("[INFO] Starting fine-tuning...")
    ft_model = load_sentence_transformer(
        SentenceTransformer,
        args.model_name,
        device=device,
        trust_remote_code=args.trust_remote_code,
    )
    if args.max_seq_length and args.max_seq_length > 0:
        ft_model.max_seq_length = args.max_seq_length

    # Use MultipleNegativesRankingLoss (in-batch negatives)
    train_loss = losses.MultipleNegativesRankingLoss(ft_model)

    warmup_steps = int(len(train_dataloader) * args.epochs * args.warmup_ratio)

    ft_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=os.path.join(args.output_dir, "model"),
        show_progress_bar=True,
        optimizer_params={"lr": args.learning_rate},
    )

    # 6. Post-training evaluation
    print("\n[INFO] Post-training evaluation...")
    ft_model = load_sentence_transformer(
        SentenceTransformer,
        os.path.join(args.output_dir, "model"),
        device=device,
        trust_remote_code=args.trust_remote_code,
    )
    if args.max_seq_length and args.max_seq_length > 0:
        ft_model.max_seq_length = args.max_seq_length
    post_metrics_test = compute_retrieval_metrics(ft_model, test_data, args.model_name)
    print(f"[RESULT] Fine-tuned (test): {post_metrics_test}")
    post_metrics_val = None
    if val_data is not None:
        post_metrics_val = compute_retrieval_metrics(ft_model, val_data, args.model_name)
        print(f"[RESULT] Fine-tuned (val):  {post_metrics_val}")

    # 7. Error analysis
    print("\n[INFO] Running error analysis...")
    failures = run_error_analysis(ft_model, test_data, args.model_name)
    print(f"[INFO] Found {len(failures)} failure cases.")

    # 8. Save metrics
    metrics = {
        "track": "C – Embedding Fine-Tuning",
        "model": args.model_name,
        "train_file": args.train_file,
        "val_file": args.val_file,
        "test_file": args.test_file,
        "train_pairs": len(train_data),
        "val_pairs": len(val_data) if val_data is not None else 0,
        "test_pairs": len(test_data),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "baseline_test": baseline_metrics_test,
        "fine_tuned_test": post_metrics_test,
        "improvement_test": {
            k: round(post_metrics_test[k] - baseline_metrics_test[k], 4)
            for k in baseline_metrics_test
        },
        "baseline_val": baseline_metrics_val,
        "fine_tuned_val": post_metrics_val,
        "improvement_val": (
            {
                k: round(post_metrics_val[k] - baseline_metrics_val[k], 4)
                for k in baseline_metrics_val
            }
            if baseline_metrics_val is not None and post_metrics_val is not None
            else None
        ),
        # Kept for backward compatibility with older consumers.
        "baseline": baseline_metrics_test,
        "fine_tuned": post_metrics_test,
        "improvement": {
            k: round(post_metrics_test[k] - baseline_metrics_test[k], 4)
            for k in baseline_metrics_test
        },
        "error_analysis": failures,
    }

    with open(args.metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    # 9. Print summary
    print("\n" + "=" * 60)
    print("TRACK C – EMBEDDING FINE-TUNING RESULTS")
    print("=" * 60)
    print(f"  Model:          {args.model_name}")
    print(f"  Train pairs:    {len(train_data)}")
    if val_data is not None:
        print(f"  Val pairs:      {len(val_data)}")
    print(f"  Test pairs:     {len(test_data)}")
    print(f"  Epochs:         {args.epochs}")
    print()
    print("  Test Metrics")
    print(f"  {'Metric':<15} {'Baseline':>10} {'Fine-tuned':>12} {'Delta':>10}")
    print(f"  {'-'*47}")
    for key in baseline_metrics_test:
        b = baseline_metrics_test[key]
        p = post_metrics_test[key]
        d = p - b
        print(f"  {key:<15} {b:>10.4f} {p:>12.4f} {d:>+10.4f}")
    if baseline_metrics_val is not None and post_metrics_val is not None:
        print()
        print("  Val Metrics")
        print(f"  {'Metric':<15} {'Baseline':>10} {'Fine-tuned':>12} {'Delta':>10}")
        print(f"  {'-'*47}")
        for key in baseline_metrics_val:
            b = baseline_metrics_val[key]
            p = post_metrics_val[key]
            d = p - b
            print(f"  {key:<15} {b:>10.4f} {p:>12.4f} {d:>+10.4f}")
    print("=" * 60)
    print(f"  Error cases: {len(failures)}")
    print(f"  Metrics saved to {args.metrics_file}")


if __name__ == "__main__":
    main()
