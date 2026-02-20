import gc
import json
import datetime
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = Path("results/track_c")


def compute_metrics(query_embeddings, corpus_embeddings, relevant_indices, k=10):
    """Compute MRR@k, nDCG@k, Recall@k given query and corpus embeddings."""
    # Cosine similarity: normalize then dot product
    query_norm = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    corpus_norm = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    scores = query_norm @ corpus_norm.T  # (num_queries, num_corpus)

    mrr_scores = []
    ndcg_scores = []
    recall_scores = []

    for i, rel_idx in enumerate(relevant_indices):
        ranked = np.argsort(-scores[i])[:k]

        # MRR@k
        mrr = 0.0
        for rank, idx in enumerate(ranked):
            if idx == rel_idx:
                mrr = 1.0 / (rank + 1)
                break
        mrr_scores.append(mrr)

        # nDCG@k
        dcg = 0.0
        idcg = 1.0  # single relevant doc, ideal DCG = 1/log2(2) = 1
        for rank, idx in enumerate(ranked):
            if idx == rel_idx:
                dcg = 1.0 / np.log2(rank + 2)
                break
        ndcg_scores.append(dcg / idcg)

        # Recall@k
        recall_scores.append(1.0 if rel_idx in ranked else 0.0)

    return {
        f"MRR@{k}": np.mean(mrr_scores),
        f"nDCG@{k}": np.mean(ndcg_scores),
        f"Recall@{k}": np.mean(recall_scores),
    }


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_model(model_id, device):
    kwargs = {}
    if device == "cuda":
        kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
    return SentenceTransformer(model_id, device=device, trust_remote_code=True, **kwargs)


def eval_model(model, corpus_unique, all_queries, relevant_indices):
    with torch.no_grad():
        corpus_embeddings = model.encode(
            corpus_unique, batch_size=8, show_progress_bar=True, convert_to_numpy=True
        )
        query_embeddings = model.encode(
            all_queries, batch_size=8, show_progress_bar=True, convert_to_numpy=True
        )
    metrics = compute_metrics(query_embeddings, corpus_embeddings, relevant_indices, k=10)
    del corpus_embeddings, query_embeddings
    free_memory()
    return metrics


def main():
    base_model_id = "Qwen/Qwen3-Embedding-0.6B"
    finetuned_model_id = "archit11/assesment_qwen3_embedding_06b_e3"
    dataset_id = "archit11/assesment_embeddings_new"

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading dataset {dataset_id}...")
    dataset = load_dataset(dataset_id, split="test")
    print(f"Loaded {len(dataset)} examples | Columns: {dataset.column_names}")

    # corpus = anchors (code); queries come from the queries list field
    corpus_unique = list(dataset["anchor"])
    all_queries = []
    relevant_indices = []
    for i, row in enumerate(dataset):
        for q in row["queries"]:
            all_queries.append(q)
            relevant_indices.append(i)
    print(f"Corpus size: {len(corpus_unique)}, Total queries: {len(all_queries)}")

    # Baseline
    print(f"\nEvaluating base model: {base_model_id}...")
    base_model = load_model(base_model_id, device)
    baseline = eval_model(base_model, corpus_unique, all_queries, relevant_indices)
    del base_model
    free_memory()

    # Fine-tuned
    print(f"\nEvaluating fine-tuned model: {finetuned_model_id}...")
    ft_model = load_model(finetuned_model_id, device)
    finetuned = eval_model(ft_model, corpus_unique, all_queries, relevant_indices)
    del ft_model
    free_memory()

    print("\n--- Results ---")
    print(f"{'Metric':<15} {'Baseline':>10} {'Fine-Tuned':>12} {'Δ':>8}")
    print("-" * 47)
    for metric in baseline:
        b, f = baseline[metric], finetuned[metric]
        print(f"{metric:<15} {b:>10.4f} {f:>12.4f} {f-b:>+8.4f}")


if __name__ == "__main__":
    main()
