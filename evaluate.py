"""
Consolidated Evaluation Script
================================
Loads metrics from all three tracks, generates a consolidated summary,
and exports results to JSON + prints a formatted report.
"""

import os
import json
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────
METRICS_FILES = {
    "Track A": "results/track_a_metrics.json",
    "Track B": "results/track_b_metrics.json",
    "Track C": "results/track_c_metrics.json",
}
OUTPUT_FILE = "results/all_metrics.json"
DATA_CARDS = {
    "SFT": "data/sft_data_card.json",
    "Retrieval": "data/retrieval_data_card.json",
}


def load_json(path: str) -> dict:
    """Load JSON file, return empty dict if not found."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"  [WARN] {path} not found, skipping.")
        return {}


def main():
    print("=" * 70)
    print("CONSOLIDATED EVALUATION REPORT")
    print("=" * 70)

    all_metrics = {}
    tracks_passed = 0

    # ── Track A ─────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("TRACK A – Extended Pre-Training")
    print("─" * 70)
    a = load_json(METRICS_FILES["Track A"])
    if a:
        all_metrics["track_a"] = a
        baseline = a.get("baseline_perplexity", 0)
        post = a.get("post_training_perplexity", 0)
        delta = a.get("perplexity_reduction", 0)
        pct = a.get("improvement_percent", 0)
        passed = post < baseline

        print(f"  Model:              {a.get('model', 'N/A')}")
        print(f"  Corpus files:       {a.get('corpus_files', 'N/A')}")
        print(f"  Train chunks:       {a.get('train_chunks', 'N/A')}")
        print(f"  Epochs:             {a.get('epochs', 'N/A')}")
        print(f"  Baseline PPL:       {baseline:.2f}")
        print(f"  Post-training PPL:  {post:.2f}")
        print(f"  Reduction:          {delta:.2f} ({pct:.1f}%)")
        print(f"  Status:             {'✓ PASSED' if passed else '✗ FAILED'}")
        if passed:
            tracks_passed += 1
    else:
        print("  [SKIP] No metrics available.")

    # ── Track B ─────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("TRACK B – Instruction Tuning (SFT)")
    print("─" * 70)
    b = load_json(METRICS_FILES["Track B"])
    if b:
        all_metrics["track_b"] = b
        bl = b.get("baseline", {})
        pt = b.get("post_training", {})
        imp = b.get("improvement", {})

        pass1_delta = imp.get("pass_at_1_delta", 0)
        style_delta = imp.get("style_delta", 0)
        passed = pass1_delta > 0 or style_delta > 0

        print(f"  Model:              {b.get('model', 'N/A')}")
        print(f"  Train samples:      {b.get('train_samples', 'N/A')}")
        print(f"  Epochs:             {b.get('epochs', 'N/A')}")
        print(f"  Training loss:      {b.get('training_loss', 'N/A')}")
        print()
        print(f"  {'Metric':<25} {'Baseline':>10} {'Post-train':>12} {'Delta':>10}")
        print(f"  {'-'*57}")
        print(f"  {'pass@1':<25} {bl.get('pass_at_1',0):>10.4f} {pt.get('pass_at_1',0):>12.4f} {pass1_delta:>+10.4f}")
        print(f"  {'Style Score':<25} {bl.get('avg_style_score',0):>10.4f} {pt.get('avg_style_score',0):>12.4f} {style_delta:>+10.4f}")
        print(f"  {'Hallucination Rate':<25} {bl.get('hallucination_rate',0):>10.4f} {pt.get('hallucination_rate',0):>12.4f}")
        print()

        # Per-category breakdown
        if pt.get("category_scores"):
            print(f"  {'Category':<15} {'Syntax Rate':>12} {'Avg Style':>12}")
            print(f"  {'-'*39}")
            for cat, vals in pt["category_scores"].items():
                print(f"  {cat:<15} {vals.get('syntax_rate',0):>12.3f} {vals.get('avg_style',0):>12.3f}")

        print(f"\n  Status:             {'✓ PASSED' if passed else '✗ FAILED'}")

        # Error analysis
        errors = b.get("error_analysis", [])
        if errors:
            print(f"\n  Error Analysis ({len(errors)} cases):")
            for i, err in enumerate(errors[:5]):
                print(f"    [{i+1}] Category: {err.get('category', '?')}, "
                      f"Syntax: {err.get('syntax_ok', '?')}, "
                      f"Style: {err.get('style_score', '?'):.2f}")
                print(f"         Instruction: {err.get('instruction_preview', '')[:80]}...")

        if passed:
            tracks_passed += 1
    else:
        print("  [SKIP] No metrics available.")

    # ── Track C ─────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("TRACK C – Embedding Fine-Tuning")
    print("─" * 70)
    c = load_json(METRICS_FILES["Track C"])
    if c:
        all_metrics["track_c"] = c
        bl = c.get("baseline", {})
        ft = c.get("fine_tuned", {})
        imp = c.get("improvement", {})

        any_improved = any(v > 0 for v in imp.values())

        print(f"  Model:          {c.get('model', 'N/A')}")
        print(f"  Train pairs:    {c.get('train_pairs', 'N/A')}")
        print(f"  Epochs:         {c.get('epochs', 'N/A')}")
        print()
        print(f"  {'Metric':<15} {'Baseline':>10} {'Fine-tuned':>12} {'Delta':>10}")
        print(f"  {'-'*47}")
        for key in bl:
            b_val = bl[key]
            f_val = ft.get(key, 0)
            d_val = imp.get(key, 0)
            print(f"  {key:<15} {b_val:>10.4f} {f_val:>12.4f} {d_val:>+10.4f}")

        print(f"\n  Status:         {'✓ PASSED' if any_improved else '✗ FAILED'}")

        # Error analysis
        errors = c.get("error_analysis", [])
        if errors:
            print(f"\n  Error Analysis ({len(errors)} cases):")
            for i, err in enumerate(errors[:5]):
                print(f"    [{i+1}] Query: \"{err.get('query', '?')[:60]}\"")
                print(f"         Function: {err.get('function_name', '?')}, "
                      f"Rank: {err.get('correct_rank', '?')}, "
                      f"Sim: {err.get('correct_similarity', 0):.4f}")

        if any_improved:
            tracks_passed += 1
    else:
        print("  [SKIP] No metrics available.")

    # ── Data Cards ──────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("DATA CARDS")
    print("─" * 70)
    for name, path in DATA_CARDS.items():
        card = load_json(path)
        if card:
            all_metrics[f"data_card_{name.lower()}"] = card
            print(f"  {name}: {card.get('total_pairs', 'N/A')} pairs "
                  f"(train: {card.get('train_pairs', 'N/A')}, test: {card.get('test_pairs', 'N/A')})")

    # ── Final Summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"FINAL RESULT: {tracks_passed}/3 tracks passed")
    print(f"{'✓ ASSIGNMENT PASSED' if tracks_passed >= 2 else '✗ NEEDS MORE WORK'}")
    print("=" * 70)

    # Save consolidated metrics
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll metrics saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
