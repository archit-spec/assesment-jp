"""
Build Rust-focused SFT data from embedding pairs.

Input rows (jsonl) are expected to have:
- queries: list[str]
- anchor: Rust snippet
- positive: gold explanation
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


def clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n..."


def load_jsonl(path: Path) -> Iterable[Dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_examples(rows: Iterable[Dict], max_queries_per_row: int, max_code_chars: int, max_resp_chars: int) -> List[Dict]:
    out: List[Dict] = []
    for row in rows:
        queries = row.get("queries") or []
        if not queries:
            continue
        code = clip(row.get("anchor", ""), max_code_chars)
        gold = clip(row.get("positive", ""), max_resp_chars)
        path = row.get("path") or row.get("filename") or "unknown"
        symbol = row.get("symbol") or "unknown_symbol"
        if not code or not gold:
            continue

        for q in queries[:max_queries_per_row]:
            q = (q or "").strip()
            if not q:
                continue
            instruction = (
                "You are a Rust maintainer for the Hyperswitch codebase.\n"
                f"User query: {q}\n\n"
                "Explain the following Rust snippet in the context of this query.\n"
                "Your answer must include:\n"
                "1) What the function/module does\n"
                "2) Important inputs/outputs/errors\n"
                "3) Why this snippet matches the query\n\n"
                f"Path: {path}\n"
                f"Symbol: {symbol}\n\n"
                "```rust\n"
                f"{code}\n"
                "```"
            )
            out.append(
                {
                    "instruction": instruction,
                    "response": gold,
                    "category": "explain",
                    "source": "embedding_rust",
                }
            )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare Rust SFT data from embedding jsonl files")
    p.add_argument("--train-jsonl", default="data/hf_download/assesment_embeddings_new_clean/train.jsonl")
    p.add_argument("--test-jsonl", default="data/hf_download/assesment_embeddings_new_clean/test.jsonl")
    p.add_argument("--out-train", default="data/sft_rust_embed_train.json")
    p.add_argument("--out-test", default="data/sft_rust_embed_test.json")
    p.add_argument("--max-queries-per-row", type=int, default=2)
    p.add_argument("--max-code-chars", type=int, default=1400)
    p.add_argument("--max-response-chars", type=int, default=900)
    args = p.parse_args()

    train_rows = list(load_jsonl(Path(args.train_jsonl)))
    test_rows = list(load_jsonl(Path(args.test_jsonl)))

    train_examples = build_examples(
        train_rows,
        max_queries_per_row=args.max_queries_per_row,
        max_code_chars=args.max_code_chars,
        max_resp_chars=args.max_response_chars,
    )
    test_examples = build_examples(
        test_rows,
        max_queries_per_row=args.max_queries_per_row,
        max_code_chars=args.max_code_chars,
        max_resp_chars=args.max_response_chars,
    )

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_train, "w") as f:
        json.dump(train_examples, f, indent=2)
    with open(args.out_test, "w") as f:
        json.dump(test_examples, f, indent=2)

    print(f"[SAVED] {args.out_train} ({len(train_examples)} rows)")
    print(f"[SAVED] {args.out_test} ({len(test_examples)} rows)")


if __name__ == "__main__":
    main()
