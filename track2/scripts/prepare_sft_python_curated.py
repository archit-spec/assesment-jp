"""
Prepare a Python-friendly Track B SFT split (300-600 pairs target).
"""

import argparse
import json
from collections import Counter


DEFAULT_CATEGORIES = ["complete", "unit_test", "bugfix", "improve", "docstring"]


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare curated Python-friendly SFT data")
    p.add_argument("--train-in", default="data/sft_train.json")
    p.add_argument("--test-in", default="data/sft_test.json")
    p.add_argument("--train-out", default="data/sft_python_curated_train.json")
    p.add_argument("--test-out", default="data/sft_python_curated_test.json")
    p.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    args = p.parse_args()

    keep = set(args.categories)

    train = json.load(open(args.train_in))
    test = json.load(open(args.test_in))

    train_f = [x for x in train if x.get("category") in keep]
    test_f = [x for x in test if x.get("category") in keep]

    with open(args.train_out, "w") as f:
        json.dump(train_f, f, indent=2)
    with open(args.test_out, "w") as f:
        json.dump(test_f, f, indent=2)

    print(f"[SAVED] {args.train_out} ({len(train_f)} rows)")
    print(f"[SAVED] {args.test_out} ({len(test_f)} rows)")
    print("[TRAIN DIST]", dict(Counter(x.get("category", "unknown") for x in train_f)))
    print("[TEST DIST]", dict(Counter(x.get("category", "unknown") for x in test_f)))


if __name__ == "__main__":
    main()
