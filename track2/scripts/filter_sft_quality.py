"""
Quality filter + promotion for SFT instruction/response pairs.

Usage:
  uv run python filter_sft_quality.py \
    --in-file data/sft_python_curated_train.json \
    --out-file data/sft_python_curated_train_qc.json \
    --reject-file data/sft_python_curated_train_rejects.json \
    --report-file results/sft_train_qc_report.json \
    --min-score 0.68 \
    --max-per-category 120
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


CATEGORY_HINTS = {
    "explain": ("explain", "what does", "walk through"),
    "docstring": ("docstring", "documentation"),
    "bugfix": ("bug", "fix", "correct"),
    "improve": ("improve", "refactor", "suggest"),
    "unit_test": ("unit test", "pytest", "tests"),
    "complete": ("complete", "implement", "fill in"),
}

LOW_QUALITY_PATTERNS = (
    "assert True",
    "TODO:",
    "todo:",
    "placeholder",
    "lorem ipsum",
    "description.",
    "fix this",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter and promote high-quality SFT rows")
    p.add_argument("--in-file", required=True)
    p.add_argument("--out-file", required=True)
    p.add_argument("--reject-file", required=True)
    p.add_argument("--report-file", required=True)
    p.add_argument("--min-score", type=float, default=0.68)
    p.add_argument("--max-per-category", type=int, default=0)
    p.add_argument("--max-same-response", type=int, default=3)
    p.add_argument(
        "--categories",
        nargs="+",
        default=["docstring", "bugfix", "improve", "unit_test", "complete"],
    )
    return p.parse_args()


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def find_code_blocks(text: str) -> List[Tuple[str, str]]:
    return [
        (lang.lower().strip(), code.strip())
        for lang, code in re.findall(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", text or "", flags=re.DOTALL)
        if code.strip()
    ]


def has_python_code(text: str) -> bool:
    blocks = find_code_blocks(text)
    if any(lang in ("python", "py") for lang, _ in blocks):
        return True
    plain = text or ""
    return bool(re.search(r"(?m)^\s*(def |class |import |from )", plain))


def compile_python_from_text(text: str) -> bool:
    blocks = find_code_blocks(text)
    py_blocks = [code for lang, code in blocks if lang in ("python", "py")]
    if not py_blocks and has_python_code(text):
        py_blocks = [text]
    if not py_blocks:
        return False
    for code in py_blocks:
        try:
            compile(code, "<candidate>", "exec")
            return True
        except SyntaxError:
            continue
    return False


def quality_score(row: Dict, allowed_categories: set[str]) -> Tuple[float, List[str], List[str], List[str]]:
    reasons: List[str] = []
    fails: List[str] = []
    hard_fails: List[str] = []
    score = 0.0

    category = str(row.get("category", "")).strip()
    instruction = str(row.get("instruction", "")).strip()
    response = str(row.get("response", "")).strip()
    inst_norm = norm(instruction)
    resp_norm = norm(response)

    if category in allowed_categories:
        score += 0.12
        reasons.append("category_allowed")
    else:
        fails.append("category_not_allowed")
        hard_fails.append("category_not_allowed")

    if len(instruction) < 30:
        fails.append("instruction_too_short")
        hard_fails.append("instruction_too_short")
    elif len(instruction) <= 12000:
        score += 0.12
        reasons.append("instruction_len_ok")
    else:
        fails.append("instruction_too_long")

    if len(response) < 20:
        fails.append("response_too_short")
        hard_fails.append("response_too_short")
    elif len(response) <= 12000:
        score += 0.12
        reasons.append("response_len_ok")
    else:
        fails.append("response_too_long")

    hints = CATEGORY_HINTS.get(category, ())
    if hints and any(h in inst_norm for h in hints):
        score += 0.12
        reasons.append("category_instruction_aligned")
    elif hints:
        fails.append("instruction_not_aligned_to_category")

    if any(bad in response for bad in LOW_QUALITY_PATTERNS):
        fails.append("template_or_placeholder_response")
        hard_fails.append("template_or_placeholder_response")
    else:
        score += 0.12
        reasons.append("no_template_patterns")

    has_code_in_prompt = "```" in instruction
    if has_code_in_prompt:
        score += 0.08
        reasons.append("prompt_has_code_context")
    else:
        fails.append("prompt_missing_code_context")

    category_ok = False
    if category == "unit_test":
        category_ok = ("def test_" in response) or ("pytest" in resp_norm)
    elif category == "docstring":
        category_ok = ('"""' in response) or ("purpose:" in resp_norm)
    elif category == "bugfix":
        category_ok = ("corrected" in resp_norm) or ("```" in response)
    elif category == "complete":
        category_ok = "```" in response
    elif category == "improve":
        category_ok = bool(re.search(r"(?m)^\s*1[\.\)]\s+", response)) or ("suggest" in resp_norm)
    elif category == "explain":
        category_ok = len(resp_norm.split()) >= 20

    if category_ok:
        score += 0.18
        reasons.append("category_response_format_ok")
    else:
        fails.append("category_response_format_bad")

    # Reward executable Python output for coding-heavy categories.
    if category in {"unit_test", "complete", "bugfix"}:
        if compile_python_from_text(response):
            score += 0.14
            reasons.append("response_python_compiles")
        else:
            fails.append("response_python_not_compilable")
            if category in {"unit_test", "complete"}:
                hard_fails.append("response_python_not_compilable")

    return round(min(score, 1.0), 4), reasons, fails, hard_fails


def dedup_rows(rows: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for row in rows:
        key = (norm(row.get("instruction", "")), norm(row.get("response", "")), row.get("category", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def limit_per_category(rows: List[Dict], max_per_category: int) -> List[Dict]:
    if max_per_category <= 0:
        return rows
    buckets = defaultdict(list)
    for row in rows:
        buckets[row.get("category", "unknown")].append(row)
    trimmed: List[Dict] = []
    for cat, items in buckets.items():
        trimmed.extend(items[:max_per_category])
    return trimmed


def limit_response_reuse(rows: List[Dict], max_same_response: int) -> Tuple[List[Dict], List[Dict]]:
    if max_same_response <= 0:
        return rows, []
    counts = Counter()
    trimmed = []
    dropped = []
    for row in rows:
        key = norm(row.get("response", ""))
        if counts[key] >= max_same_response:
            r = dict(row)
            fails = list(r.get("quality_fails", []))
            fails.append("response_reuse_cap")
            r["quality_fails"] = fails
            dropped.append(r)
            continue
        counts[key] += 1
        trimmed.append(row)
    return trimmed, dropped


def main() -> None:
    args = parse_args()
    allowed = set(args.categories)

    src = json.loads(Path(args.in_file).read_text())
    src = dedup_rows(src)

    kept = []
    rejects = []
    fail_counter = Counter()
    score_by_cat = defaultdict(list)

    for row in src:
        score, reasons, fails, hard_fails = quality_score(row, allowed)
        annotated = dict(row)
        annotated["quality_score"] = score
        annotated["quality_reasons"] = reasons
        annotated["quality_fails"] = fails
        annotated["quality_hard_fails"] = hard_fails
        score_by_cat[row.get("category", "unknown")].append(score)
        if score >= args.min_score and not hard_fails:
            kept.append(annotated)
        else:
            rejects.append(annotated)
            for f in hard_fails or fails:
                fail_counter[f] += 1

    kept = sorted(kept, key=lambda x: x["quality_score"], reverse=True)
    kept = limit_per_category(kept, args.max_per_category)
    kept, dropped_by_reuse = limit_response_reuse(kept, args.max_same_response)
    rejects.extend(dropped_by_reuse)
    for _ in dropped_by_reuse:
        fail_counter["response_reuse_cap"] += 1

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.reject_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_file).parent.mkdir(parents=True, exist_ok=True)

    Path(args.out_file).write_text(json.dumps(kept, indent=2))
    Path(args.reject_file).write_text(json.dumps(rejects, indent=2))

    report = {
        "input_rows": len(src),
        "kept_rows": len(kept),
        "rejected_rows": len(rejects),
        "dropped_by_response_reuse": len(dropped_by_reuse),
        "keep_rate": round(len(kept) / max(1, len(src)), 4),
        "min_score": args.min_score,
        "max_same_response": args.max_same_response,
        "categories_allowed": sorted(allowed),
        "kept_by_category": dict(Counter(x.get("category", "unknown") for x in kept)),
        "reject_top_reasons": dict(fail_counter.most_common(12)),
        "avg_score_by_category": {
            cat: round(sum(vals) / max(1, len(vals)), 4) for cat, vals in score_by_cat.items()
        },
    }
    Path(args.report_file).write_text(json.dumps(report, indent=2))

    print(f"[INPUT]  {len(src)} rows")
    print(f"[KEPT]   {len(kept)} rows -> {args.out_file}")
    print(f"[REJECT] {len(rejects)} rows -> {args.reject_file}")
    print(f"[REPORT] {args.report_file}")
    print("[KEPT DIST]", report["kept_by_category"])
    print("[TOP FAILS]", report["reject_top_reasons"])


if __name__ == "__main__":
    main()
