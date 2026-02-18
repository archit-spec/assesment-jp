"""
Generate simple/medium Python SFT data using OpenRouter via curl.

This avoids async HTTP client hangs by using per-request curl with hard timeouts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv


load_dotenv()


DEFAULT_MODEL = "z-ai/glm-4.7-flash"
DEFAULT_OUT_PREFIX = "data/sft_python_simple_curl"
DEFAULT_CATEGORIES = ["docstring", "bugfix", "improve", "unit_test", "complete"]
DEFAULT_TOPICS = [
    "calculator operations",
    "temperature conversion",
    "string normalization",
    "palindrome checker",
    "anagram checker",
    "word frequency count",
    "deduplicate list preserve order",
    "list chunking",
    "moving average",
    "running sum",
    "fibonacci iterative",
    "prime number check",
    "greatest common divisor",
    "roman numeral conversion",
    "CSV line parser",
    "query string parser",
    "simple log parser",
    "email masking helper",
    "slugify title",
    "validate username",
]


SYSTEM_PROMPT = (
    "You create high-quality Python instruction-tuning samples. "
    "Return valid JSON only with keys: instruction, response, category, difficulty, topic."
)


USER_PROMPT_TEMPLATE = """Create ONE synthetic Python training sample.

Category: {category}
Difficulty: {difficulty}
Topic seed: {topic}

Rules:
1. Task must be simple or medium difficulty Python utility/problem-solving.
2. Keep it self-contained (no external service/framework setup).
3. Avoid placeholders and fake stubs.
4. Keep instruction concise.

Category-specific rules:
- complete: response includes one final Python code block.
- bugfix: response includes short bug explanation + corrected Python code block.
- unit_test: response includes pytest tests with >=3 assert statements.
- docstring: response is only docstring content text (no markdown fences).
- improve: response includes improved Python code block + 2-4 concise bullets.

Return ONLY JSON:
{{
  "instruction": "string",
  "response": "string",
  "category": "{category}",
  "difficulty": "{difficulty}",
  "topic": "{topic}"
}}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate simple Python SFT data via curl/OpenRouter")
    p.add_argument("--target", type=int, default=320)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL))
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"))
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--api-key", default="")
    p.add_argument("--out-prefix", default=DEFAULT_OUT_PREFIX)
    p.add_argument("--curl-timeout", type=int, default=40)
    p.add_argument("--max-tokens", type=int, default=280)
    p.add_argument("--max-attempts-per-item", type=int, default=6)
    p.add_argument("--max-total-attempts", type=int, default=6000)
    p.add_argument("--progress-every", type=int, default=20)
    p.add_argument("--concurrency", type=int, default=1)
    return p.parse_args()


def extract_json_block(text: str) -> Optional[Dict]:
    text = re.sub(r"```(?:json)?\s*", "", (text or "")).strip().rstrip("`").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def extract_loose_fields(text: str) -> Optional[Dict]:
    raw = re.sub(r"```(?:json)?\s*", "", (text or "")).strip().rstrip("`").strip()
    if not raw:
        return None

    def extract_string_field(field: str) -> str:
        # Find `"field" : "` and then read until the next unescaped quote.
        m = re.search(rf'"{re.escape(field)}"\s*:\s*"', raw, flags=re.IGNORECASE)
        if not m:
            return ""
        i = m.end()
        out: List[str] = []
        escaped = False
        while i < len(raw):
            ch = raw[i]
            if escaped:
                out.append(ch)
                escaped = False
            else:
                if ch == "\\":
                    escaped = True
                elif ch == '"':
                    break
                else:
                    out.append(ch)
            i += 1
        return "".join(out).strip()

    instruction = extract_string_field("instruction")
    response = extract_string_field("response")
    category = extract_string_field("category")
    difficulty = extract_string_field("difficulty")
    topic = extract_string_field("topic")

    if not instruction or not response:
        return None

    return {
        "instruction": instruction,
        "response": response,
        "category": category,
        "difficulty": difficulty,
        "topic": topic,
    }


def extract_python_blocks(text: str) -> List[str]:
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text or "", flags=re.DOTALL | re.IGNORECASE)
    blocks = [b.strip() for b in blocks if b.strip()]
    if blocks:
        return blocks
    raw = (text or "").strip()
    if re.search(r"(?m)^\s*(def |class |import |from |assert |for |while |if )", raw):
        return [raw]
    return []


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def validate_row(row: Dict, allowed_categories: set[str]) -> Tuple[bool, float, List[str]]:
    errs: List[str] = []
    score = 0.0
    instruction = str(row.get("instruction", "")).strip()
    response = str(row.get("response", "")).strip()
    category = str(row.get("category", "")).strip()

    if category in allowed_categories:
        score += 0.15
    else:
        errs.append("invalid_category")
    if 30 <= len(instruction) <= 2200:
        score += 0.15
    else:
        errs.append("instruction_length")
    if 20 <= len(response) <= 8000:
        score += 0.10
    else:
        errs.append("response_length")

    bad_patterns = ["assert True", "TODO:", "todo:", "placeholder", "lorem ipsum"]
    if any(p in response for p in bad_patterns):
        errs.append("placeholder_content")
    else:
        score += 0.10

    py_blocks = extract_python_blocks(response)
    if category in {"complete", "bugfix", "unit_test", "improve"}:
        if not py_blocks:
            errs.append("missing_python_block")
        else:
            compiled = False
            for block in py_blocks:
                try:
                    compile(block, "<generated>", "exec")
                    compiled = True
                    break
                except SyntaxError:
                    continue
            if compiled:
                score += 0.25
            else:
                errs.append("python_not_compilable")

    if category == "unit_test":
        code = "\n".join(py_blocks)
        if code.count("assert ") >= 3:
            score += 0.15
        else:
            errs.append("unit_test_low_assert_count")

    if category == "docstring":
        if len(response.split()) >= 12:
            score += 0.15
        else:
            errs.append("docstring_format")

    if category == "improve":
        if re.search(r"(?m)^\s*[-*]\s+", response) or re.search(r"(?m)^\s*[0-9]+\.\s+", response):
            score += 0.10
        else:
            errs.append("improve_missing_bullets")

    return len(errs) == 0, round(min(score, 1.0), 4), errs


def call_openrouter_curl(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout_s: int,
    max_tokens: int,
) -> str:
    is_openrouter = "openrouter.ai" in base_url
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.6,
        "max_tokens": max_tokens,
    }
    if is_openrouter:
        payload["reasoning"] = {"enabled": False}
        payload["response_format"] = {"type": "json_object"}

    cmd = [
        "curl",
        "-sS",
        "--globoff",
        "--max-time",
        str(timeout_s),
        f"{base_url.rstrip('/')}/chat/completions",
        "-H",
        "Content-Type: application/json",
    ]
    if api_key:
        cmd.extend(["-H", f"Authorization: Bearer {api_key}"])
    cmd.extend(["-d", json.dumps(payload)])
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"curl failed: {res.stderr.strip()[:200]}")
    raw = res.stdout.strip()
    if not raw:
        raise RuntimeError("empty curl response")
    obj = json.loads(raw)
    if "error" in obj:
        raise RuntimeError(str(obj["error"])[:220])
    return str(obj["choices"][0]["message"]["content"])


def one_completion(base_url: str, api_key: str, model: str, category: str, difficulty: str, topic: str, args: argparse.Namespace) -> Optional[Dict]:
    prompt = USER_PROMPT_TEMPLATE.format(category=category, difficulty=difficulty, topic=topic)
    for attempt in range(1, args.max_attempts_per_item + 1):
        try:
            text = call_openrouter_curl(base_url, api_key, model, prompt, args.curl_timeout, args.max_tokens)
            parsed = extract_json_block(text) or extract_loose_fields(text)
            if parsed:
                return parsed
            print(f"  [WARN] Bad JSON attempt {attempt} cat={category} topic={topic}")
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            print(f"  [WARN] API attempt {attempt} cat={category}: {msg[:180]}")
            wait_s = min(25, 1.7 ** attempt)
            if "429" in msg or "rate" in msg.lower():
                wait_s = min(35, 2 ** attempt)
            time.sleep(wait_s)
    return None


def stratified_split(rows: List[Dict], test_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for r in rows:
        buckets[r.get("category", "unknown")].append(r)
    train: List[Dict] = []
    test: List[Dict] = []
    for _, items in buckets.items():
        rng.shuffle(items)
        n_test = max(1, int(len(items) * test_ratio)) if len(items) >= 3 else 1 if len(items) == 2 else 0
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    api_key = (args.api_key or os.getenv(args.api_key_env, "")).strip()
    if "openrouter.ai" in args.base_url and not api_key:
        raise RuntimeError(f"{args.api_key_env} not found in .env (required for OpenRouter)")

    categories = list(DEFAULT_CATEGORIES)
    allowed = set(categories)
    target_per_cat = args.target // len(categories)
    target_counts = {cat: target_per_cat for cat in categories}
    for i in range(args.target - sum(target_counts.values())):
        target_counts[categories[i % len(categories)]] += 1

    counts = Counter()
    fail_counts = Counter()
    fail_by_cat = Counter()
    rows: List[Dict] = []
    seen = set()
    attempts = 0
    started = time.time()

    print(f"[CONFIG] base_url={args.base_url}")
    print(f"[CONFIG] model={args.model}")
    print(f"[CONFIG] target={args.target}")
    print(f"[CONFIG] target_by_category={target_counts}")
    print(f"[CONFIG] concurrency={args.concurrency}")

    difficulties = ["simple", "medium"]

    max_workers = max(1, int(args.concurrency))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        while len(rows) < args.target and attempts < args.max_total_attempts:
            batch_specs: List[Tuple[str, str, str]] = []
            batch_n = min(max_workers, args.target - len(rows), args.max_total_attempts - attempts)

            for _ in range(batch_n):
                pending = [c for c in categories if counts[c] < target_counts[c]]
                if not pending:
                    break
                category = sorted(
                    pending,
                    key=lambda c: (target_counts[c] - counts[c], -fail_by_cat[c]),
                    reverse=True,
                )[0]
                topic = DEFAULT_TOPICS[attempts % len(DEFAULT_TOPICS)]
                difficulty = difficulties[(attempts // 3) % 2]
                attempts += 1
                batch_specs.append((category, difficulty, topic))

            if not batch_specs:
                break

            fut_to_spec = {
                pool.submit(
                    one_completion,
                    args.base_url,
                    api_key,
                    args.model,
                    spec[0],
                    spec[1],
                    spec[2],
                    args,
                ): spec
                for spec in batch_specs
            }

            for fut in concurrent.futures.as_completed(fut_to_spec):
                category, difficulty, topic = fut_to_spec[fut]
                out = fut.result()

                if counts[category] >= target_counts[category]:
                    continue

                if not out:
                    fail_counts["api_or_json_fail"] += 1
                    fail_by_cat[category] += 1
                    continue

                out["category"] = category
                out["difficulty"] = out.get("difficulty", difficulty)
                out["topic"] = out.get("topic", topic)
                out["source"] = "openrouter_synth_simple_python_curl"
                out["model"] = args.model

                ok, score, errs = validate_row(out, allowed)
                out["quality_score"] = score
                out["quality_errors"] = errs

                key = (
                    normalize_text(str(out.get("instruction", ""))),
                    normalize_text(str(out.get("response", ""))),
                    category,
                )
                if key in seen:
                    fail_counts["duplicate_row"] += 1
                    fail_by_cat[category] += 1
                    continue
                if not ok:
                    for e in errs:
                        fail_counts[e] += 1
                    fail_by_cat[category] += 1
                    continue

                seen.add(key)
                rows.append(out)
                counts[category] += 1

                if len(rows) % args.progress_every == 0:
                    elapsed = max(time.time() - started, 1e-6)
                    print(
                        f"[PROGRESS] kept={len(rows)}/{args.target} attempts={attempts} "
                        f"rate={len(rows)/elapsed:.2f} rows/s dist={dict(counts)}"
                    )

    train, test = stratified_split(rows, args.test_ratio, args.seed)
    out_prefix = args.out_prefix
    all_path = Path(f"{out_prefix}_all.json")
    train_path = Path(f"{out_prefix}_train.json")
    test_path = Path(f"{out_prefix}_test.json")
    card_path = Path(f"{out_prefix}_data_card.json")

    all_path.parent.mkdir(parents=True, exist_ok=True)
    all_path.write_text(json.dumps(rows, indent=2))
    train_path.write_text(json.dumps(train, indent=2))
    test_path.write_text(json.dumps(test, indent=2))

    summary = {
        "target": args.target,
        "kept": len(rows),
        "attempts": attempts,
        "elapsed_sec": round(time.time() - started, 2),
        "target_by_category": target_counts,
        "kept_by_category": dict(counts),
        "fail_by_category": dict(fail_by_cat),
        "fail_counts": dict(fail_counts),
        "model": args.model,
        "seed": args.seed,
        "train_rows": len(train),
        "test_rows": len(test),
    }
    card_path.write_text(json.dumps(summary, indent=2))

    print(f"[SAVED] {all_path} ({len(rows)} rows)")
    print(f"[SAVED] {train_path} ({len(train)} rows)")
    print(f"[SAVED] {test_path} ({len(test)} rows)")
    print(f"[SAVED] {card_path}")
    print("[SUMMARY]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
