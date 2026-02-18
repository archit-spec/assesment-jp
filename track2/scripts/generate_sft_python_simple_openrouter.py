"""
Generate simple/medium Python SFT tasks via OpenRouter.

Outputs:
- data/sft_python_simple_openrouter_all.json
- data/sft_python_simple_openrouter_train.json
- data/sft_python_simple_openrouter_test.json
- data/sft_python_simple_openrouter_data_card.json

Optional:
- Upload outputs to Hugging Face dataset repo.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import httpx


load_dotenv()


DEFAULT_MODEL = "z-ai/glm-4.7-flash"
DEFAULT_OUT_PREFIX = "data/sft_python_simple_openrouter"
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
    "merge overlapping intervals",
    "interval containment",
    "count vowels and consonants",
    "safe divide utility",
    "retry backoff calculator",
    "time range overlap",
    "weekday scheduler helper",
    "JSON field extraction",
    "flatten nested list",
    "rotate array",
]


SYSTEM_PROMPT = """You create high-quality Python instruction-tuning samples.
Return valid JSON only. No markdown outside JSON."""


USER_PROMPT_TEMPLATE = """Create ONE synthetic Python training sample.

Category: {category}
Difficulty: {difficulty}
Topic seed: {topic}

Rules:
1. Task must be simple or medium difficulty Python (utility/problem-solving).
2. Keep it self-contained; no external services/framework setup.
3. Use realistic developer wording.
4. Avoid placeholders and fake stubs.
5. Keep instruction concise.

Category-specific rules:
- complete:
  - instruction asks to complete/implement a function.
  - response contains exactly one Python code block with final implementation.
- bugfix:
  - instruction includes buggy snippet and asks to fix.
  - response contains: short bug explanation + one Python code block with corrected code.
- unit_test:
  - instruction asks for pytest tests for given function.
  - response contains exactly one Python code block with >=3 concrete asserts.
  - do not use assert True placeholders.
- docstring:
  - instruction asks to write docstring for provided function.
  - response must be only docstring content text (no markdown/code fences).
- improve:
  - instruction asks for improvements/refactor for provided snippet.
  - response should include improved Python code block and 2-4 brief bullet improvements.

Return ONLY:
{{
  "instruction": "string",
  "response": "string",
  "category": "{category}",
  "difficulty": "{difficulty}",
  "topic": "{topic}"
}}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate simple Python SFT tasks with OpenRouter")
    p.add_argument("--target", type=int, default=360, help="Total rows to keep")
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL))
    p.add_argument("--provider", choices=["openrouter"], default="openrouter")
    p.add_argument("--max-attempts-per-item", type=int, default=6)
    p.add_argument("--max-total-attempts", type=int, default=5000)
    p.add_argument("--progress-every", type=int, default=20)
    p.add_argument("--concurrency", type=int, default=1, help="Use 1 for safer rate-limit behavior")
    p.add_argument("--out-prefix", default=DEFAULT_OUT_PREFIX)
    p.add_argument("--upload", action="store_true")
    p.add_argument("--hf-repo", default="")
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

    def cap(pattern: str) -> str:
        m = re.search(pattern, raw, flags=re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    # Tolerates malformed JSON with unescaped newlines in "response".
    instruction = cap(r'"instruction"\s*:\s*"(.*?)"\s*,\s*"response"')
    response = cap(r'"response"\s*:\s*"(.*?)"\s*,\s*"category"')
    category = cap(r'"category"\s*:\s*"(.*?)"\s*(?:,|\})')
    difficulty = cap(r'"difficulty"\s*:\s*"(.*?)"\s*(?:,|\})')
    topic = cap(r'"topic"\s*:\s*"(.*?)"\s*(?:,|\})')

    if not instruction or not response:
        return None

    return {
        "instruction": instruction.replace('\\"', '"'),
        "response": response.replace('\\"', '"'),
        "category": category.replace('\\"', '"'),
        "difficulty": difficulty.replace('\\"', '"'),
        "topic": topic.replace('\\"', '"'),
    }


def message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: List[str] = []
        for item in content:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                val = item.get("text") or item.get("content")
                if isinstance(val, str):
                    out.append(val)
        return "\n".join(x for x in out if x).strip()
    return str(content).strip()


def completion_text(resp: Any) -> str:
    choices = getattr(resp, "choices", None)
    if isinstance(choices, (list, tuple)) and choices:
        msg = getattr(choices[0], "message", None)
        if msg is not None:
            text = message_content_to_text(getattr(msg, "content", None))
            if text:
                return text
    if hasattr(resp, "model_dump"):
        try:
            payload = resp.model_dump()
        except Exception:
            payload = None
    else:
        payload = resp if isinstance(resp, dict) else None
    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            return message_content_to_text(msg.get("content"))
    return ""


def extract_python_blocks(text: str) -> List[str]:
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text or "", flags=re.DOTALL | re.IGNORECASE)
    blocks = [b.strip() for b in blocks if b.strip()]
    if blocks:
        return blocks

    # Fallback: treat full response as code if it looks like Python.
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
        if not py_blocks:
            errs.append("unit_test_no_code")
        else:
            code = "\n".join(py_blocks)
            if code.count("assert ") >= 3:
                score += 0.15
            else:
                errs.append("unit_test_low_assert_count")

    if category == "docstring":
        doc = response.strip()
        if len(doc.split()) >= 12:
            score += 0.15
        else:
            errs.append("docstring_format")

    if category == "improve":
        if re.search(r"(?m)^\s*[-*]\s+", response) or re.search(r"(?m)^\s*[0-9]+\.\s+", response):
            score += 0.10
        else:
            errs.append("improve_missing_bullets")

    return len(errs) == 0, round(min(score, 1.0), 4), errs


async def one_completion(
    raw_client: httpx.AsyncClient,
    api_key: str,
    model: str,
    category: str,
    difficulty: str,
    topic: str,
    max_attempts: int,
) -> Optional[Dict]:
    prompt = USER_PROMPT_TEMPLATE.format(
        category=category,
        difficulty=difficulty,
        topic=topic,
    )

    last_preview = ""
    for attempt in range(1, max_attempts + 1):
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.6,
                "max_tokens": 520,
                "reasoning": {"enabled": False},
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            try:
                payload["response_format"] = {"type": "json_object"}
                resp = await raw_client.post(
                    "/chat/completions",
                    json=payload,
                    headers=headers,
                )
                if resp.status_code >= 400:
                    raise RuntimeError(f"{resp.status_code}: {resp.text[:240]}")
            except Exception:
                payload.pop("response_format", None)
                resp = await raw_client.post(
                    "/chat/completions",
                    json=payload,
                    headers=headers,
                )
                if resp.status_code >= 400:
                    raise RuntimeError(f"{resp.status_code}: {resp.text[:240]}")

            data = resp.json()
            text = ""
            try:
                text = data["choices"][0]["message"]["content"]
            except Exception:
                text = completion_text(data)
            parsed = extract_json_block(text)
            if not parsed:
                parsed = extract_loose_fields(text)
            if parsed:
                return parsed

            last_preview = (text or "")[:240].replace("\n", " ")
            print(f"  [WARN] Bad JSON (attempt {attempt}) topic={topic} cat={category}")
            if last_preview:
                print(f"         RAW: {last_preview} ...")

        except Exception as e:  # noqa: BLE001
            msg = str(e)
            print(f"  [WARN] API error (attempt {attempt}) topic={topic} cat={category}: {msg[:180]}")
            wait_s = min(20, 1.5 ** attempt)
            if "429" in msg or "rate" in msg.lower():
                wait_s = min(30, 2 ** attempt)
            await asyncio.sleep(wait_s)

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


async def generate(args: argparse.Namespace) -> Tuple[List[Dict], Dict]:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not found in .env")

    raw_client = httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1",
        timeout=httpx.Timeout(60.0, connect=15.0, read=60.0, write=30.0),
    )

    rng = random.Random(args.seed)
    categories = list(DEFAULT_CATEGORIES)
    allowed = set(categories)
    target_per_cat = max(1, args.target // len(categories))
    difficulties = ["simple", "medium"]

    counts = Counter()
    fail_by_cat = Counter()
    fail_counts = Counter()
    rows: List[Dict] = []
    seen_keys = set()
    attempts = 0
    started = time.time()

    print(f"[CONFIG] model={args.model}")
    print(f"[CONFIG] target={args.target} ({target_per_cat} per category approx)")
    print(f"[CONFIG] categories={categories}")
    print(f"[CONFIG] concurrency={args.concurrency}")

    target_counts = {cat: target_per_cat for cat in categories}
    remainder = args.target - sum(target_counts.values())
    for i in range(remainder):
        target_counts[categories[i % len(categories)]] += 1

    while len(rows) < args.target and attempts < args.max_total_attempts:
        pending = [c for c in categories if counts[c] < target_counts[c]]
        if not pending:
            break

        # Prefer categories with highest remaining deficit; tie-break on fewer failures.
        category = sorted(
            pending,
            key=lambda c: (target_counts[c] - counts[c], -fail_by_cat[c]),
            reverse=True,
        )[0]

        if counts[category] >= target_counts[category]:
            continue

        topic = DEFAULT_TOPICS[attempts % len(DEFAULT_TOPICS)]
        difficulty = difficulties[(attempts // 3) % 2]
        attempts += 1

        out = await one_completion(
            raw_client=raw_client,
            api_key=key,
            model=args.model,
            category=category,
            difficulty=difficulty,
            topic=topic,
            max_attempts=args.max_attempts_per_item,
        )
        if not out:
            fail_counts["api_or_json_fail"] += 1
            fail_by_cat[category] += 1
            continue

        out["category"] = category
        out["difficulty"] = out.get("difficulty", difficulty)
        out["topic"] = out.get("topic", topic)

        ok, score, errs = validate_row(out, allowed)
        out["quality_score"] = score
        out["quality_errors"] = errs
        out["source"] = "openrouter_synth_simple_python"
        out["model"] = args.model

        key_d = (
            normalize_text(str(out.get("instruction", ""))),
            normalize_text(str(out.get("response", ""))),
            category,
        )
        if key_d in seen_keys:
            fail_counts["duplicate_row"] += 1
            fail_by_cat[category] += 1
            continue

        if not ok:
            for e in errs:
                fail_counts[e] += 1
            fail_by_cat[category] += 1
            continue

        seen_keys.add(key_d)
        rows.append(out)
        counts[category] += 1

        if len(rows) % args.progress_every == 0:
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"[PROGRESS] kept={len(rows)}/{args.target} attempts={attempts} "
                f"rate={len(rows)/elapsed:.2f} rows/s dist={dict(counts)}"
            )
        elif attempts % (args.progress_every * 3) == 0:
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"[STATUS] kept={len(rows)}/{args.target} attempts={attempts} "
                f"rate={len(rows)/elapsed:.2f} rows/s"
            )

    elapsed = time.time() - started
    await raw_client.aclose()
    summary = {
        "target": args.target,
        "kept": len(rows),
        "attempts": attempts,
        "elapsed_sec": round(elapsed, 2),
        "kept_by_category": dict(counts),
        "target_by_category": target_counts,
        "fail_by_category": dict(fail_by_cat),
        "fail_counts": dict(fail_counts),
        "model": args.model,
        "seed": args.seed,
    }
    return rows, summary


def save_outputs(rows: List[Dict], args: argparse.Namespace, summary: Dict) -> Dict[str, str]:
    out_prefix = args.out_prefix
    all_path = f"{out_prefix}_all.json"
    train_path = f"{out_prefix}_train.json"
    test_path = f"{out_prefix}_test.json"
    card_path = f"{out_prefix}_data_card.json"

    train, test = stratified_split(rows, args.test_ratio, args.seed)

    Path(all_path).parent.mkdir(parents=True, exist_ok=True)
    Path(all_path).write_text(json.dumps(rows, indent=2))
    Path(train_path).write_text(json.dumps(train, indent=2))
    Path(test_path).write_text(json.dumps(test, indent=2))

    card = {
        "name": "Simple Python SFT Tasks (OpenRouter Generated)",
        "version": "1.0",
        "description": "Synthetic instruction-response pairs for simple/medium Python coding tasks.",
        "generation_model": summary["model"],
        "provider": "openrouter",
        "target_rows": summary["target"],
        "kept_rows": summary["kept"],
        "train_rows": len(train),
        "test_rows": len(test),
        "kept_by_category": summary["kept_by_category"],
        "fail_counts": summary["fail_counts"],
        "quality_rules": [
            "Category allowlist",
            "Length checks",
            "No placeholder content",
            "Code categories require compilable Python block",
            "Unit tests require >=3 assert statements",
            "Deduplicate normalized instruction/response/category",
        ],
        "seed": summary["seed"],
        "created_at_epoch": int(time.time()),
    }
    Path(card_path).write_text(json.dumps(card, indent=2))

    print(f"[SAVED] {all_path} ({len(rows)} rows)")
    print(f"[SAVED] {train_path} ({len(train)} rows)")
    print(f"[SAVED] {test_path} ({len(test)} rows)")
    print(f"[SAVED] {card_path}")

    return {
        "all": all_path,
        "train": train_path,
        "test": test_path,
        "card": card_path,
    }


def upload_to_hf(paths: Dict[str, str], hf_repo: str) -> None:
    from huggingface_hub import HfApi

    token = os.getenv("HF_TOKEN", "").strip().strip('"')
    if not token:
        raise RuntimeError("HF_TOKEN missing in .env for upload")
    if not hf_repo:
        raise RuntimeError("--hf-repo is required with --upload")

    api = HfApi(token=token)
    api.create_repo(repo_id=hf_repo, repo_type="dataset", exist_ok=True)
    for name in ("all", "train", "test", "card"):
        p = Path(paths[name])
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=p.name,
            repo_id=hf_repo,
            repo_type="dataset",
            token=token,
        )
        print(f"[HF] Uploaded {p}")
    print(f"[HF] Done: https://huggingface.co/datasets/{hf_repo}")


async def async_main() -> None:
    args = parse_args()
    random.seed(args.seed)

    rows, summary = await generate(args)
    paths = save_outputs(rows, args, summary)

    print("[SUMMARY]", json.dumps(summary, indent=2))
    if args.upload:
        upload_to_hf(paths, args.hf_repo)


if __name__ == "__main__":
    asyncio.run(async_main())
