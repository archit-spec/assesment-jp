"""
Track B – SFT Data Generation (vLLM Synthetic)
=================================================
Uses the local vLLM server (or any OpenAI-compatible API) to generate
high-quality synthetic instruction–response pairs for coding tasks.

Two-stage pipeline:
  Stage 1 – Extract real Python functions from verl corpus via AST
  Stage 2 – Use the LLM to generate gold responses for each task

Five task families:
  1. Docstring Generation  – given code without docstring, generate one
  2. Code Completion       – given signature + docstring, complete body
  3. Bug Fix               – given buggy code, identify and fix the bug
  4. Code Explanation      – given code, explain what it does
  5. Unit Test Generation  – given code, generate pytest tests

Usage:
    # Using local vLLM server (default):
    python generate_sft_trackb.py --target 400

    # Using OpenRouter or Modal:
    python generate_sft_trackb.py --base-url https://openrouter.ai/api/v1 \
        --api-key sk-... --model z-ai/glm-4.7-flash

    # Dry-run (AST-only, no LLM calls):
    python generate_sft_trackb.py --dry-run
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import os
import random
import re
import subprocess
import textwrap
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# ── Defaults ────────────────────────────────────────────────────────
DEFAULT_CORPUS = "data/code_corpus_verl"
DEFAULT_TARGET = 400
DEFAULT_TEST_RATIO = 0.15
SEED = 42
MIN_FUNC_LINES = 5
MAX_FUNC_LINES = 60
MIN_BODY_LINES = 3

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = ""  # auto-detect from server
DEFAULT_CATEGORIES = ["docstring", "complete", "bugfix", "explain", "unit_test"]

# ── AST Extraction ──────────────────────────────────────────────────

def extract_functions(corpus_dir: str) -> List[Dict]:
    """Walk corpus and extract every parseable function with metadata."""
    functions = []
    for root, _dirs, files in os.walk(corpus_dir):
        for fname in sorted(files):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, corpus_dir)
            try:
                with open(fpath) as fh:
                    source = fh.read()
                tree = ast.parse(source)
            except Exception:
                continue

            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                try:
                    code = ast.get_source_segment(source, node)
                    if code is None:
                        continue
                except Exception:
                    continue

                lines = code.split("\n")
                nlines = len(lines)
                if nlines < MIN_FUNC_LINES or nlines > MAX_FUNC_LINES:
                    continue

                docstring = _get_docstring(node)
                sig = _get_signature(node)
                args_info = _get_args(node)
                ret = _get_return(node)

                functions.append({
                    "name": node.name,
                    "file": rel,
                    "code": code,
                    "lines": nlines,
                    "signature": sig,
                    "docstring": docstring,
                    "args": args_info,
                    "return_type": ret,
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "has_return_stmt": _has_return(node),
                    "num_branches": _count_branches(node),
                    "calls": _get_calls(node),
                    "exceptions": _get_exceptions(node),
                })
    return functions


def _get_docstring(node) -> Optional[str]:
    if (node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
        return node.body[0].value.value
    return None

def _get_signature(node) -> str:
    try:
        parts = []
        allargs = node.args
        doff = len(allargs.args) - len(allargs.defaults)
        for i, a in enumerate(allargs.args):
            s = a.arg
            if a.annotation: s += f": {ast.unparse(a.annotation)}"
            if i >= doff: s += f" = {ast.unparse(allargs.defaults[i - doff])}"
            parts.append(s)
        if allargs.vararg:
            s = f"*{allargs.vararg.arg}"
            if allargs.vararg.annotation: s += f": {ast.unparse(allargs.vararg.annotation)}"
            parts.append(s)
        for i, a in enumerate(allargs.kwonlyargs):
            s = a.arg
            if a.annotation: s += f": {ast.unparse(a.annotation)}"
            if allargs.kw_defaults[i] is not None: s += f" = {ast.unparse(allargs.kw_defaults[i])}"
            parts.append(s)
        if allargs.kwarg:
            s = f"**{allargs.kwarg.arg}"
            if allargs.kwarg.annotation: s += f": {ast.unparse(allargs.kwarg.annotation)}"
            parts.append(s)
        ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {node.name}({', '.join(parts)}){ret}:"
    except Exception:
        return f"def {node.name}(...):"

def _get_args(node) -> List[Dict]:
    result = []
    for a in node.args.args:
        if a.arg in ("self", "cls"): continue
        info = {"name": a.arg}
        if a.annotation: info["type"] = ast.unparse(a.annotation)
        result.append(info)
    return result

def _get_return(node) -> Optional[str]:
    return ast.unparse(node.returns) if node.returns else None

def _has_return(node) -> bool:
    return any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(node))

def _count_branches(node) -> int:
    return sum(1 for n in ast.walk(node) if isinstance(n, (ast.If, ast.IfExp)))

def _get_calls(node) -> List[str]:
    c = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name): c.append(n.func.id)
            elif isinstance(n.func, ast.Attribute): c.append(n.func.attr)
    return list(set(c))

def _get_exceptions(node) -> List[str]:
    e = []
    for n in ast.walk(node):
        if isinstance(n, ast.Raise) and n.exc:
            if isinstance(n.exc, ast.Call) and isinstance(n.exc.func, ast.Name):
                e.append(n.exc.func.id)
            elif isinstance(n.exc, ast.Name): e.append(n.exc.id)
    return list(set(e))


def _strip_docstring(code: str) -> Optional[str]:
    """Remove docstring from function code."""
    try:
        tree = ast.parse(textwrap.dedent(code))
        func = tree.body[0]
        if not _get_docstring(func):
            return None
        func.body = func.body[1:]
        if not func.body:
            return None
        return ast.unparse(tree)
    except Exception:
        return None


# ── Bug mutations ───────────────────────────────────────────────────
_BUG_MUTATIONS = [
    ("Off-by-one: removed '- 1'",    " - 1",     "",           lambda c: " - 1" in c),
    ("Off-by-one: removed '+ 1'",    " + 1",     "",           lambda c: " + 1" in c),
    ("Flipped '==' to '!='",         " == ",      " != ",      lambda c: " == " in c),
    ("Flipped 'is not' to 'is'",     " is not ",  " is ",      lambda c: " is not " in c),
    ("Flipped 'not in' to 'in'",     " not in ",  " in ",      lambda c: " not in " in c),
    ("Commented out return",         "return ",   "# return ", lambda c: c.count("return ") == 1),
    ("Swapped True/False",           "True",      "False",     lambda c: "True" in c and c.count("True") == 1),
    ("Swapped False/True",           "False",     "True",      lambda c: "False" in c and c.count("False") == 1),
    ("Changed '<=' to '<'",          " <= ",      " < ",       lambda c: " <= " in c),
    ("Changed '>=' to '>'",          " >= ",      " > ",       lambda c: " >= " in c),
    ("Removed 'not' from condition", "if not ",   "if ",       lambda c: "if not " in c),
    ("Swapped 'and' / 'or'",        " and ",     " or ",      lambda c: c.count(" and ") == 1),
]


# ── Prompt Templates for LLM ───────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert Python programmer and technical writer. "
    "You produce clean, accurate, well-structured responses for coding tasks."
)

PROMPTS = {
    "docstring": (
        "Write a Google-style docstring for the following Python function. "
        "Include a one-line summary, Args section (if parameters exist), "
        "Returns section (if applicable), and Raises section (if exceptions are raised). "
        "Return ONLY the docstring text (with triple quotes).\n\n"
        "```python\n{code}\n```"
    ),
    "complete": (
        "Complete the implementation of the following Python function. "
        "Return the complete function with the full body filled in.\n\n"
        "```python\n{partial}\n```"
    ),
    "bugfix": (
        "The following Python function contains a bug. "
        "First, identify the bug in 1-2 sentences. "
        "Then provide the corrected code.\n\n"
        "```python\n{buggy_code}\n```"
    ),
    "explain": (
        "Explain what the following Python function does. "
        "Cover: (1) its purpose, (2) parameters and their roles, "
        "(3) return value, (4) any notable implementation details.\n\n"
        "```python\n{code}\n```"
    ),
    "unit_test": (
        "Write pytest unit tests for the following Python function. "
        "Include at least 3 test cases covering: basic functionality, "
        "edge cases, and error handling. Use descriptive test names.\n\n"
        "```python\n{code}\n```"
    ),
}

# ── LLM Calling ────────────────────────────────────────────────────

def detect_model(base_url: str) -> str:
    """Auto-detect model name from the server."""
    try:
        cmd = ["curl", "-sS", "--max-time", "5",
               f"{base_url.rstrip('/')}/models"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0 and res.stdout.strip():
            obj = json.loads(res.stdout)
            models = obj.get("data", [])
            if models:
                name = models[0].get("id", "")
                print(f"  [AUTO] Detected model: {name}")
                return name
    except Exception:
        pass
    return ""


def call_llm(
    base_url: str, api_key: str, model: str,
    system: str, user_prompt: str,
    timeout: int = 60, max_tokens: int = 512,
    temperature: float = 0.4,
) -> str:
    """Call LLM via curl (same pattern as existing scripts)."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    cmd = [
        "curl", "-sS", "--globoff", "--max-time", str(timeout),
        f"{base_url.rstrip('/')}/chat/completions",
        "-H", "Content-Type: application/json",
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


# ── Task-specific input builders ────────────────────────────────────

def build_task_input(func: Dict, category: str) -> Optional[Dict]:
    """
    Build the instruction prompt + metadata for a given function and category.
    Returns None if the function is not suitable for this category.
    """
    code = func["code"]

    if category == "docstring":
        stripped = _strip_docstring(code)
        if stripped is None:
            # No docstring to strip — still valid, ask LLM to write one
            stripped = code
        prompt = PROMPTS["docstring"].format(code=stripped)
        return {
            "prompt": prompt,
            "instruction_for_dataset": (
                "Write a Google-style docstring for the following Python function.\n\n"
                f"```python\n{stripped}\n```"
            ),
            "stripped_code": stripped,
        }

    elif category == "complete":
        lines = code.split("\n")
        if len(lines) < MIN_BODY_LINES + 2:
            return None
        # Build partial: signature + docstring + placeholder
        sig_line = lines[0]
        partial_lines = [sig_line]
        doc = func["docstring"]
        if doc:
            in_doc = False
            for line in lines[1:]:
                partial_lines.append(line)
                if '"""' in line or "'''" in line:
                    if in_doc: break
                    in_doc = True
                    if line.count('"""') >= 2 or line.count("'''") >= 2: break
        partial_lines.append("    # Complete the implementation")
        partial = "\n".join(partial_lines)
        prompt = PROMPTS["complete"].format(partial=partial)
        return {
            "prompt": prompt,
            "instruction_for_dataset": (
                "Complete the implementation of the following Python function.\n\n"
                f"```python\n{partial}\n```"
            ),
            "partial_code": partial,
        }

    elif category == "bugfix":
        muts = list(_BUG_MUTATIONS)
        random.shuffle(muts)
        for label, find, replace, cond in muts:
            if cond(code):
                buggy = code.replace(find, replace, 1)
                if buggy != code:
                    prompt = PROMPTS["bugfix"].format(buggy_code=buggy)
                    return {
                        "prompt": prompt,
                        "instruction_for_dataset": (
                            "The following Python function contains a bug. "
                            "Identify the bug and provide the corrected code.\n\n"
                            f"```python\n{buggy}\n```"
                        ),
                        "buggy_code": buggy,
                        "bug_label": label,
                        "original_code": code,
                    }
        return None

    elif category == "explain":
        prompt = PROMPTS["explain"].format(code=code)
        return {
            "prompt": prompt,
            "instruction_for_dataset": (
                "Explain what the following Python function does.\n\n"
                f"```python\n{code}\n```"
            ),
        }

    elif category == "unit_test":
        if func["is_async"]:
            return None
        prompt = PROMPTS["unit_test"].format(code=code)
        return {
            "prompt": prompt,
            "instruction_for_dataset": (
                "Write pytest unit tests for the following Python function.\n\n"
                f"```python\n{code}\n```"
            ),
        }

    return None


# ── Validation ──────────────────────────────────────────────────────

def extract_python_blocks(text: str) -> List[str]:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", flags=re.DOTALL)
    blocks = [b.strip() for b in blocks if b.strip()]
    if blocks:
        return blocks
    raw = (text or "").strip()
    if re.search(r"(?m)^\s*(def |class |import |assert )", raw):
        return [raw]
    return []


def validate_response(response: str, category: str) -> Tuple[bool, float, List[str]]:
    """Validate a generated response. Returns (ok, score, errors)."""
    errors: List[str] = []
    score = 0.0

    if len(response.strip()) < 15:
        errors.append("too_short")
        return False, 0.0, errors

    score += 0.2  # non-empty

    bad_patterns = ["assert True\n", "TODO:", "placeholder", "lorem ipsum",
                    "your_module", "some_module"]
    if any(p.lower() in response.lower() for p in bad_patterns):
        errors.append("placeholder_content")
    else:
        score += 0.1

    py_blocks = extract_python_blocks(response)

    if category in ("complete", "bugfix", "unit_test"):
        if not py_blocks:
            errors.append("missing_code_block")
        else:
            compiled = False
            for block in py_blocks:
                try:
                    compile(block, "<gen>", "exec")
                    compiled = True
                    break
                except SyntaxError:
                    continue
            if compiled:
                score += 0.3
            else:
                errors.append("code_not_compilable")

    if category == "docstring":
        if '"""' in response or "'''" in response or len(response.split()) >= 10:
            score += 0.3
        else:
            errors.append("bad_docstring_format")

    if category == "explain":
        if len(response.split()) >= 15:
            score += 0.3
        else:
            errors.append("explanation_too_short")

    if category == "unit_test":
        code_text = "\n".join(py_blocks)
        if "def test_" in code_text and "assert " in code_text:
            score += 0.2
        elif "assert " in code_text:
            score += 0.1
        else:
            errors.append("no_test_assertions")

    if category == "bugfix":
        lower = response.lower()
        if any(w in lower for w in ["bug", "fix", "issue", "error", "incorrect", "wrong"]):
            score += 0.15
        else:
            errors.append("no_bug_explanation")

    return len(errors) == 0, round(min(score, 1.0), 4), errors


# ── Main Pipeline ──────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate Track B SFT data via vLLM")
    p.add_argument("--corpus-dir", default=DEFAULT_CORPUS)
    p.add_argument("--target", type=int, default=DEFAULT_TARGET)
    p.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out-prefix", default="data/sft_trackb")
    p.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    # LLM options
    p.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", ""))
    p.add_argument("--model", default=os.getenv("VLLM_MODEL", DEFAULT_MODEL))
    p.add_argument("--max-tokens", type=int, default=600)
    p.add_argument("--curl-timeout", type=int, default=90)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--dry-run", action="store_true",
                   help="AST-only mode, no LLM calls")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    t0 = time.time()

    # Auto-detect model if not specified
    model = args.model
    if not model and not args.dry_run:
        model = detect_model(args.base_url)
        if not model:
            print("[WARN] Could not auto-detect model. Specify --model.")
            return
    print(f"[CONFIG] base_url = {args.base_url}")
    print(f"[CONFIG] model    = {model}")
    print(f"[CONFIG] target   = {args.target}")
    print(f"[CONFIG] categories = {args.categories}")
    print(f"[CONFIG] dry_run  = {args.dry_run}")

    # 1. Extract functions
    print(f"\n[1/4] Extracting functions from {args.corpus_dir} ...")
    funcs = extract_functions(args.corpus_dir)
    print(f"  Found {len(funcs)} functions ({MIN_FUNC_LINES}-{MAX_FUNC_LINES} lines)")

    # 2. Build task inputs for each category
    print(f"\n[2/4] Building task inputs ...")
    task_queue: List[Tuple[Dict, str, Dict]] = []  # (func, category, task_input)
    for cat in args.categories:
        random.shuffle(funcs)
        cat_items = 0
        per_cat = (args.target // len(args.categories)) + 5  # overshoot slightly
        for func in funcs:
            if cat_items >= per_cat:
                break
            task_input = build_task_input(func, cat)
            if task_input is not None:
                task_queue.append((func, cat, task_input))
                cat_items += 1
        print(f"  {cat}: {cat_items} task inputs queued")

    random.shuffle(task_queue)
    print(f"  Total queued: {len(task_queue)}")

    # 3. Generate responses via LLM (or dry-run)
    print(f"\n[3/4] Generating responses ...")
    rows: List[Dict] = []
    fail_counts: Counter = Counter()
    seen_hashes: set = set()
    target_per_cat = args.target // len(args.categories)
    cat_counts: Counter = Counter()

    def process_one(item: Tuple[Dict, str, Dict]) -> Optional[Dict]:
        func, cat, task_input = item

        if args.dry_run:
            # Fallback: use AST-based responses
            response = f"[DRY RUN] Placeholder for {cat} on {func['name']}"
        else:
            for attempt in range(1, args.max_retries + 1):
                try:
                    response = call_llm(
                        args.base_url, args.api_key, model,
                        SYSTEM_PROMPT, task_input["prompt"],
                        timeout=args.curl_timeout,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                    break
                except Exception as e:
                    msg = str(e)[:150]
                    if attempt == args.max_retries:
                        return {"error": msg, "category": cat}
                    wait = min(15, 2 ** attempt)
                    if "429" in msg or "rate" in msg.lower():
                        wait = min(30, 3 ** attempt)
                    time.sleep(wait)

        # Validate
        ok, score, errs = validate_response(response, cat)

        return {
            "instruction": task_input["instruction_for_dataset"],
            "response": response,
            "category": cat,
            "source_function": func["name"],
            "source_file": func["file"],
            "quality_score": score,
            "quality_errors": errs,
            "valid": ok,
        }

    total_processed = 0
    workers = max(1, args.concurrency) if not args.dry_run else 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        pending_idx = 0

        # Submit initial batch
        batch_size = min(workers * 2, len(task_queue))
        for i in range(batch_size):
            if pending_idx < len(task_queue):
                fut = pool.submit(process_one, task_queue[pending_idx])
                futures[fut] = pending_idx
                pending_idx += 1

        while futures and len(rows) < args.target:
            done, _ = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for fut in done:
                idx = futures.pop(fut)
                result = fut.result()
                total_processed += 1

                if result is None or "error" in result:
                    fail_counts["api_error"] += 1
                elif not result.get("valid", False):
                    for e in result.get("quality_errors", []):
                        fail_counts[e] += 1
                else:
                    cat = result["category"]
                    # Dedup check
                    h = hashlib.md5(
                        f"{result['instruction'][:100]}|{result['response'][:200]}".encode()
                    ).hexdigest()
                    if h in seen_hashes:
                        fail_counts["duplicate"] += 1
                    elif cat_counts[cat] >= target_per_cat:
                        pass  # category full
                    else:
                        seen_hashes.add(h)
                        rows.append(result)
                        cat_counts[cat] += 1

                # Submit more work
                if pending_idx < len(task_queue) and len(rows) < args.target:
                    fut = pool.submit(process_one, task_queue[pending_idx])
                    futures[fut] = pending_idx
                    pending_idx += 1

                # Progress
                if total_processed % 10 == 0:
                    elapsed = time.time() - t0
                    rate = total_processed / max(elapsed, 1)
                    print(f"  [PROGRESS] kept={len(rows)}/{args.target} "
                          f"processed={total_processed} "
                          f"rate={rate:.1f}/s dist={dict(cat_counts)}")

    # 4. Split and save
    print(f"\n[4/4] Saving ...")

    # Remove internal fields
    clean_rows = []
    for r in rows:
        clean_rows.append({
            "instruction": r["instruction"],
            "response": r["response"],
            "category": r["category"],
            "source_function": r.get("source_function", ""),
            "source_file": r.get("source_file", ""),
        })

    random.shuffle(clean_rows)
    split = int(len(clean_rows) * (1 - args.test_ratio))
    train = clean_rows[:split]
    test = clean_rows[split:]

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    train_f = f"{args.out_prefix}_train.json"
    test_f = f"{args.out_prefix}_test.json"
    all_f = f"{args.out_prefix}_all.json"
    card_f = f"{args.out_prefix}_data_card.json"

    for path, data in [(train_f, train), (test_f, test), (all_f, clean_rows)]:
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)

    # Data card
    data_card = {
        "name": "Track B SFT – vLLM Synthetic Instruction Pairs",
        "version": "2.0",
        "description": (
            "Instruction–response pairs for coding tasks, generated from the verl "
            "Python corpus. Instructions are built deterministically from AST analysis; "
            "responses are generated by an LLM for diversity and quality."
        ),
        "source_corpus": args.corpus_dir,
        "generation_model": model,
        "generation_method": "AST-based instruction building + LLM-generated gold responses",
        "random_seed": args.seed,
        "total_pairs": len(clean_rows),
        "train_pairs": len(train),
        "test_pairs": len(test),
        "category_distribution": dict(cat_counts),
        "total_processed": total_processed,
        "fail_counts": dict(fail_counts),
        "elapsed_sec": round(time.time() - t0, 1),
        "categories": {
            "docstring": "Given function code (docstring stripped if present), generate Google-style docstring",
            "complete": "Given function signature + docstring, generate the function body",
            "bugfix": "Given code with an injected bug, identify and fix the bug",
            "explain": "Given function code, explain its purpose, parameters, and behavior",
            "unit_test": "Given function code, generate pytest unit tests with assertions",
        },
        "bug_mutation_types": [m[0] for m in _BUG_MUTATIONS],
        "filtering_rules": [
            f"Source functions: {MIN_FUNC_LINES}-{MAX_FUNC_LINES} lines",
            "Responses validated for: length, placeholder detection, compilability (for code categories)",
            "De-duplicated by instruction+response hash",
            "Category-balanced to target count",
        ],
        "prompt_templates": {cat: PROMPTS[cat][:100] + "..." for cat in PROMPTS},
    }
    with open(card_f, "w") as fh:
        json.dump(data_card, fh, indent=2)

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"Track B SFT Data Generation – Summary")
    print(f"{'='*55}")
    print(f"  Model:            {model}")
    print(f"  Source functions:  {len(funcs)}")
    print(f"  Total processed:  {total_processed}")
    print(f"  Kept:             {len(clean_rows)}")
    print(f"  Train:            {len(train)}")
    print(f"  Test:             {len(test)}")
    print(f"  Elapsed:          {elapsed:.1f}s")
    print(f"  Fail breakdown:   {dict(fail_counts)}")
    print(f"\n  Category distribution:")
    for cat, cnt in sorted(cat_counts.items()):
        print(f"    {cat:<15} {cnt:>5}")
    print(f"\n  Files saved:")
    for p in [train_f, test_f, all_f, card_f]:
        print(f"    {p}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
