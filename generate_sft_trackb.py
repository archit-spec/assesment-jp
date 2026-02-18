"""
Track B – SFT Data Generation (vLLM Synthetic)
=================================================
Two-stage pipeline:
  Stage 1 – Extract real Python functions from verl corpus via AST
  Stage 2 – Use the LLM to generate gold responses for each task

Five task families: docstring, complete, bugfix, explain, unit_test

Usage:
    python generate_sft_trackb.py --target 400
    python generate_sft_trackb.py --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-Coder-30B-A3B-Instruct
    python generate_sft_trackb.py --dry-run
"""
from __future__ import annotations
import argparse, ast, concurrent.futures, hashlib, json, os, random, re
import subprocess, textwrap, time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()

DEFAULT_CORPUS = "data/code_corpus_verl"
DEFAULT_TARGET = 400
DEFAULT_TEST_RATIO = 0.15
SEED = 42
MIN_FUNC_LINES = 5
MAX_FUNC_LINES = 60
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_CATEGORIES = ["docstring", "complete", "bugfix", "explain", "unit_test"]

# ── AST helpers ──────────────────────────────────────────────────────

def extract_functions(corpus_dir: str) -> List[Dict]:
    functions = []
    for root, _dirs, files in os.walk(corpus_dir):
        for fname in sorted(files):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, corpus_dir)
            try:
                src = open(fpath).read()
                tree = ast.parse(src)
            except Exception:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                try:
                    code = ast.get_source_segment(src, node)
                    if not code:
                        continue
                except Exception:
                    continue
                lines = code.split("\n")
                if not (MIN_FUNC_LINES <= len(lines) <= MAX_FUNC_LINES):
                    continue
                functions.append({
                    "name": node.name,
                    "file": rel,
                    "code": code,
                    "lines": len(lines),
                    "docstring": _docstring(node),
                    "args": _args(node),
                    "return_type": ast.unparse(node.returns) if node.returns else None,
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "has_return": any(isinstance(n, ast.Return) and n.value for n in ast.walk(node)),
                    "exceptions": list({
                        n.exc.func.id if isinstance(n.exc, ast.Call) and isinstance(n.exc.func, ast.Name) else
                        n.exc.id if isinstance(n.exc, ast.Name) else ""
                        for n in ast.walk(node)
                        if isinstance(n, ast.Raise) and n.exc
                    } - {""}),
                })
    return functions

def _docstring(node) -> Optional[str]:
    if (node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
        return node.body[0].value.value
    return None

def _args(node) -> List[Dict]:
    result = []
    for a in node.args.args:
        if a.arg in ("self", "cls"): continue
        info = {"name": a.arg}
        if a.annotation: info["type"] = ast.unparse(a.annotation)
        result.append(info)
    return result

def _strip_docstring(code: str) -> Optional[str]:
    try:
        tree = ast.parse(textwrap.dedent(code))
        func = tree.body[0]
        if not _docstring(func): return None
        func.body = func.body[1:]
        return ast.unparse(tree) if func.body else None
    except Exception:
        return None

_BUG_MUTATIONS = [
    ("Off-by-one: removed '- 1'",    " - 1",    "",           lambda c: " - 1" in c),
    ("Off-by-one: removed '+ 1'",    " + 1",    "",           lambda c: " + 1" in c),
    ("Flipped '==' to '!='",         " == ",    " != ",       lambda c: " == " in c),
    ("Flipped 'is not' to 'is'",     " is not ", " is ",      lambda c: " is not " in c),
    ("Flipped 'not in' to 'in'",     " not in ", " in ",      lambda c: " not in " in c),
    ("Commented out return",         "return ", "# return ",  lambda c: c.count("return ") == 1),
    ("Swapped True/False",           "True",    "False",      lambda c: "True" in c and c.count("True") == 1),
    ("Swapped False/True",           "False",   "True",       lambda c: "False" in c and c.count("False") == 1),
    ("Changed '<=' to '<'",          " <= ",    " < ",        lambda c: " <= " in c),
    ("Changed '>=' to '>'",          " >= ",    " > ",        lambda c: " >= " in c),
    ("Removed 'not' from condition", "if not ", "if ",        lambda c: "if not " in c),
    ("Swapped 'and' / 'or'",        " and ",   " or ",       lambda c: c.count(" and ") == 1),
]

# ── Prompt templates ─────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert Python programmer and technical writer. "
    "Produce clean, accurate, well-structured responses for coding tasks."
)

PROMPTS = {
    "docstring": (
        "Write a Google-style docstring for the following Python function. "
        "Include: one-line summary, Args section (if params exist), "
        "Returns section (if applicable), Raises section (if exceptions raised). "
        "Return ONLY the docstring text with triple quotes.\n\n"
        "```python\n{code}\n```"
    ),
    "complete": (
        "Complete the implementation of the following Python function. "
        "Return the complete function with the full body filled in.\n\n"
        "```python\n{partial}\n```"
    ),
    "bugfix": (
        "The following Python function contains a bug. "
        "First identify the bug in 1-2 sentences, then provide the corrected code.\n\n"
        "```python\n{buggy_code}\n```"
    ),
    "explain": (
        "Explain what the following Python function does. "
        "Cover: (1) purpose, (2) parameters, (3) return value, (4) notable implementation details.\n\n"
        "```python\n{code}\n```"
    ),
    "unit_test": (
        "Write pytest unit tests for the following Python function. "
        "Include at least 3 test cases: basic functionality, edge cases, and error handling. "
        "Use descriptive test names and real assert statements.\n\n"
        "```python\n{code}\n```"
    ),
}

# ── Build task input ─────────────────────────────────────────────────

def build_task_input(func: Dict, category: str) -> Optional[Dict]:
    code = func["code"]

    if category == "docstring":
        stripped = _strip_docstring(code) or code
        return {
            "prompt": PROMPTS["docstring"].format(code=stripped),
            "instruction": f"Write a Google-style docstring for the following Python function.\n\n```python\n{stripped}\n```",
        }

    elif category == "complete":
        lines = code.split("\n")
        if len(lines) < 5:
            return None
        partial_lines = [lines[0]]
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
        return {
            "prompt": PROMPTS["complete"].format(partial=partial),
            "instruction": f"Complete the implementation of the following Python function.\n\n```python\n{partial}\n```",
        }

    elif category == "bugfix":
        muts = list(_BUG_MUTATIONS)
        random.shuffle(muts)
        for label, find, replace, cond in muts:
            if cond(code):
                buggy = code.replace(find, replace, 1)
                if buggy != code:
                    return {
                        "prompt": PROMPTS["bugfix"].format(buggy_code=buggy),
                        "instruction": f"The following Python function contains a bug. Identify the bug and provide the corrected code.\n\n```python\n{buggy}\n```",
                        "bug_label": label,
                    }
        return None

    elif category == "explain":
        return {
            "prompt": PROMPTS["explain"].format(code=code),
            "instruction": f"Explain what the following Python function does.\n\n```python\n{code}\n```",
        }

    elif category == "unit_test":
        # Allow async functions — just note it in the prompt
        return {
            "prompt": PROMPTS["unit_test"].format(code=code),
            "instruction": f"Write pytest unit tests for the following Python function.\n\n```python\n{code}\n```",
        }

    return None

# ── LLM call ─────────────────────────────────────────────────────────

def call_llm(base_url, api_key, model, system, user_prompt, timeout=90, max_tokens=600, temperature=0.4) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    cmd = ["curl", "-sS", "--globoff", "--max-time", str(timeout),
           f"{base_url.rstrip('/')}/chat/completions",
           "-H", "Content-Type: application/json"]
    if api_key:
        cmd.extend(["-H", f"Authorization: Bearer {api_key}"])
    cmd.extend(["-d", json.dumps(payload)])
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"curl failed: {res.stderr.strip()[:200]}")
    raw = res.stdout.strip()
    if not raw:
        raise RuntimeError("empty response")
    obj = json.loads(raw)
    if "error" in obj:
        raise RuntimeError(str(obj["error"])[:200])
    return str(obj["choices"][0]["message"]["content"])

def detect_model(base_url: str) -> str:
    try:
        cmd = ["curl", "-sS", "--max-time", "8", f"{base_url.rstrip('/')}/models"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0:
            obj = json.loads(res.stdout)
            models = obj.get("data", [])
            if models:
                name = models[0].get("id", "")
                print(f"  [AUTO] Detected model: {name}")
                return name
    except Exception:
        pass
    return ""

# ── Validation ────────────────────────────────────────────────────────

def extract_python_blocks(text: str) -> List[str]:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", flags=re.DOTALL)
    blocks = [b.strip() for b in blocks if b.strip()]
    if blocks: return blocks
    raw = (text or "").strip()
    if re.search(r"(?m)^\s*(def |class |import |assert )", raw): return [raw]
    return []

def validate_response(response: str, category: str) -> Tuple[bool, float, List[str]]:
    errors: List[str] = []
    score = 0.0
    if len(response.strip()) < 15:
        return False, 0.0, ["too_short"]
    score += 0.2
    bad = ["assert True\n", "TODO:", "placeholder", "lorem ipsum", "your_module"]
    if any(p.lower() in response.lower() for p in bad):
        errors.append("placeholder_content")
    else:
        score += 0.1
    py_blocks = extract_python_blocks(response)
    if category in ("complete", "bugfix", "unit_test"):
        if not py_blocks:
            errors.append("missing_code_block")
        else:
            compiled = any(
                __import__("builtins").compile(b, "<g>", "exec") or True
                for b in py_blocks
                if not __import__("builtins").compile.__doc__  # always False, use try
            )
            compiled = False
            for b in py_blocks:
                try:
                    compile(b, "<g>", "exec"); compiled = True; break
                except SyntaxError: pass
            if compiled: score += 0.3
            else: errors.append("code_not_compilable")
    if category == "docstring":
        if '"""' in response or "'''" in response or len(response.split()) >= 10:
            score += 0.3
        else:
            errors.append("bad_docstring_format")
    if category == "explain":
        if len(response.split()) >= 15: score += 0.3
        else: errors.append("explanation_too_short")
    if category == "unit_test":
        code_text = "\n".join(py_blocks)
        if "def test_" in code_text and "assert " in code_text: score += 0.2
        elif "assert " in code_text: score += 0.1
        else: errors.append("no_test_assertions")
    if category == "bugfix":
        if any(w in response.lower() for w in ["bug", "fix", "issue", "error", "incorrect", "wrong"]):
            score += 0.15
        else:
            errors.append("no_bug_explanation")
    return len(errors) == 0, round(min(score, 1.0), 4), errors

# ── Main ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-dir", default=DEFAULT_CORPUS)
    p.add_argument("--target", type=int, default=DEFAULT_TARGET)
    p.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out-prefix", default="data/sft_trackb")
    p.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    p.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", ""))
    p.add_argument("--model", default=os.getenv("VLLM_MODEL", ""))
    p.add_argument("--max-tokens", type=int, default=600)
    p.add_argument("--curl-timeout", type=int, default=90)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    t0 = time.time()

    model = args.model
    if not model and not args.dry_run:
        model = detect_model(args.base_url)
        if not model:
            print("[ERROR] Could not detect model. Use --model."); return

    print(f"[CONFIG] base_url={args.base_url}  model={model}  target={args.target}")

    print(f"\n[1/4] Extracting functions from {args.corpus_dir} ...")
    funcs = extract_functions(args.corpus_dir)
    print(f"  Found {len(funcs)} functions")

    print(f"\n[2/4] Building task inputs ...")
    task_queue: List[Tuple[Dict, str, Dict]] = []
    for cat in args.categories:
        random.shuffle(funcs)
        per_cat = (args.target // len(args.categories)) + 10
        added = 0
        for func in funcs:
            if added >= per_cat: break
            ti = build_task_input(func, cat)
            if ti:
                task_queue.append((func, cat, ti))
                added += 1
        print(f"  {cat}: {added} queued")
    random.shuffle(task_queue)

    print(f"\n[3/4] Generating responses ...")
    rows: List[Dict] = []
    fail_counts: Counter = Counter()
    seen: set = set()
    target_per_cat = args.target // len(args.categories)
    cat_counts: Counter = Counter()

    def process_one(item):
        func, cat, ti = item
        if args.dry_run:
            response = f"[DRY RUN] {cat} response for {func['name']}"
        else:
            for attempt in range(1, args.max_retries + 1):
                try:
                    response = call_llm(args.base_url, args.api_key, model,
                                        SYSTEM_PROMPT, ti["prompt"],
                                        timeout=args.curl_timeout,
                                        max_tokens=args.max_tokens,
                                        temperature=args.temperature)
                    break
                except Exception as e:
                    msg = str(e)[:150]
                    if attempt == args.max_retries:
                        return {"error": msg, "category": cat}
                    time.sleep(min(15, 2 ** attempt))
        ok, score, errs = validate_response(response, cat)
        return {
            "instruction": ti["instruction"],
            "response": response,
            "category": cat,
            "source_function": func["name"],
            "source_file": func["file"],
            "quality_score": score,
            "quality_errors": errs,
            "valid": ok,
        }

    workers = max(1, args.concurrency) if not args.dry_run else 1
    total_processed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        pending_idx = 0
        for i in range(min(workers * 2, len(task_queue))):
            fut = pool.submit(process_one, task_queue[pending_idx])
            futures[fut] = pending_idx
            pending_idx += 1

        while futures and len(rows) < args.target:
            done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in done:
                futures.pop(fut)
                result = fut.result()
                total_processed += 1

                if result and "error" not in result and result.get("valid"):
                    cat = result["category"]
                    h = hashlib.md5(f"{result['instruction'][:100]}|{result['response'][:200]}".encode()).hexdigest()
                    if h not in seen and cat_counts[cat] < target_per_cat:
                        seen.add(h)
                        rows.append({k: result[k] for k in ("instruction", "response", "category", "source_function", "source_file")})
                        cat_counts[cat] += 1
                elif result and "error" in result:
                    fail_counts["api_error"] += 1
                else:
                    for e in (result or {}).get("quality_errors", []):
                        fail_counts[e] += 1

                if pending_idx < len(task_queue) and len(rows) < args.target:
                    fut = pool.submit(process_one, task_queue[pending_idx])
                    futures[fut] = pending_idx
                    pending_idx += 1

                if total_processed % 10 == 0:
                    elapsed = time.time() - t0
                    print(f"  [PROGRESS] kept={len(rows)}/{args.target} processed={total_processed} "
                          f"rate={total_processed/max(elapsed,1):.1f}/s dist={dict(cat_counts)}")

    print(f"\n[4/4] Saving ...")
    random.shuffle(rows)
    split = int(len(rows) * (1 - args.test_ratio))
    train, test = rows[:split], rows[split:]

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    for path, data in [(f"{args.out_prefix}_train.json", train),
                       (f"{args.out_prefix}_test.json", test),
                       (f"{args.out_prefix}_all.json", rows)]:
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)

    card = {
        "name": "Track B SFT – vLLM Synthetic Instruction Pairs",
        "version": "2.0",
        "source_corpus": args.corpus_dir,
        "generation_model": model,
        "generation_method": "AST instruction building + LLM gold responses",
        "random_seed": args.seed,
        "total_pairs": len(rows),
        "train_pairs": len(train),
        "test_pairs": len(test),
        "category_distribution": dict(cat_counts),
        "fail_counts": dict(fail_counts),
        "elapsed_sec": round(time.time() - t0, 1),
        "categories": {
            "docstring": "Given function (docstring stripped), generate Google-style docstring",
            "complete": "Given signature + docstring, generate function body",
            "bugfix": "Given code with injected bug, identify and fix it",
            "explain": "Given function, explain purpose/params/behavior",
            "unit_test": "Given function, generate pytest tests with assertions",
        },
    }
    with open(f"{args.out_prefix}_data_card.json", "w") as fh:
        json.dump(card, fh, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"  Model:    {model}")
    print(f"  Kept:     {len(rows)}  (train={len(train)}, test={len(test)})")
    print(f"  Elapsed:  {elapsed:.1f}s")
    print(f"  Fails:    {dict(fail_counts)}")
    print(f"  Dist:     {dict(cat_counts)}")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
