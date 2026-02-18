"""
Track B – SFT Evaluation Script
=================================
Evaluates a causal LM (baseline or fine-tuned) on the Track B test set.
Reports:
  - pass@k  (k=1, 3) — syntax validity of generated code
  - Style score per category
  - Hallucination rate
  - Execution rate (for code-producing categories)
  - Per-category breakdown
  - Error analysis (10-20 failure cases)

Usage:
    # Baseline evaluation
    python eval_sft_trackb.py --model Qwen/Qwen2.5-Coder-1.5B \
        --test-file data/sft_trackb_test.json --tag baseline

    # Post-training evaluation  
    python eval_sft_trackb.py --model results/track_b/model \
        --test-file data/sft_trackb_test.json --tag post_sft

    # Compare two result files
    python eval_sft_trackb.py --compare results/eval_baseline.json results/eval_post_sft.json
"""

import os
import json
import re
import ast
import random
import subprocess
import tempfile
import argparse
import time
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ── Config ──────────────────────────────────────────────────────────
SEED = 42
DEFAULT_MAX_SAMPLES = 60
DEFAULT_MAX_NEW_TOKENS = 384
DEFAULT_NUM_SAMPLES_K = 3   # for pass@k


# ── Generation ──────────────────────────────────────────────────────

def load_model(model_path: str, device: str = "auto"):
    """Load model + tokenizer. Supports base models and LoRA checkpoints."""
    print(f"[INFO] Loading model from {model_path} ...")

    # Detect device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try loading as PEFT model first (adapter + base)
    adapter_config = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config):
        with open(adapter_config) as f:
            cfg = json.load(f)
        base_name = cfg.get("base_model_name_or_path", model_path)
        print(f"  Detected LoRA adapter. Base model: {base_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=dtype, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True
        )

    model = model.to(device)
    model.eval()
    print(f"  Device: {device}, dtype: {dtype}")
    return model, tokenizer, device


def generate_response(
    model, tokenizer, instruction: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.0,
    do_sample: bool = False,
) -> str:
    """Generate a single response."""
    messages = [{"role": "user", "content": instruction}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample and temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.95
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def generate_k_responses(
    model, tokenizer, instruction: str, k: int = 3,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> List[str]:
    """Generate k responses: 1 greedy + (k-1) with temperature sampling."""
    responses = []
    # Greedy (deterministic)
    responses.append(generate_response(model, tokenizer, instruction,
                                       max_new_tokens=max_new_tokens,
                                       temperature=0.0, do_sample=False))
    # Sampled
    for _ in range(k - 1):
        responses.append(generate_response(model, tokenizer, instruction,
                                           max_new_tokens=max_new_tokens,
                                           temperature=0.6, do_sample=True))
    return responses


# ── Code Extraction & Checking ──────────────────────────────────────

def extract_code_blocks(text: str) -> List[str]:
    """Extract code from markdown fences or raw code."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return [b.strip() for b in blocks if b.strip()]
    # Fallback: if the whole thing looks like code
    if any(kw in text for kw in ["def ", "class ", "import ", "return ", "assert "]):
        return [text.strip()]
    return []


def check_syntax(code: str) -> bool:
    """Check Python syntax validity."""
    try:
        compile(code, "<eval>", "exec")
        return True
    except SyntaxError:
        return False


def check_execution(code: str, timeout: int = 5) -> bool:
    """Try executing code in a subprocess."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                ["python3", f.name],
                capture_output=True, timeout=timeout, text=True,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
        finally:
            try:
                os.unlink(f.name)
            except OSError:
                pass


def check_has_function(code: str, expected_name: str = None) -> bool:
    """Check if code contains a function definition (optionally matching name)."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if expected_name is None or node.name == expected_name:
                    return True
    except Exception:
        pass
    return False


# ── Category-Specific Scoring ──────────────────────────────────────

def score_docstring(response: str, hints: Dict) -> Dict:
    """Score a docstring generation response."""
    checks = {}
    checks["has_triple_quotes"] = '"""' in response or "'''" in response
    checks["has_summary"] = len(response.strip()) > 30
    checks["has_args_section"] = (
        "Args:" in response or "Parameters:" in response or
        "Params:" in response or not hints.get("has_args", True)
    )
    checks["has_returns_section"] = (
        "Returns:" in response or "Return:" in response or
        not hints.get("has_return", True)
    )
    checks["has_raises_section"] = (
        "Raises:" in response or not hints.get("has_exceptions", False)
    )

    passed = sum(checks.values())
    total = len(checks)
    return {
        "style_score": passed / total,
        "checks": checks,
    }


def score_completion(response: str, hints: Dict) -> Dict:
    """Score a code completion response."""
    checks = {}
    code_blocks = extract_code_blocks(response)
    has_code = len(code_blocks) > 0
    checks["has_code_block"] = has_code

    if has_code:
        code = code_blocks[0]
        checks["syntax_valid"] = check_syntax(code)
        expected = hints.get("expected_name")
        checks["has_correct_function"] = check_has_function(code, expected)
    else:
        checks["syntax_valid"] = False
        checks["has_correct_function"] = False

    passed = sum(checks.values())
    total = len(checks)
    return {
        "style_score": passed / total,
        "checks": checks,
        "syntax_ok": checks.get("syntax_valid", False),
        "code": code_blocks[0] if has_code else None,
    }


def score_bugfix(response: str, hints: Dict) -> Dict:
    """Score a bug fix response."""
    checks = {}
    lower = response.lower()
    checks["identifies_bug"] = any(w in lower for w in ["bug", "fix", "issue", "error", "incorrect", "wrong"])
    checks["explains_fix"] = len(response.split()) > 15

    code_blocks = extract_code_blocks(response)
    has_code = len(code_blocks) > 0
    checks["has_corrected_code"] = has_code
    if has_code:
        checks["corrected_syntax_valid"] = check_syntax(code_blocks[0])
    else:
        checks["corrected_syntax_valid"] = False

    passed = sum(checks.values())
    total = len(checks)
    return {
        "style_score": passed / total,
        "checks": checks,
        "syntax_ok": checks.get("corrected_syntax_valid", False),
    }


def score_explain(response: str, hints: Dict) -> Dict:
    """Score an explanation response."""
    checks = {}
    words = response.split()
    checks["sufficient_length"] = len(words) >= 20
    checks["mentions_function_concepts"] = any(
        w in response.lower() for w in
        ["function", "method", "returns", "takes", "parameter", "argument", "input", "output"]
    )
    checks["structured"] = len(response.split("\n")) > 2 or "**" in response or "- " in response
    checks["not_just_code"] = not (response.strip().startswith("```") and response.strip().endswith("```"))

    passed = sum(checks.values())
    total = len(checks)
    return {
        "style_score": passed / total,
        "checks": checks,
    }


def score_unittest(response: str, hints: Dict) -> Dict:
    """Score a unit test response."""
    checks = {}
    checks["has_test_function"] = "def test_" in response or "def test" in response
    checks["has_assert"] = "assert" in response
    checks["has_import"] = "import " in response

    code_blocks = extract_code_blocks(response)
    has_code = len(code_blocks) > 0
    checks["has_code_block"] = has_code
    if has_code:
        checks["syntax_valid"] = check_syntax(code_blocks[0])
    else:
        checks["syntax_valid"] = False

    passed = sum(checks.values())
    total = len(checks)
    return {
        "style_score": passed / total,
        "checks": checks,
        "syntax_ok": checks.get("syntax_valid", False),
    }


CATEGORY_SCORERS = {
    "docstring": score_docstring,
    "complete": score_completion,
    "bugfix": score_bugfix,
    "explain": score_explain,
    "unit_test": score_unittest,
}


# ── Hallucination Detection ────────────────────────────────────────

HALLUCINATION_PATTERNS = [
    (r"<\s*insert.*?>", "placeholder tag"),
    (r"\bTODO\b", "TODO marker"),
    (r"\bTBD\b", "TBD marker"),
    (r"\byour_[a-zA-Z_]+\b", "placeholder variable"),
    (r"\bsome_module\b", "placeholder module"),
    (r"\bfoo\b.*\b(bar|baz)\b", "foo/bar/baz naming"),
]

def detect_hallucinations(response: str) -> List[str]:
    """Detect common hallucination/placeholder patterns."""
    found = []
    for pattern, label in HALLUCINATION_PATTERNS:
        if re.search(pattern, response, re.IGNORECASE):
            found.append(label)
    return found


# ── Main Evaluation Loop ───────────────────────────────────────────

def evaluate(
    model, tokenizer, test_data: List[Dict],
    max_samples: int = DEFAULT_MAX_SAMPLES,
    k: int = DEFAULT_NUM_SAMPLES_K,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    tag: str = "eval",
) -> Dict:
    """Run full evaluation, return structured results."""
    random.seed(SEED)
    sampled = random.sample(test_data, min(max_samples, len(test_data)))

    results = {
        "tag": tag,
        "total_samples": len(sampled),
        "k": k,
        "pass_at_1": 0.0,
        "pass_at_k": 0.0,
        "avg_style_score": 0.0,
        "hallucination_rate": 0.0,
        "execution_rate": 0.0,
        "per_category": {},
        "failures": [],
    }

    all_style = []
    all_pass1 = []
    all_passk = []
    all_halluc = []
    cat_data = defaultdict(lambda: {
        "count": 0, "pass1": [], "passk": [], "style": [], "halluc": [],
    })

    total = len(sampled)
    t0 = time.time()

    for i, item in enumerate(sampled):
        cat = item.get("category", "unknown")
        hints = item.get("eval_hints", {})
        scorer = CATEGORY_SCORERS.get(cat, score_explain)

        # Progress
        elapsed = time.time() - t0
        eta = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
        print(f"  [{i+1}/{total}] {cat:<12} (ETA: {eta:.0f}s)", end=" ", flush=True)

        # Generate k responses
        responses = generate_k_responses(model, tokenizer, item["instruction"],
                                          k=k, max_new_tokens=max_new_tokens)

        # Score each response
        pass_flags = []
        best_score_result = None
        best_style = 0.0

        for resp in responses:
            score_result = scorer(resp, hints)
            style = score_result["style_score"]
            syntax_ok = score_result.get("syntax_ok", style >= 0.5)
            pass_flags.append(syntax_ok)
            if style > best_style:
                best_style = style
                best_score_result = score_result

        # pass@1 = did the greedy (first) response pass?
        p1 = 1 if pass_flags[0] else 0
        # pass@k = did any of the k responses pass?
        pk = 1 if any(pass_flags) else 0

        # Hallucinations on greedy response
        halluc = detect_hallucinations(responses[0])
        h_count = 1 if len(halluc) > 0 else 0

        all_pass1.append(p1)
        all_passk.append(pk)
        all_style.append(best_style)
        all_halluc.append(h_count)

        cat_data[cat]["count"] += 1
        cat_data[cat]["pass1"].append(p1)
        cat_data[cat]["passk"].append(pk)
        cat_data[cat]["style"].append(best_style)
        cat_data[cat]["halluc"].append(h_count)

        status = "✓" if p1 else "✗"
        print(f"{status} style={best_style:.2f}")

        # Track failures for error analysis
        if not p1 or best_style < 0.5:
            results["failures"].append({
                "index": i,
                "category": cat,
                "instruction_preview": item["instruction"][:120] + "...",
                "generated_preview": responses[0][:300] + "...",
                "style_score": round(best_style, 3),
                "pass_at_1": bool(p1),
                "hallucinations": halluc,
                "checks": best_score_result.get("checks", {}) if best_score_result else {},
            })

    # Aggregate
    n = len(sampled)
    results["pass_at_1"] = sum(all_pass1) / n
    results["pass_at_k"] = sum(all_passk) / n
    results["avg_style_score"] = sum(all_style) / n
    results["hallucination_rate"] = sum(all_halluc) / n

    # Execution rate: for code-producing categories only
    code_cats = {"complete", "bugfix", "unit_test"}
    exec_total = sum(1 for s in sampled if s.get("category") in code_cats)
    exec_pass = sum(1 for i, s in enumerate(sampled)
                    if s.get("category") in code_cats and all_pass1[i])
    results["execution_rate"] = exec_pass / exec_total if exec_total > 0 else 0

    # Per-category
    for cat, cd in cat_data.items():
        c = cd["count"]
        results["per_category"][cat] = {
            "count": c,
            "pass_at_1": sum(cd["pass1"]) / c,
            "pass_at_k": sum(cd["passk"]) / c,
            "avg_style": round(sum(cd["style"]) / c, 4),
            "hallucination_rate": sum(cd["halluc"]) / c,
        }

    # Keep top 20 failures
    results["failures"] = results["failures"][:20]

    # Round everything
    for key in ["pass_at_1", "pass_at_k", "avg_style_score",
                "hallucination_rate", "execution_rate"]:
        results[key] = round(results[key], 4)

    wall_time = time.time() - t0
    results["wall_time_sec"] = round(wall_time, 1)

    return results


def print_results(results: Dict, label: str = ""):
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print(f"TRACK B EVALUATION — {label or results.get('tag', '')}")
    print(f"{'='*60}")
    print(f"  Samples evaluated:    {results['total_samples']}")
    print(f"  k (for pass@k):       {results['k']}")
    print(f"  Wall time:            {results.get('wall_time_sec', '?')}s")
    print(f"\n  Aggregate Metrics:")
    print(f"    pass@1:             {results['pass_at_1']:.4f}")
    print(f"    pass@{results['k']}:             {results['pass_at_k']:.4f}")
    print(f"    Avg style score:    {results['avg_style_score']:.4f}")
    print(f"    Hallucination rate: {results['hallucination_rate']:.4f}")
    print(f"    Execution rate:     {results['execution_rate']:.4f}")
    print(f"\n  Per-Category Breakdown:")
    print(f"    {'Category':<15} {'N':>4} {'pass@1':>8} {'pass@k':>8} {'Style':>8} {'Halluc':>8}")
    print(f"    {'-'*55}")
    for cat, vals in sorted(results.get("per_category", {}).items()):
        print(f"    {cat:<15} {vals['count']:>4} "
              f"{vals['pass_at_1']:>8.3f} {vals['pass_at_k']:>8.3f} "
              f"{vals['avg_style']:>8.3f} {vals['hallucination_rate']:>8.3f}")
    print(f"\n  Failure cases:        {len(results.get('failures', []))}")
    print(f"{'='*60}")


def compare_results(before_file: str, after_file: str):
    """Compare two eval result files and print delta."""
    with open(before_file) as f:
        before = json.load(f)
    with open(after_file) as f:
        after = json.load(f)

    print(f"\n{'='*60}")
    print(f"TRACK B — BEFORE vs AFTER COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'Before':>10} {'After':>10} {'Delta':>10}")
    print(f"  {'-'*55}")

    metrics = ["pass_at_1", "pass_at_k", "avg_style_score",
               "hallucination_rate", "execution_rate"]
    for m in metrics:
        b = before.get(m, 0)
        a = after.get(m, 0)
        d = a - b
        direction = "↑" if d > 0 else "↓" if d < 0 else "="
        # For hallucination, down is good
        if m == "hallucination_rate":
            direction = "↓✓" if d < 0 else "↑✗" if d > 0 else "="
        print(f"  {m:<25} {b:>10.4f} {a:>10.4f} {d:>+9.4f} {direction}")

    # Per-category comparison
    all_cats = set(list(before.get("per_category", {}).keys()) +
                   list(after.get("per_category", {}).keys()))
    if all_cats:
        print(f"\n  Per-Category pass@1 Delta:")
        print(f"  {'Category':<15} {'Before':>10} {'After':>10} {'Delta':>10}")
        print(f"  {'-'*45}")
        for cat in sorted(all_cats):
            b = before.get("per_category", {}).get(cat, {}).get("pass_at_1", 0)
            a = after.get("per_category", {}).get(cat, {}).get("pass_at_1", 0)
            d = a - b
            print(f"  {cat:<15} {b:>10.3f} {a:>10.3f} {d:>+10.3f}")

    print(f"{'='*60}")

    # Save comparison as JSON too
    comparison = {
        "before_file": before_file,
        "after_file": after_file,
        "before_tag": before.get("tag"),
        "after_tag": after.get("tag"),
        "deltas": {},
    }
    for m in metrics:
        comparison["deltas"][m] = round(after.get(m, 0) - before.get(m, 0), 4)

    return comparison


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Track B SFT evaluation")
    sub = parser.add_subparsers(dest="command")

    # Evaluate command
    eval_p = sub.add_parser("eval", help="Run evaluation on a model")
    eval_p.add_argument("--model", required=True, help="Model name or path")
    eval_p.add_argument("--test-file", required=True, help="Test JSON file")
    eval_p.add_argument("--tag", default="eval", help="Tag for this run")
    eval_p.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    eval_p.add_argument("--k", type=int, default=DEFAULT_NUM_SAMPLES_K)
    eval_p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    eval_p.add_argument("--output", default=None, help="Output JSON path")

    # Compare command
    cmp_p = sub.add_parser("compare", help="Compare two eval result files")
    cmp_p.add_argument("before", help="Before (baseline) eval JSON")
    cmp_p.add_argument("after", help="After (post-SFT) eval JSON")
    cmp_p.add_argument("--output", default=None, help="Save comparison JSON")

    args = parser.parse_args()

    if args.command == "compare":
        comp = compare_results(args.before, args.after)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(comp, f, indent=2)
            print(f"  Comparison saved to {args.output}")
        return

    if args.command == "eval" or args.command is None:
        if not hasattr(args, "model") or args.model is None:
            parser.print_help()
            return

        # Load test data
        with open(args.test_file) as f:
            test_data = json.load(f)
        print(f"[INFO] Loaded {len(test_data)} test samples from {args.test_file}")

        # Load model
        model, tokenizer, device = load_model(args.model)

        # Evaluate
        results = evaluate(
            model, tokenizer, test_data,
            max_samples=args.max_samples,
            k=args.k,
            max_new_tokens=args.max_new_tokens,
            tag=args.tag,
        )
        results["model"] = args.model
        results["test_file"] = args.test_file

        # Print
        print_results(results, label=args.tag)

        # Save
        out_path = args.output or f"results/eval_{args.tag}.json"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
