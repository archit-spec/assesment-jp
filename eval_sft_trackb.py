"""
Track B – SFT Evaluation Script
=================================
Evaluates a causal LM (baseline or fine-tuned) on the Track B test set.

Scoring per category:
  - complete  → execute generated code, pass if exit code 0
  - bugfix    → execute fixed code, pass if exit code 0
  - unit_test → execute tests with pytest, pass if all assertions pass
  - docstring → syntax check + structural checks (triple quotes, Args/Returns)
  - explain   → structural checks (length, markdown structure)

Reports: pass@1, pass@k, exec_rate, style score, hallucination rate,
         per-category breakdown, top-20 failures.
"""
from __future__ import annotations
import argparse, ast, json, os, random, re, subprocess, sys, tempfile, time
from collections import defaultdict
from typing import Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED = 42
DEFAULT_MAX_SAMPLES = 60
DEFAULT_MAX_NEW_TOKENS = 384
DEFAULT_K = 3
EXEC_TIMEOUT = 8  # seconds per subprocess run

# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_path: str):
    if torch.cuda.is_available(): device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
    else: device = "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[INFO] Loading {model_path} on {device} ...")
    adapter_cfg = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        from peft import PeftModel
        base_name = json.load(open(adapter_cfg)).get("base_model_name_or_path", model_path)
        print(f"  LoRA adapter detected. Base: {base_name}")
        base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype, trust_remote_code=True)
        model = PeftModel.from_pretrained(base, model_path).merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device).eval()
    return model, tokenizer, device

# ── Generation ────────────────────────────────────────────────────────────────

def generate_response(model, tokenizer, instruction: str,
                      max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                      temperature=0.0, do_sample=False) -> str:
    messages = [{"role": "user", "content": instruction}]
    try: text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except: text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    if do_sample and temperature > 0: gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
    else: gen_kwargs["do_sample"] = False
    with torch.no_grad(): out = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

def generate_k(model, tokenizer, instruction: str, k=3,
               max_new_tokens=DEFAULT_MAX_NEW_TOKENS) -> List[str]:
    res = [generate_response(model, tokenizer, instruction, max_new_tokens=max_new_tokens)]
    for _ in range(k - 1):
        res.append(generate_response(model, tokenizer, instruction,
                                     max_new_tokens=max_new_tokens,
                                     temperature=0.6, do_sample=True))
    return res

# ── Code utilities ────────────────────────────────────────────────────────────

def extract_code(text: str) -> List[str]:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks: return [b.strip() for b in blocks if b.strip()]
    if any(kw in text for kw in ["def ", "class ", "import ", "assert "]): return [text.strip()]
    return []

def syntax_ok(code: str) -> bool:
    try: compile(code, "<e>", "exec"); return True
    except SyntaxError: return False

def execute_code(code: str, timeout: int = EXEC_TIMEOUT) -> Dict:
    """Run code in a subprocess sandbox. Returns {success, stdout, stderr, error}."""
    if not code.strip():
        return {"success": False, "stdout": "", "stderr": "empty", "error": "empty"}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            [sys.executable, fname],
            capture_output=True, text=True, timeout=timeout
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[:500],
            "stderr": result.stderr[:300],
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": "", "error": "timeout"}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": "", "error": str(e)}
    finally:
        try: os.unlink(fname)
        except: pass

def execute_pytest(code: str, timeout: int = EXEC_TIMEOUT) -> Dict:
    """Run code with pytest. Pass if all test functions pass."""
    if not code.strip():
        return {"success": False, "stdout": "", "stderr": "empty", "error": "empty"}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="test_") as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", fname, "-x", "-q", "--tb=short", "--no-header"],
            capture_output=True, text=True, timeout=timeout
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[:500],
            "stderr": result.stderr[:300],
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": "", "error": "timeout"}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": "", "error": str(e)}
    finally:
        try: os.unlink(fname)
        except: pass

# ── Per-category scorers ──────────────────────────────────────────────────────

def score_docstring(resp: str, hints: Dict) -> Dict:
    """Structural check — no execution needed for docstrings."""
    c = {
        "triple_quotes": '"""' in resp or "'''" in resp,
        "summary": len(resp.strip()) > 30,
        "args_or_returns": "Args:" in resp or "Parameters:" in resp or "Returns:" in resp,
    }
    style = sum(c.values()) / len(c)
    return {"pass1": style >= 0.67, "exec_ok": None, "style_score": style, "checks": c}

def score_complete(resp: str, hints: Dict) -> Dict:
    """Execute the completed function. Pass if it runs without error."""
    blocks = extract_code(resp)
    if not blocks:
        return {"pass1": False, "exec_ok": False, "style_score": 0.0, "checks": {"has_code": False}}
    code = blocks[0]
    syn = syntax_ok(code)
    exec_result = execute_code(code) if syn else {"success": False, "error": "syntax"}
    exec_ok = exec_result["success"]
    c = {"has_code": True, "syntax": syn, "exec": exec_ok}
    return {"pass1": exec_ok, "exec_ok": exec_ok, "style_score": sum(c.values())/len(c), "checks": c}

def score_bugfix(resp: str, hints: Dict) -> Dict:
    """Execute the fixed code. Pass if it runs without error."""
    blocks = extract_code(resp)
    bug_id = any(w in resp.lower() for w in ["bug", "fix", "error", "issue", "problem"])
    if not blocks:
        c = {"bug_identified": bug_id, "has_code": False, "syntax": False, "exec": False}
        return {"pass1": False, "exec_ok": False, "style_score": sum(c.values())/len(c), "checks": c}
    code = blocks[0]
    syn = syntax_ok(code)
    exec_result = execute_code(code) if syn else {"success": False, "error": "syntax"}
    exec_ok = exec_result["success"]
    c = {"bug_identified": bug_id, "has_code": True, "syntax": syn, "exec": exec_ok}
    return {"pass1": exec_ok, "exec_ok": exec_ok, "style_score": sum(c.values())/len(c), "checks": c}

def score_explain(resp: str, hints: Dict) -> Dict:
    """Structural check for explanation quality."""
    c = {
        "length": len(resp.split()) >= 20,
        "structure": "**" in resp or "- " in resp or "\n" in resp.strip(),
    }
    style = sum(c.values()) / len(c)
    return {"pass1": style >= 0.5, "exec_ok": None, "style_score": style, "checks": c}

def score_unittest(resp: str, hints: Dict) -> Dict:
    """Run generated tests with pytest. Pass if all assertions pass."""
    blocks = extract_code(resp)
    has_test = "def test_" in resp
    has_assert = "assert " in resp
    if not blocks or not has_test:
        c = {"test_func": has_test, "assert": has_assert, "syntax": False, "exec": False}
        return {"pass1": False, "exec_ok": False, "style_score": sum(c.values())/len(c), "checks": c}
    code = blocks[0]
    syn = syntax_ok(code)
    exec_result = execute_pytest(code) if syn else {"success": False, "error": "syntax"}
    exec_ok = exec_result["success"]
    c = {"test_func": has_test, "assert": has_assert, "syntax": syn, "exec": exec_ok}
    return {"pass1": exec_ok, "exec_ok": exec_ok, "style_score": sum(c.values())/len(c), "checks": c}

SCORERS = {
    "docstring": score_docstring,
    "complete":  score_complete,
    "bugfix":    score_bugfix,
    "explain":   score_explain,
    "unit_test": score_unittest,
}

# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate(model, tokenizer, test_data, max_samples=DEFAULT_MAX_SAMPLES,
             k=DEFAULT_K, max_new_tokens=DEFAULT_MAX_NEW_TOKENS, tag="eval"):
    random.seed(SEED)
    sampled = random.sample(test_data, min(max_samples, len(test_data)))
    results = {"tag": tag, "k": k, "n": len(sampled), "per_category": {}, "failures": []}
    cat_data = defaultdict(lambda: {"count": 0, "p1": [], "pk": [], "style": [],
                                    "exec": [], "hallu": []})
    t0 = time.time()

    for i, item in enumerate(sampled):
        cat = item.get("category", "unknown")
        print(f"  [{i+1}/{len(sampled)}] {cat:<12}", end=" ", flush=True)
        resps = generate_k(model, tokenizer, item["instruction"], k=k,
                           max_new_tokens=max_new_tokens)

        scorer = SCORERS.get(cat, score_explain)
        scores = [scorer(r, {}) for r in resps]
        s0 = scores[0]

        p1    = s0["pass1"]
        pk    = any(s["pass1"] for s in scores)
        style = max(s["style_score"] for s in scores)
        exec_ok = s0["exec_ok"]  # None for text tasks
        hallu = 1 if any(w in resps[0] for w in ["TODO", "placeholder", "NotImplemented"]) else 0

        cd = cat_data[cat]
        cd["count"] += 1
        cd["p1"].append(p1); cd["pk"].append(pk)
        cd["style"].append(style); cd["hallu"].append(hallu)
        if exec_ok is not None: cd["exec"].append(exec_ok)

        exec_str = f" exec={'✓' if exec_ok else '✗'}" if exec_ok is not None else ""
        print(f"{'✓' if p1 else '✗'} style={style:.2f}{exec_str}")

        if not p1:
            results["failures"].append({
                "idx": i, "cat": cat,
                "inst": item["instruction"][:100],
                "gen": resps[0][:200],
                "checks": s0["checks"],
            })

    n = len(sampled)
    results["pass_at_1"]  = sum(sum(d["p1"]) for d in cat_data.values()) / n
    results["pass_at_k"]  = sum(sum(d["pk"]) for d in cat_data.values()) / n
    results["avg_style"]  = sum(sum(d["style"]) for d in cat_data.values()) / n
    all_exec  = [v for d in cat_data.values() for v in d["exec"]]
    all_hallu = [v for d in cat_data.values() for v in d["hallu"]]
    results["exec_rate"]          = sum(all_exec) / len(all_exec) if all_exec else None
    results["hallucination_rate"] = sum(all_hallu) / n
    results["wall_time"]          = round(time.time() - t0, 1)

    for cat, d in cat_data.items():
        results["per_category"][cat] = {
            "n":       d["count"],
            "pass@1":  sum(d["p1"]) / d["count"],
            "pass@k":  sum(d["pk"]) / d["count"],
            "style":   sum(d["style"]) / d["count"],
            "exec_rate": sum(d["exec"]) / len(d["exec"]) if d["exec"] else None,
        }

    # Summary
    print(f"\n{'='*60}")
    print(f"  TRACK B EVAL — {tag}")
    print(f"{'='*60}")
    print(f"  Samples:        {n}")
    print(f"  pass@1:         {results['pass_at_1']:.4f}")
    print(f"  pass@{k}:         {results['pass_at_k']:.4f}")
    print(f"  Avg style:      {results['avg_style']:.4f}")
    if results["exec_rate"] is not None:
        print(f"  Exec rate:      {results['exec_rate']:.4f}  (code tasks only)")
    print(f"  Hallucination:  {results['hallucination_rate']:.4f}")
    print(f"  Wall time:      {results['wall_time']}s")
    print(f"\n  {'Category':<12} {'N':>3} {'pass@1':>7} {'pass@k':>7} {'style':>6} {'exec':>6}")
    print(f"  {'-'*12} {'-'*3} {'-'*7} {'-'*7} {'-'*6} {'-'*6}")
    for cat, d in sorted(results["per_category"].items()):
        exec_s = f"{d['exec_rate']:.2f}" if d["exec_rate"] is not None else "  n/a"
        print(f"  {cat:<12} {d['n']:>3} {d['pass@1']:>7.3f} {d['pass@k']:>7.3f} {d['style']:>6.2f} {exec_s:>6}")
    print(f"{'='*60}")
    return results

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Track B SFT Evaluator")
    sub = p.add_subparsers(dest="command")

    ep = sub.add_parser("eval")
    ep.add_argument("--model", required=True)
    ep.add_argument("--test-file", required=True)
    ep.add_argument("--tag", default="eval")
    ep.add_argument("--output")
    ep.add_argument("--k", type=int, default=DEFAULT_K)
    ep.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)

    cp = sub.add_parser("compare")
    cp.add_argument("before")
    cp.add_argument("after")

    args = p.parse_args()

    if args.command == "compare":
        b, a = json.load(open(args.before)), json.load(open(args.after))
        print(f"\n{'='*55}")
        print(f"  COMPARISON: {b['tag']} → {a['tag']}")
        print(f"{'='*55}")
        for metric in ["pass_at_1", "pass_at_k", "avg_style", "exec_rate", "hallucination_rate"]:
            bv = b.get(metric); av = a.get(metric)
            if bv is None and av is None: continue
            bv = bv or 0.0; av = av or 0.0
            delta = av - bv
            arrow = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "→")
            print(f"  {metric:<22} {bv:.3f} → {av:.3f}  {delta:+.3f} {arrow}")
        print(f"\n  Per-category pass@1:")
        all_cats = set(b.get("per_category", {})) | set(a.get("per_category", {}))
        for cat in sorted(all_cats):
            bv = b.get("per_category", {}).get(cat, {}).get("pass@1", 0)
            av = a.get("per_category", {}).get(cat, {}).get("pass@1", 0)
            print(f"    {cat:<12}  {bv:.2f} → {av:.2f}  ({av-bv:+.2f})")
        print(f"{'='*55}")
        return

    if args.command == "eval":
        data = json.load(open(args.test_file))
        print(f"[INFO] {len(data)} test samples from {args.test_file}")
        model, tokenizer, _ = load_model(args.model)
        res = evaluate(model, tokenizer, data, max_samples=args.max_samples,
                       k=args.k, tag=args.tag)
        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            json.dump(res, open(args.output, "w"), indent=2)
            print(f"[INFO] Saved → {args.output}")

if __name__ == "__main__":
    main()
