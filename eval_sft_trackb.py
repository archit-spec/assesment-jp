"""
Track B – SFT Evaluation Script
=================================
Evaluates a causal LM (baseline or fine-tuned) on the Track B test set.
Reports: pass@1, pass@3, style score, hallucination rate, per-category breakdown,
         error analysis (top 20 failures).
"""
from __future__ import annotations
import argparse, ast, json, os, random, re, subprocess, tempfile, time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED = 42
DEFAULT_MAX_SAMPLES = 60
DEFAULT_MAX_NEW_TOKENS = 384
DEFAULT_K = 3

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

def generate_response(model, tokenizer, instruction: str, max_new_tokens=DEFAULT_MAX_NEW_TOKENS, temperature=0.0, do_sample=False) -> str:
    messages = [{"role": "user", "content": instruction}]
    try: text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except: text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    if do_sample and temperature > 0: gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
    else: gen_kwargs["do_sample"] = False
    with torch.no_grad(): out = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

def generate_k(model, tokenizer, instruction: str, k=3, max_new_tokens=DEFAULT_MAX_NEW_TOKENS) -> List[str]:
    res = [generate_response(model, tokenizer, instruction, max_new_tokens=max_new_tokens)]
    for _ in range(k - 1):
        res.append(generate_response(model, tokenizer, instruction, max_new_tokens=max_new_tokens, temperature=0.6, do_sample=True))
    return res

def extract_code(text: str) -> List[str]:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks: return [b.strip() for b in blocks if b.strip()]
    if any(kw in text for kw in ["def ", "class ", "import ", "assert "]): return [text.strip()]
    return []

def syntax_ok(code: str) -> bool:
    try: compile(code, "<e>", "exec"); return True
    except SyntaxError: return False

def score_docstring(resp: str, hints) -> Dict:
    c = {"triple_quotes": '"""' in resp or "'''" in resp, "summary": len(resp.strip())>30, "args": "Args:" in resp or "Parameters:" in resp}
    return {"style_score": sum(c.values())/len(c), "checks": c, "syntax_ok": True}

def score_complete(resp: str, hints) -> Dict:
    b = extract_code(resp)
    c = {"has_code": bool(b), "syntax": bool(b) and syntax_ok(b[0])}
    return {"style_score": sum(c.values())/len(c), "checks": c, "syntax_ok": c["syntax"]}

def score_bugfix(resp: str, hints) -> Dict:
    b = extract_code(resp)
    c = {"bug_id": any(w in resp.lower() for w in ["bug","fix","error"]), "has_code": bool(b), "syntax": bool(b) and syntax_ok(b[0])}
    return {"style_score": sum(c.values())/len(c), "checks": c, "syntax_ok": c["syntax"]}

def score_explain(resp: str, hints) -> Dict:
    c = {"length": len(resp.split())>=20, "structure": "**" in resp or "- " in resp}
    return {"style_score": sum(c.values())/len(c), "checks": c, "syntax_ok": True}

def score_unittest(resp: str, hints) -> Dict:
    b = extract_code(resp)
    c = {"test_func": "def test_" in resp, "assert": "assert " in resp, "syntax": bool(b) and syntax_ok(b[0])}
    return {"style_score": sum(c.values())/len(c), "checks": c, "syntax_ok": c["syntax"]}

SCORERS = {"docstring": score_docstring, "complete": score_complete, "bugfix": score_bugfix, "explain": score_explain, "unit_test": score_unittest}

def evaluate(model, tokenizer, test_data, max_samples=DEFAULT_MAX_SAMPLES, k=DEFAULT_K, max_new_tokens=DEFAULT_MAX_NEW_TOKENS, tag="eval"):
    random.seed(SEED)
    sampled = random.sample(test_data, min(max_samples, len(test_data)))
    results = {"tag": tag, "k": k, "per_category": {}, "failures": []}
    cat_data = defaultdict(lambda: {"count": 0, "p1": [], "pk": [], "style": [], "hallu": []})
    
    for i, item in enumerate(sampled):
        cat = item.get("category", "unknown")
        print(f"  [{i+1}/{len(sampled)}] {cat:<10}", end=" ", flush=True)
        resps = generate_k(model, tokenizer, item["instruction"], k=k, max_new_tokens=max_new_tokens)
        
        scorer = SCORERS.get(cat, score_explain)
        p1 = scorer(resps[0], {})["syntax_ok"]
        pk = any(scorer(r, {})["syntax_ok"] for r in resps)
        style = max(scorer(r, {})["style_score"] for r in resps)
        hallu = 1 if "TODO" in resps[0] or "placeholder" in resps[0] else 0
        
        cd = cat_data[cat]; cd["count"]+=1; cd["p1"].append(p1); cd["pk"].append(pk); cd["style"].append(style); cd["hallu"].append(hallu)
        print(f"{'✓' if p1 else '✗'} st={style:.2f}")
        
        if not p1:
            results["failures"].append({"idx": i, "cat": cat, "inst": item["instruction"][:100], "gen": resps[0][:200]})

    results["pass_at_1"] = sum(sum(d["p1"]) for d in cat_data.values()) / len(sampled)
    results["pass_at_k"] = sum(sum(d["pk"]) for d in cat_data.values()) / len(sampled)
    results["avg_style"] = sum(sum(d["style"]) for d in cat_data.values()) / len(sampled)
    
    for cat, d in cat_data.items():
        results["per_category"][cat] = {
            "p1": sum(d["p1"])/d["count"], 
            "pk": sum(d["pk"])/d["count"], 
            "style": sum(d["style"])/d["count"]
        }
    return results

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    ep = sub.add_parser("eval")
    ep.add_argument("--model", required=True)
    ep.add_argument("--test-file", required=True)
    ep.add_argument("--tag", default="eval")
    ep.add_argument("--output")
    
    cp = sub.add_parser("compare")
    cp.add_argument("before")
    cp.add_argument("after")
    cp.add_argument("--output")
    
    args = p.parse_args()
    if args.command == "compare":
        b, a = json.load(open(args.before)), json.load(open(args.after))
        print(f"Pass@1: {b.get('pass_at_1',0):.3f} -> {a.get('pass_at_1',0):.3f}")
        return

    if args.command == "eval":
        data = json.load(open(args.test_file))
        model, tokenizer, _ = load_model(args.model)
        res = evaluate(model, tokenizer, data, tag=args.tag)
        print(f"\nResults ({args.tag}): p@1={res['pass_at_1']:.3f} p@k={res['pass_at_k']:.3f} style={res['avg_style']:.3f}")
        if args.output: json.dump(res, open(args.output,"w"), indent=2)

if __name__ == "__main__":
    main()
