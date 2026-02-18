"""
Track B – One-Shot Colab Script (T4 GPU)
=========================================
Run this entire file on a Google Colab T4 instance to:
  1. Install dependencies
  2. Load the dataset from Hugging Face
  3. Evaluate the baseline model
  4. Fine-tune with LoRA (3 epochs, ~60s on T4)
  5. Evaluate the fine-tuned model
  6. Print a before/after comparison

Usage:
    !python colab_track_b.py

Or paste each section into separate Colab cells.
"""

# ── 0. Install dependencies ───────────────────────────────────────────────────
import subprocess, sys

def pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

# Pin versions verified to work together on T4
pip(
    "transformers==5.2.0",
    "peft==0.18.1",
    "trl==0.28.0",
    "datasets",
    "accelerate",
    "huggingface_hub",
)

# ── 1. Imports ────────────────────────────────────────────────────────────────
import json, os, re, ast, tempfile, textwrap, time
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig

# ── 2. Config ─────────────────────────────────────────────────────────────────
BASE_MODEL    = "Qwen/Qwen2.5-Coder-1.5B"
DATASET_REPO  = "archit11/track_b_sft"
OUTPUT_DIR    = "/content/track_b_sft"
EPOCHS        = 3
BATCH_SIZE    = 4
GRAD_ACCUM    = 4
LR            = 2e-4
MAX_LEN       = 1024
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"Base model: {BASE_MODEL}")
print(f"Dataset: {DATASET_REPO}")

# ── 3. Load dataset ───────────────────────────────────────────────────────────
print("\n[DATA] Loading dataset from HF...")
hf_ds = load_dataset(DATASET_REPO)
train_data = list(hf_ds["train"])
test_data  = list(hf_ds["test"])
print(f"  Train: {len(train_data)}  Test: {len(test_data)}")

# ── 4. Evaluation helper ──────────────────────────────────────────────────────
def load_model_and_tokenizer(model_path, is_lora=False, base=BASE_MODEL):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        base if is_lora else model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    if is_lora:
        m = PeftModel.from_pretrained(m, model_path).merge_and_unload()
    m.eval()
    return m, tok

def generate(model, tokenizer, prompt, max_new=256, temperature=None):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(DEVICE)
    with torch.no_grad():
        if temperature:
            out = model.generate(**inputs, max_new_tokens=max_new,
                                 do_sample=True, temperature=temperature,
                                 pad_token_id=tokenizer.eos_token_id)
        else:
            out = model.generate(**inputs, max_new_tokens=max_new,
                                 do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

def check_pass(response, category):
    """Simple pass/fail check per category."""
    r = response.strip()
    if not r or len(r) < 10:
        return False
    if category == "docstring":
        return '"""' in r or "'''" in r
    if category == "complete":
        try:
            ast.parse(r); return True
        except:
            return "def " in r or "return" in r
    if category == "bugfix":
        return any(w in r.lower() for w in ["fix", "bug", "error", "change", "replace", "correct"])
    if category == "explain":
        return len(r.split()) >= 20
    if category == "unit_test":
        return "def test_" in r and "assert" in r
    return True

def evaluate(model, tokenizer, test_data, tag):
    print(f"\n[EVAL] Evaluating {tag} on {len(test_data)} examples...")
    results, t0 = [], time.time()
    for i, ex in enumerate(test_data):
        prompt = (
            f"<|im_start|>user\n{ex['instruction']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        resp = generate(model, tokenizer, prompt)
        passed = check_pass(resp, ex["category"])
        results.append({"category": ex["category"], "pass": passed})
        print(f"  [{i+1}/{len(test_data)}] {ex['category']:12s} {'✓' if passed else '✗'}")

    total = len(results)
    passed = sum(r["pass"] for r in results)
    by_cat = {}
    for r in results:
        c = r["category"]
        by_cat.setdefault(c, {"n": 0, "p": 0})
        by_cat[c]["n"] += 1
        by_cat[c]["p"] += r["pass"]

    print(f"\n{'='*50}")
    print(f"  {tag.upper()} RESULTS")
    print(f"{'='*50}")
    print(f"  pass@1 overall: {passed/total:.3f}  ({passed}/{total})")
    print(f"\n  Per category:")
    for cat, v in sorted(by_cat.items()):
        print(f"    {cat:12s}  {v['p']}/{v['n']}  ({v['p']/v['n']:.2f})")
    print(f"  Wall time: {time.time()-t0:.1f}s")
    return {"tag": tag, "pass@1": passed/total, "by_category": by_cat}

# ── 5. Baseline evaluation ────────────────────────────────────────────────────
print("\n[BASELINE] Loading base model...")
model, tokenizer = load_model_and_tokenizer(BASE_MODEL)
baseline_results = evaluate(model, tokenizer, test_data, "baseline")

# Free memory before training
del model
torch.cuda.empty_cache()

# ── 6. Fine-tune with LoRA ────────────────────────────────────────────────────
print("\n[TRAIN] Fine-tuning with LoRA...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto"
)

train_dataset = Dataset.from_list([
    {
        "text": (
            f"<|im_start|>user\n{d['instruction']}<|im_end|>\n"
            f"<|im_start|>assistant\n{d['response']}<|im_end|>"
        )
    }
    for d in train_data
])

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none", task_type="CAUSAL_LM",
)

# Build SFTConfig — handle API differences across trl versions
import inspect as _inspect
_sft_params = set(_inspect.signature(SFTConfig.__init__).parameters)
_sft_kwargs = dict(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    fp16=True,
    logging_steps=10,
    save_strategy="no",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    max_grad_norm=0.3,
    report_to="none",
    dataloader_num_workers=0,
    dataset_text_field="text",
)
# max_length (trl>=0.16) vs max_seq_length (older trl)
if "max_length" in _sft_params:
    _sft_kwargs["max_length"] = MAX_LEN
elif "max_seq_length" in _sft_params:
    _sft_kwargs["max_seq_length"] = MAX_LEN
sft_config = SFTConfig(**_sft_kwargs)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
)

t_start = time.time()
trainer.train()
print(f"Training done in {time.time()-t_start:.1f}s")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ── 7. Post-SFT evaluation ────────────────────────────────────────────────────
print("\n[POST-SFT] Loading fine-tuned model...")
del model
torch.cuda.empty_cache()

ft_model, ft_tokenizer = load_model_and_tokenizer(OUTPUT_DIR, is_lora=True)
postsft_results = evaluate(ft_model, ft_tokenizer, test_data, "post_sft")

# ── 8. Final comparison ───────────────────────────────────────────────────────
print("\n" + "="*50)
print("  FINAL COMPARISON")
print("="*50)
b, a = baseline_results["pass@1"], postsft_results["pass@1"]
delta = a - b
print(f"  Baseline  pass@1: {b:.3f}")
print(f"  Post-SFT  pass@1: {a:.3f}")
print(f"  Delta:           {delta:+.3f}  {'✓ IMPROVED' if delta > 0 else '✗ REGRESSED'}")
print("\n  Per-category delta:")
all_cats = set(baseline_results["by_category"]) | set(postsft_results["by_category"])
for cat in sorted(all_cats):
    bv = baseline_results["by_category"].get(cat, {"p": 0, "n": 1})
    av = postsft_results["by_category"].get(cat, {"p": 0, "n": 1})
    bd, ad = bv["p"]/bv["n"], av["p"]/av["n"]
    print(f"    {cat:12s}  {bd:.2f} → {ad:.2f}  ({ad-bd:+.2f})")
print("="*50)
