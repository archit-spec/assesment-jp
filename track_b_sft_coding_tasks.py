"""
Track B – Instruction Tuning (Coding Tasks)
-------------------------------------------
Small-scale LoRA SFT on synthetic instruction/response pairs.
Evaluation uses mini coding-task test set with pass@1 and pass@3.
"""

import inspect
import json
import os
import random
import re
from collections import defaultdict
from typing import Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from trl import SFTConfig, SFTTrainer
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("trl is required. Install with `uv pip install trl`.") from exc


MODEL_NAME = "Qwen/Qwen2.5-3B"
TRAIN_FILE = os.getenv("SFT_TRAIN_FILE", "data/sft_python_curated_train.json")
TEST_FILE = os.getenv("SFT_TEST_FILE", "data/sft_python_curated_test.json")
OUTPUT_DIR = os.getenv("SFT_OUTPUT_DIR", "results/track_b_coding_qwen25_3b")
METRICS_FILE = os.getenv("SFT_METRICS_FILE", "results/track_b_coding_qwen25_3b_metrics.json")

SEED = 42
TRAIN_SIZE = int(os.getenv("TRAIN_SIZE", "350"))  # Requirement target: ~300-600 synthetic pairs
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "1"))
BATCH_SIZE = 1
GRAD_ACCUM = 16
LR = 5e-5
MAX_SEQ_LENGTH = 1024
MAX_NEW_TOKENS = 256
MAX_GRAD_NORM = 0.5
EVAL_MINI_SIZE = 20
EVAL_K = 3

# Coding-task mini eval categories
EVAL_CODING_CATEGORIES = {"complete", "unit_test"}
EVAL_PROMPT_CATEGORIES = ["docstring", "bugfix", "improve", "unit_test", "complete"]


def load_json(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def sample_train(data: List[Dict], n: int, seed: int) -> List[Dict]:
    if len(data) <= n:
        return data
    rng = random.Random(seed)
    return rng.sample(data, n)


def format_prompt(instruction: str, response: str | None = None) -> str:
    if response is None:
        return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    return (
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )


def to_text_dataset(data: List[Dict]) -> Dataset:
    rows = []
    for x in data:
        rows.append(
            {
                "text": format_prompt(x["instruction"], x["response"]),
                "category": x.get("category", "unknown"),
            }
        )
    return Dataset.from_list(rows)


def extract_python_candidate(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[0]
    if any(k in text for k in ("def ", "class ", "import ", "assert ")):
        return text
    return ""


def normalize_words(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", (text or "").lower())


def token_f1(pred: str, gold: str) -> float:
    p = normalize_words(pred)
    g = normalize_words(gold)
    if not p or not g:
        return 0.0
    p_counts = defaultdict(int)
    g_counts = defaultdict(int)
    for t in p:
        p_counts[t] += 1
    for t in g:
        g_counts[t] += 1
    common = sum(min(p_counts[t], g_counts[t]) for t in p_counts.keys() & g_counts.keys())
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(g)
    return 2 * precision * recall / (precision + recall)


def is_success(candidate: str) -> bool:
    code = extract_python_candidate(candidate)
    if not code.strip():
        return False
    try:
        compile(code, "<candidate>", "exec")
        return True
    except SyntaxError:
        return False


def generate_k(model, tokenizer, instruction: str, k: int) -> List[str]:
    prompt = format_prompt(instruction, None)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=MAX_NEW_TOKENS,
            num_return_sequences=k,
            pad_token_id=tokenizer.eos_token_id,
        )
    start = inputs["input_ids"].shape[1]
    return [tokenizer.decode(out[start:], skip_special_tokens=True) for out in outputs]


def build_mini_test(test_data: List[Dict], total_size: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for x in test_data:
        by_cat[x.get("category", "unknown")].append(x)

    # Stratify across the task families from the challenge prompt.
    target_per_cat = max(1, total_size // max(1, len(EVAL_PROMPT_CATEGORIES)))
    picked: List[Dict] = []
    for cat in EVAL_PROMPT_CATEGORIES:
        pool = by_cat.get(cat, [])
        if not pool:
            continue
        take = min(target_per_cat, len(pool))
        picked.extend(rng.sample(pool, take))

    # Fill remainder from unused rows.
    if len(picked) < total_size:
        picked_ids = {id(x) for x in picked}
        remainder = [x for x in test_data if id(x) not in picked_ids]
        if remainder:
            picked.extend(rng.sample(remainder, min(total_size - len(picked), len(remainder))))
    return picked[:total_size]


def evaluate_pass_at_k(model, tokenizer, test_data: List[Dict], mini_size: int = EVAL_MINI_SIZE, k: int = EVAL_K) -> Dict:
    mini = build_mini_test(test_data, mini_size, SEED)
    if not mini:
        raise ValueError("Mini test set is empty.")

    pass1 = 0
    passk = 0
    coding_n = 0
    f1_1_sum = 0.0
    f1_k_sum = 0.0
    details = []
    for i, x in enumerate(mini, start=1):
        cands = generate_k(model, tokenizer, x["instruction"], k=k)
        f1_1 = token_f1(cands[0], x.get("response", ""))
        f1_k = max(token_f1(c, x.get("response", "")) for c in cands)
        f1_1_sum += f1_1
        f1_k_sum += f1_k

        category = x.get("category")
        s1 = False
        sk = False
        if category in EVAL_CODING_CATEGORIES:
            coding_n += 1
            s1 = is_success(cands[0])
            sk = any(is_success(c) for c in cands)
            pass1 += int(s1)
            passk += int(sk)
        details.append(
            {
                "idx": i,
                "category": category,
                "pass@1": bool(s1),
                f"pass@{k}": bool(sk),
                "token_f1@1": round(f1_1, 4),
                f"token_f1@{k}": round(f1_k, 4),
            }
        )
        print(
            f"  [{i}/{len(mini)}] {category} "
            f"p1={int(s1)} p{k}={int(sk)} f1@1={f1_1:.3f} f1@{k}={f1_k:.3f}"
        )

    n = len(mini)
    return {
        "mini_size": n,
        "coding_size": coding_n,
        "k": k,
        "pass@1": round(pass1 / coding_n, 4) if coding_n else 0.0,
        f"pass@{k}": round(passk / coding_n, 4) if coding_n else 0.0,
        "token_f1@1": round(f1_1_sum / n, 4),
        f"token_f1@{k}": round(f1_k_sum / n, 4),
        "details": details[:20],
    }


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    train_all = load_json(TRAIN_FILE)
    test_all = load_json(TEST_FILE)
    train_data = sample_train(train_all, TRAIN_SIZE, SEED)
    print(f"[INFO] Train rows (sampled): {len(train_data)} / {len(train_all)}")
    print(f"[INFO] Test rows: {len(test_all)}")

    print("[INFO] Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    print("[INFO] Applying LoRA...")
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.to(device)

    print("[INFO] Baseline mini coding eval...")
    baseline = evaluate_pass_at_k(model, tokenizer, test_all, mini_size=EVAL_MINI_SIZE, k=EVAL_K)
    print(f"[RESULT] Baseline pass@1={baseline['pass@1']}, pass@{EVAL_K}={baseline[f'pass@{EVAL_K}']}")

    print("[INFO] Preparing SFT dataset...")
    train_ds = to_text_dataset(train_data)

    cfg_kwargs = {
        "output_dir": OUTPUT_DIR,
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "learning_rate": LR,
        "weight_decay": 0.01,
        "max_grad_norm": MAX_GRAD_NORM,
        "fp16": device == "cuda",
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "report_to": "none",
        "seed": SEED,
        "max_seq_length": MAX_SEQ_LENGTH,  # older TRL
        "max_length": MAX_SEQ_LENGTH,      # newer TRL
        "dataset_text_field": "text",
        "do_train": True,
    }
    cfg_params = inspect.signature(SFTConfig.__init__).parameters
    cfg = SFTConfig(**{k: v for k, v in cfg_kwargs.items() if k in cfg_params})

    trainer_kwargs = {"model": model, "args": cfg, "train_dataset": train_ds}
    sft_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in sft_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)

    print("[INFO] Training...")
    out = trainer.train()
    train_loss = float(out.training_loss)
    print(f"[INFO] Training done. Loss={train_loss:.4f}")

    print("[INFO] Post-training mini coding eval...")
    post = evaluate_pass_at_k(model, tokenizer, test_all, mini_size=EVAL_MINI_SIZE, k=EVAL_K)
    print(f"[RESULT] Post pass@1={post['pass@1']}, pass@{EVAL_K}={post[f'pass@{EVAL_K}']}")

    metrics = {
        "track": "B – Instruction Tuning (coding tasks)",
        "model": MODEL_NAME,
        "train_rows": len(train_data),
        "test_rows": len(test_all),
        "eval_categories": sorted(EVAL_CODING_CATEGORIES),
        "mini_eval_size": post["mini_size"],
        "epochs": NUM_EPOCHS,
        "training_loss": round(train_loss, 4),
        "baseline": {
            "pass@1": baseline["pass@1"],
            f"pass@{EVAL_K}": baseline[f"pass@{EVAL_K}"],
            "token_f1@1": baseline["token_f1@1"],
            f"token_f1@{EVAL_K}": baseline[f"token_f1@{EVAL_K}"],
        },
        "post_training": {
            "pass@1": post["pass@1"],
            f"pass@{EVAL_K}": post[f"pass@{EVAL_K}"],
            "token_f1@1": post["token_f1@1"],
            f"token_f1@{EVAL_K}": post[f"token_f1@{EVAL_K}"],
        },
        "improvement": {
            "pass@1_delta": round(post["pass@1"] - baseline["pass@1"], 4),
            f"pass@{EVAL_K}_delta": round(post[f"pass@{EVAL_K}"] - baseline[f"pass@{EVAL_K}"], 4),
            "token_f1@1_delta": round(post["token_f1@1"] - baseline["token_f1@1"], 4),
            f"token_f1@{EVAL_K}_delta": round(post[f"token_f1@{EVAL_K}"] - baseline[f"token_f1@{EVAL_K}"], 4),
        },
        "baseline_details": baseline["details"],
        "post_details": post["details"],
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    model.save_pretrained(os.path.join(OUTPUT_DIR, "model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "model"))

    print("=" * 60)
    print("TRACK B CODING RESULTS")
    print("=" * 60)
    print(f"  Train rows:   {len(train_data)}")
    print(f"  Mini test:    {post['mini_size']} total / {post['coding_size']} coding")
    print(f"  Baseline:     pass@1={baseline['pass@1']}, pass@{EVAL_K}={baseline[f'pass@{EVAL_K}']}")
    print(f"  Post-train:   pass@1={post['pass@1']}, pass@{EVAL_K}={post[f'pass@{EVAL_K}']}")
    print(f"  Baseline F1:  f1@1={baseline['token_f1@1']}, f1@{EVAL_K}={baseline[f'token_f1@{EVAL_K}']}")
    print(f"  Post F1:      f1@1={post['token_f1@1']}, f1@{EVAL_K}={post[f'token_f1@{EVAL_K}']}")
    print(f"  Delta:        pass@1={metrics['improvement']['pass@1_delta']}, "
          f"pass@{EVAL_K}={metrics['improvement'][f'pass@{EVAL_K}_delta']}, "
          f"f1@1={metrics['improvement']['token_f1@1_delta']}, "
          f"f1@{EVAL_K}={metrics['improvement'][f'token_f1@{EVAL_K}_delta']}")
    print(f"  Metrics:      {METRICS_FILE}")
    print(f"  Model:        {OUTPUT_DIR}/model")


if __name__ == "__main__":
    main()
