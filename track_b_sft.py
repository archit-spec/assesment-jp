"""
Track B – Instruction Tuning (SFT)
====================================
Fine-tune Qwen/Qwen2.5-Coder-3B with LoRA on synthetic instruction–response pairs.
Evaluates with pass@k, style scoring, and hallucination detection.

Designed for Google Colab T4 (16GB VRAM).
"""

import os
import json
import random
import re
import subprocess
import tempfile
from typing import List, Dict, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# ── Config ──────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-Coder-3B"
SFT_TRAIN_FILE = "data/sft_train.json"
SFT_TEST_FILE = "data/sft_test.json"
OUTPUT_DIR = "results/track_b"
METRICS_FILE = "results/track_b_metrics.json"

MAX_SEQ_LENGTH = 1024
RANDOM_SEED = 42

# Training hyperparameters
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16       # effective batch size = 16
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
FP16 = True

# LoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ── Data Formatting ────────────────────────────────────────────────

def format_chat_template(example: Dict, tokenizer) -> str:
    """Format instruction-response pair into chat template."""
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    # Use model's chat template
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        # Fallback manual template
        text = (
            f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
            f"<|im_start|>assistant\n{example['response']}<|im_end|>"
        )
    return text


def prepare_dataset(data_file: str, tokenizer) -> Dataset:
    """Load and format SFT data."""
    with open(data_file) as f:
        data = json.load(f)

    formatted = []
    for item in data:
        text = format_chat_template(item, tokenizer)
        formatted.append({"text": text, "category": item.get("category", "unknown")})

    return Dataset.from_list(formatted)


# ── Evaluation Functions ───────────────────────────────────────────

def generate_response(model, tokenizer, instruction: str, max_new_tokens: int = 512) -> str:
    """Generate a response from the model given an instruction."""
    messages = [{"role": "user", "content": instruction}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the new tokens
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def extract_code_blocks(text: str) -> List[str]:
    """Extract code blocks from a response."""
    # Try markdown code blocks first
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks
    # Fall back to the full text if it looks like code
    if any(kw in text for kw in ["def ", "class ", "import ", "return "]):
        return [text]
    return []


def check_code_syntax(code: str) -> bool:
    """Check if code is syntactically valid Python."""
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def check_code_execution(code: str, timeout: int = 5) -> bool:
    """Try to execute code (for unit test generation and completion tasks)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                ["python3", f.name],
                capture_output=True,
                timeout=timeout,
                text=True,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
        finally:
            os.unlink(f.name)


def score_style(response: str, category: str) -> float:
    """Score style quality (0-1) based on category-specific criteria."""
    score = 0.0
    checks = 0

    if category == "docstring":
        checks = 4
        if '"""' in response or "'''" in response:
            score += 1
        if "Args:" in response or "Parameters:" in response:
            score += 1
        if "Returns:" in response or "Return:" in response:
            score += 1
        if len(response.strip()) > 20:
            score += 1

    elif category == "explain":
        checks = 3
        if len(response.split()) > 20:  # at least 20 words
            score += 1
        if any(kw in response.lower() for kw in ["function", "method", "returns", "takes", "parameter"]):
            score += 1
        if len(response.split("\n")) > 1:  # multi-line
            score += 1

    elif category == "unit_test":
        checks = 3
        if "def test_" in response or "def test" in response:
            score += 1
        if "assert" in response:
            score += 1
        code_blocks = extract_code_blocks(response)
        if code_blocks and check_code_syntax(code_blocks[0]):
            score += 1

    elif category == "improve":
        checks = 3
        if len(response.split()) > 30:
            score += 1
        if any(kw in response.lower() for kw in ["improve", "suggest", "change", "better", "optimization"]):
            score += 1
        if "```" in response:
            score += 1

    elif category == "bugfix":
        checks = 3
        if "bug" in response.lower() or "fix" in response.lower() or "issue" in response.lower():
            score += 1
        if "```" in response:
            score += 1
        code_blocks = extract_code_blocks(response)
        if code_blocks and check_code_syntax(code_blocks[0]):
            score += 1

    else:  # complete
        checks = 2
        code_blocks = extract_code_blocks(response)
        if code_blocks:
            score += 1
            if check_code_syntax(code_blocks[0]):
                score += 1

    return score / checks if checks > 0 else 0.0


def detect_hallucinations(response: str) -> List[str]:
    """Detect potential hallucinations in generated code."""
    hallucinations = []

    # Check for made-up imports
    fake_import_patterns = [
        r"from\s+httpx\.(\w+)\s+import",
        r"import\s+httpx\.(\w+)",
    ]
    known_modules = {"_client", "_models", "_urls", "_content", "_config", "_types",
                     "_transports", "_auth", "_status_codes", "_decoders", "_multipart",
                     "_utils", "_exceptions"}

    for pattern in fake_import_patterns:
        matches = re.findall(pattern, response)
        for match in matches:
            if match not in known_modules:
                hallucinations.append(f"Possibly fabricated module: httpx.{match}")

    # Check for fabricated function calls
    if "httpx." in response:
        func_calls = re.findall(r"httpx\.(\w+)\(", response)
        known_funcs = {"get", "post", "put", "delete", "patch", "head", "options",
                       "request", "stream", "Client", "AsyncClient", "Request", "Response",
                       "URL", "Headers", "QueryParams", "Timeout", "Limits", "Proxy"}
        for func in func_calls:
            if func not in known_funcs and not func.startswith("_"):
                hallucinations.append(f"Possibly fabricated function: httpx.{func}")

    return hallucinations


def evaluate_model(model, tokenizer, test_data: List[Dict], num_samples: int = 50) -> Dict:
    """Full evaluation of the fine-tuned model."""
    results = {
        "total_samples": 0,
        "syntax_correct": 0,
        "executable": 0,
        "avg_style_score": 0.0,
        "hallucination_count": 0,
        "category_scores": {},
        "failures": [],
    }

    sampled = random.sample(test_data, min(num_samples, len(test_data)))
    style_scores = []

    for i, item in enumerate(sampled):
        print(f"  [{i+1}/{len(sampled)}] Evaluating {item.get('category', 'unknown')}...", end=" ")

        # Generate response
        generated = generate_response(model, tokenizer, item["instruction"])
        category = item.get("category", "unknown")

        # Extract code
        code_blocks = extract_code_blocks(generated)
        has_code = len(code_blocks) > 0
        syntax_ok = has_code and check_code_syntax(code_blocks[0])
        executable = False

        if syntax_ok and category in ("unit_test", "complete", "bugfix"):
            executable = check_code_execution(code_blocks[0])

        # Style score
        style_score = score_style(generated, category)
        style_scores.append(style_score)

        # Hallucinations
        hallucinations = detect_hallucinations(generated)

        # Update results
        results["total_samples"] += 1
        if syntax_ok:
            results["syntax_correct"] += 1
        if executable:
            results["executable"] += 1
        results["hallucination_count"] += len(hallucinations)

        # Per-category tracking
        if category not in results["category_scores"]:
            results["category_scores"][category] = {"count": 0, "syntax_ok": 0, "style_sum": 0.0}
        results["category_scores"][category]["count"] += 1
        if syntax_ok:
            results["category_scores"][category]["syntax_ok"] += 1
        results["category_scores"][category]["style_sum"] += style_score

        # Failure analysis
        if not syntax_ok or style_score < 0.5:
            results["failures"].append({
                "category": category,
                "instruction_preview": item["instruction"][:100],
                "generated_preview": generated[:200],
                "syntax_ok": syntax_ok,
                "style_score": style_score,
                "hallucinations": hallucinations,
            })

        status = "✓" if syntax_ok else "✗"
        print(f"{status} (style: {style_score:.2f})")

    # Compute aggregates
    n = results["total_samples"]
    results["pass_at_1"] = results["syntax_correct"] / n if n > 0 else 0
    results["execution_rate"] = results["executable"] / n if n > 0 else 0
    results["avg_style_score"] = sum(style_scores) / len(style_scores) if style_scores else 0
    results["hallucination_rate"] = results["hallucination_count"] / n if n > 0 else 0

    # Per-category averages
    for cat, vals in results["category_scores"].items():
        c = vals["count"]
        vals["syntax_rate"] = vals["syntax_ok"] / c if c > 0 else 0
        vals["avg_style"] = vals["style_sum"] / c if c > 0 else 0

    # Keep only top-20 failures for error analysis
    results["failures"] = results["failures"][:20]

    return results


# ── Main ────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # 1. Load tokenizer and model
    print("[INFO] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        trust_remote_code=True,
    )

    # 2. Apply LoRA
    print("[INFO] Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Load training data
    print("[INFO] Loading SFT training data...")
    train_dataset = prepare_dataset(SFT_TRAIN_FILE, tokenizer)
    print(f"[INFO] Train samples: {len(train_dataset)}")

    # 4. Load test data for evaluation
    with open(SFT_TEST_FILE) as f:
        test_data = json.load(f)
    print(f"[INFO] Test samples: {len(test_data)}")

    # 5. Baseline evaluation (before fine-tuning)
    print("\n[INFO] Running baseline evaluation...")
    model.to(device)
    baseline_results = evaluate_model(model, tokenizer, test_data, num_samples=min(30, len(test_data)))
    print(f"[RESULT] Baseline pass@1: {baseline_results['pass_at_1']:.3f}")
    print(f"[RESULT] Baseline style:  {baseline_results['avg_style_score']:.3f}")

    # 6. Training
    print("\n[INFO] Starting SFT training...")
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        fp16=FP16 and device == "cuda",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=RANDOM_SEED,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
    )

    # Create collator for instruction masking
    # Mask everything up to the start of the assistant's response
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    train_result = trainer.train()
    print(f"[INFO] Training complete. Loss: {train_result.training_loss:.4f}")

    # 7. Post-training evaluation
    print("\n[INFO] Running post-training evaluation...")
    post_results = evaluate_model(model, tokenizer, test_data, num_samples=min(30, len(test_data)))
    print(f"[RESULT] Post-training pass@1: {post_results['pass_at_1']:.3f}")
    print(f"[RESULT] Post-training style:  {post_results['avg_style_score']:.3f}")

    # 8. Save metrics
    metrics = {
        "track": "B – Instruction Tuning (SFT)",
        "model": MODEL_NAME,
        "train_samples": len(train_dataset),
        "test_samples": len(test_data),
        "epochs": NUM_EPOCHS,
        "lora_r": LORA_R,
        "training_loss": round(train_result.training_loss, 4),
        "baseline": {
            "pass_at_1": round(baseline_results["pass_at_1"], 4),
            "avg_style_score": round(baseline_results["avg_style_score"], 4),
            "hallucination_rate": round(baseline_results["hallucination_rate"], 4),
            "category_scores": baseline_results["category_scores"],
        },
        "post_training": {
            "pass_at_1": round(post_results["pass_at_1"], 4),
            "avg_style_score": round(post_results["avg_style_score"], 4),
            "hallucination_rate": round(post_results["hallucination_rate"], 4),
            "category_scores": post_results["category_scores"],
        },
        "improvement": {
            "pass_at_1_delta": round(post_results["pass_at_1"] - baseline_results["pass_at_1"], 4),
            "style_delta": round(post_results["avg_style_score"] - baseline_results["avg_style_score"], 4),
        },
        "error_analysis": post_results["failures"],
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    # 9. Print summary
    print("\n" + "=" * 60)
    print("TRACK B – INSTRUCTION TUNING RESULTS")
    print("=" * 60)
    print(f"  Model:                {MODEL_NAME}")
    print(f"  Train samples:        {len(train_dataset)}")
    print(f"  Epochs:               {NUM_EPOCHS}")
    print(f"  Training loss:        {train_result.training_loss:.4f}")
    print(f"  Baseline pass@1:      {baseline_results['pass_at_1']:.3f}")
    print(f"  Post-train pass@1:    {post_results['pass_at_1']:.3f}")
    print(f"  Baseline style:       {baseline_results['avg_style_score']:.3f}")
    print(f"  Post-train style:     {post_results['avg_style_score']:.3f}")
    print(f"  Hallucination rate:   {post_results['hallucination_rate']:.3f}")
    print(f"  Error cases:          {len(post_results['failures'])}")
    print("=" * 60)
    print(f"  Metrics saved to {METRICS_FILE}")

    # Save model
    model.save_pretrained(os.path.join(OUTPUT_DIR, "model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "model"))
    print(f"  Model saved to {OUTPUT_DIR}/model")


if __name__ == "__main__":
    main()
