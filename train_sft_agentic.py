import os
import json
import random
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# --- Config ---
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATA_FILE = "data/sft_agentic_final.jsonl"
OUTPUT_DIR = "results/track_b_agentic"

# --- Training Hyperparameters ---
MAX_SEQ_LENGTH = 1536
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1 # Keep it short for assessment/demonstration

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load tokenizer and model
    print(f"Loading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Important for SFT

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # 2. Apply LoRA
    print("Setting up LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Load Dataset
    print(f"Loading dataset {DATA_FILE}...")
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    
    # Split for validation
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # 4. SFT Setting
    print("Starting training...")
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="messages", # TRL handles list of messages automatically
        packing=False,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=5,
        save_strategy="no", # Don't save for now to save space
        eval_strategy="steps",
        eval_steps=20,
        fp16=False,
        bf16=True, # Recommended for Qwen2
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=sft_config,
    )

    trainer.train()

    # 5. Save Final Model
    print("Saving model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    print("Training complete.")

if __name__ == "__main__":
    main()
