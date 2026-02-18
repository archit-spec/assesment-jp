"""
Track B – SFT Training Script (trl 0.28 / transformers 5.2 / T4-optimized)
============================================================================
Fine-tunes Qwen/Qwen2.5-Coder-1.5B on the Track B dataset using LoRA.

Usage:
    python train_sft_trackb.py
    python train_sft_trackb.py --model Qwen/Qwen2.5-Coder-1.5B --epochs 3
"""
import argparse, json, os, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B")
    p.add_argument("--data", default="data/sft_trackb_train.json")
    p.add_argument("--output", default="results/track_b_sft")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[CONFIG] model={args.model}")
    print(f"[CONFIG] data={args.data}")
    print(f"[CONFIG] output={args.output}")
    print(f"[CONFIG] epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}")

    # Load data and format as ChatML text field
    raw = json.load(open(args.data))
    print(f"[DATA] {len(raw)} training examples")

    dataset = Dataset.from_list([
        {
            "text": (
                f"<|im_start|>user\n{d['instruction']}<|im_end|>\n"
                f"<|im_start|>assistant\n{d['response']}<|im_end|>"
            )
        }
        for d in raw
    ])

    # Load tokenizer and model (fp16, no quantization — 1.5B fits on T4)
    print(f"[MODEL] Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )

    # LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # SFTConfig — use dataset_text_field + max_length (trl 0.28 API)
    sft_config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=1,
        save_strategy="epoch",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        report_to="wandb",
        dataloader_num_workers=0,
        dataset_text_field="text",
        max_length=args.max_len,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("[TRAIN] Starting ...")
    trainer.train()

    print(f"[SAVE] Saving to {args.output}")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print("[DONE]")


if __name__ == "__main__":
    main()
