
import os
import math
import random
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

# Use the same config as Track A
MODEL_NAME = "Qwen/Qwen2.5-Coder-3B"
CORPUS_DIR = "data/code_corpus"
BLOCK_SIZE = 512
VAL_RATIO = 0.10
RANDOM_SEED = 42

def load_code_files(corpus_dir: str) -> list:
    """Load all code files as a list of (filename, content) tuples."""
    corpus_path = Path(corpus_dir)
    files = []
    print(f"[INFO] Scanning {corpus_path}...")
    for py_file in sorted(corpus_path.glob("*.py")):
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            files.append((py_file.name, f"# FILE: {py_file.name}\n{content}"))
        except Exception as e:
            print(f"[WARN] Failed to read {py_file}: {e}")
    print(f"[INFO] Loaded {len(files)} files.")
    return files

def tokenize_and_chunk(text: str, tokenizer, block_size: int) -> Dataset:
    """Tokenize text and create chunks."""
    tokenized = tokenizer(text, return_attention_mask=False)
    input_ids = tokenized["input_ids"]
    print(f"[INFO] Total tokens: {len(input_ids):,}")
    
    chunks = []
    for i in range(0, len(input_ids) - block_size, block_size):
        chunk = input_ids[i : i + block_size]
        chunks.append({"input_ids": chunk})
    
    print(f"[INFO] Created {len(chunks)} chunks of {block_size} tokens each.")
    return Dataset.from_list(chunks)

@torch.no_grad()
def compute_perplexity(model, dataset, tokenizer, batch_size: int = 8) -> float:
    """Compute perplexity on a dataset."""
    model.eval()
    device = next(model.parameters()).device
    
    total_loss = 0.0
    total_tokens = 0
    
    print(f"[INFO] Computing perplexity on {len(dataset)} chunks...")
    
    for i in range(0, len(dataset), batch_size):
        batch_items = dataset[i : i + batch_size]
        input_ids = torch.tensor(batch_items["input_ids"]).to(device)
        labels = input_ids.clone()
        
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        num_tokens = input_ids.numel()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    
    # 1. Load tokenizer and model
    print(f"[INFO] Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    
    # 2. Load corpus and split at FILE level
    all_files = load_code_files(CORPUS_DIR)
    random.shuffle(all_files)

    val_count = max(1, int(len(all_files) * VAL_RATIO))
    val_files = all_files[:val_count]
    print(f"[INFO] File-level split: {len(all_files)-val_count} train files, {val_count} val files")
    print(f"[INFO] Val files: {[name for name, _ in val_files]}")

    # 3. Tokenize only the validation set
    val_text = "\n\n".join(content for _, content in val_files)
    print(f"[INFO] Val text: {len(val_text):,} chars")
    
    if not val_text:
        print("[ERROR] No validation text!")
        return

    val_dataset = tokenize_and_chunk(val_text, tokenizer, BLOCK_SIZE)
    print(f"[INFO] Validation chunks: {len(val_dataset)}")
    
    # 5. Measure perplexity
    ppl = compute_perplexity(model, val_dataset, tokenizer)
    
    print("\n" + "="*40)
    print(f"BASELINE PERPLEXITY: {ppl:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
