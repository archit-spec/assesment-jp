import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

BASE_MODEL_ID    = "Qwen/Qwen2.5-Coder-3B"
FINETUNED_MODEL_ID = "archit11/qwen2.5-coder-3b-hyperswitch-track-a-merged"
DATASET_ID       = "archit11/hyperswitch-code-corpus-track-a"

# Colab T4 safe: 1024 tokens/chunk, stride=512, 20 samples max
MAX_LENGTH  = 1024
STRIDE      = 512
NUM_SAMPLES = 20   # number of dataset rows to use; reduce if still slow


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def compute_perplexity(model, encodings, max_length, stride):
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            loss = model(input_ids, labels=target_ids).loss
        nlls.append(loss.cpu())
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    return torch.exp(torch.stack(nlls).mean()).item()


def load_model(model_id, device):
    print(f"  Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load dataset once
    print(f"Loading dataset {DATASET_ID}...")
    try:
        dataset = load_dataset(DATASET_ID, split="validation")
    except Exception:
        dataset = load_dataset(DATASET_ID, split="train")
    text_column = "content" if "content" in dataset.column_names else "text"
    subset = dataset[text_column][:NUM_SAMPLES]
    print(f"Using {len(subset)} samples, column='{text_column}'")

    # Tokenize once (shared tokenizer — same vocab for both models)
    print("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_ID, trust_remote_code=True)
    encodings = tokenizer("\n\n".join(subset), return_tensors="pt")
    print(f"Sequence length: {encodings.input_ids.size(1)} tokens")

    results = {}

    # Base model
    base_model = load_model(BASE_MODEL_ID, device)
    results["Base"] = compute_perplexity(base_model, encodings, MAX_LENGTH, STRIDE)
    print(f"Base perplexity: {results['Base']:.4f}")
    del base_model
    free_memory()

    # Fine-tuned model
    ft_model = load_model(FINETUNED_MODEL_ID, device)
    results["Fine-Tuned"] = compute_perplexity(ft_model, encodings, MAX_LENGTH, STRIDE)
    print(f"Fine-tuned perplexity: {results['Fine-Tuned']:.4f}")
    del ft_model
    free_memory()

    # Summary
    delta = results["Fine-Tuned"] - results["Base"]
    pct = delta / results["Base"] * 100
    print("\n--- Results ---")
    print(f"{'Model':<15} {'Perplexity':>12}")
    print("-" * 28)
    print(f"{'Base':<15} {results['Base']:>12.4f}")
    print(f"{'Fine-Tuned':<15} {results['Fine-Tuned']:>12.4f}")
    print(f"{'Δ':<15} {delta:>+12.4f}  ({pct:+.2f}%)")


if __name__ == "__main__":
    main()