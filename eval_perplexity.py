import sys
import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

def main():
    model_id = "archit11/qwen2.5-coder-3b-hyperswitch-track-a-merged"
    dataset_id = "archit11/hyperswitch-code-corpus-track-a"

    print(f"Loading tokenizer {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print(f"Loading model {model_id}...")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(device)

    print(f"Loading dataset {dataset_id}...")
    try:
        dataset = load_dataset(dataset_id, split="test")
        print("Loaded test split.")
    except Exception as e:
        dataset = load_dataset(dataset_id, split="validation")
        print("Loaded validation split.")

    text_column = "text"
    if "content" in dataset.column_names:
        text_column = "content"
    
    print(f"Using column '{text_column}' for perplexity calculation.")

    # On M4 Mac with unified memory, 4096 is reasonable; reduce if OOM
    max_length = 4096
    stride = 512
    
    encodings = tokenizer("\n\n".join(dataset[text_column]), return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    print(f"Sequence length is {seq_len}")
    
    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # Add item directly and calculate exponent at the end
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"Perplexity: {ppl.item()}")

if __name__ == "__main__":
    main()
