import json
import pandas as pd
from datasets import load_dataset, Dataset
import os

def format_agentic_sample(sample):
    """
    Formats a single sample into the Agentic Tool Use format.
    Expected keys in sample: 'file', 'old_str', 'new_str', 'message' (commit message)
    Adapts based on available keys.
    """
    # map keys if necessary (robustness)
    file_path = sample.get('file', sample.get('file_path', 'unknown.py'))
    old_code = sample.get('old_str', sample.get('old_code', ''))
    new_code = sample.get('new_str', sample.get('new_code', ''))
    # Use message or summary as the instruction context
    context = sample.get('message', sample.get('instruction', 'Fix the issue in the code.'))
    
    # System Prompt
    system_prompt = "You are an expert software engineer. Given a code change request, provide the exact find-and-replace operation needed."
    
    # User Prompt (The "Issue" or "Request")
    user_prompt = f"""I need to make a code change in file `{file_path}`.

Commit context: {context}

Current code:
```
{old_code}
```

Please provide the updated code for this change."""

    # Assistant Response (The "Tool Call" / Change)
    assistant_response = f"""I'll help you update that code. Here's the replacement:

```
{new_code}
```"""

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]
    }

def main():
    print("Loading dataset archit11/new2...")
    try:
        # Load a subset to verify execution
        dataset = load_dataset("archit11/new2", split="train") 
        print(f"Loaded {len(dataset)} samples.")
        
        # Verify columns
        print("Dataset columns:", dataset.column_names)
        
        # Transform
        print("Formatting data...")
        formatted_samples = []
        for sample in dataset:
            # Basic filtering for empty code
            if not sample.get('old_str') or not sample.get('new_str'):
                continue
            formatted_samples.append(format_agentic_sample(sample))
            
        print(f"Formatted {len(formatted_samples)} valid samples.")
        
        # Save as JSONL for inspection/training
        output_file = "data/sft_agentic_data.jsonl"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            for sample in formatted_samples:
                f.write(json.dumps(sample) + '\n')
                
        print(f"Saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing SFT data: {e}")

if __name__ == "__main__":
    main()
