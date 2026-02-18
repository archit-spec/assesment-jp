import json
from datasets import load_dataset
import os
import whatthepatch
import re

def parse_patch_to_edits(patch_text):
    """
    Parses a git patch and converts it to a list of (file, old, new) edits.
    """
    edits = []
    try:
        for diff in whatthepatch.parse_patch(patch_text):
            if not diff.header or not diff.changes:
                continue
            
            file_path = diff.header.new_path if diff.header.new_path else diff.header.old_path
            if file_path.startswith('a/') or file_path.startswith('b/'):
                file_path = file_path[2:]

            old_lines = []
            new_lines = []
            for change in diff.changes:
                if change.old is not None:
                    old_lines.append(change.line if change.line else "")
                if change.new is not None:
                    new_lines.append(change.line if change.line else "")
            
            # Simple heuristic to get a small block for demonstration
            edits.append({
                "file": file_path,
                "old_str": "\n".join(old_lines[:30]),
                "new_str": "\n".join(new_lines[:30])
            })
    except Exception as e:
        print(f"Error parsing patch: {e}")
    return edits

def format_as_agentic(sample, dataset_type="new2"):
    """
    Formats sample into Agentic Tool Use format.
    """
    if dataset_type == "new2":
        return {
            "messages": [
                {"role": "system", "content": "You are an expert software engineer."},
                {"role": "user", "content": sample['prompt']},
                {"role": "assistant", "content": sample['response']}
            ]
        }
    elif dataset_type == "hyperswitch-filenames":
        issue = sample['problem_statement']
        edits = parse_patch_to_edits(sample['patch'])
        
        tool_calls = ""
        for edit in edits:
            tool_calls += f"\n<tool_code>\nedit_file(\n    path=\"{edit['file']}\",\n    original=\"\"\"{edit['old_str']}\"\"\",\n    replacement=\"\"\"{edit['new_str']}\"\"\"\n)\n</tool_code>"

        response = f"<think>\nI will fix the reported issue by updating the code.\n</think>\n{tool_calls}"
        
        return {
            "messages": [
                {"role": "system", "content": "You are an expert coding agent."},
                {"role": "user", "content": issue},
                {"role": "assistant", "content": response}
            ]
        }
    return None

def main():
    output_file = "data/sft_agentic_final.jsonl"
    os.makedirs("data", exist_ok=True)
    
    all_formatted = []

    print("Processing archit11/new2 (subset)...")
    try:
        ds_new2 = load_dataset("archit11/new2", split="train")
        # limit to 500 for training efficiency
        for i, sample in enumerate(ds_new2):
            if i >= 500: break
            all_formatted.append(format_as_agentic(sample, "new2"))
    except Exception as e:
        print(f"Error loading new2: {e}")

    print("Processing archit11/hyperswitch-filenames (subset)...")
    try:
        ds_hw = load_dataset("archit11/hyperswitch-filenames", split="train")
        for i, sample in enumerate(ds_hw):
            if i >= 300: break
            fmt = format_as_agentic(sample, "hyperswitch-filenames")
            if fmt:
                all_formatted.append(fmt)
    except Exception as e:
        print(f"Error loading hf: {e}")

    print(f"Writing {len(all_formatted)} samples to {output_file}...")
    with open(output_file, 'w') as f:
        for entry in all_formatted:
            f.write(json.dumps(entry) + "\n")
    print("SFT data preparation complete.")

if __name__ == "__main__":
    main()
