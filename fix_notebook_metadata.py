
import json
import os

def fix_notebook(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    metadata = data.get('metadata', {})
    if 'widgets' in metadata:
        print(f"Removing 'metadata.widgets' from {file_path}")
        del metadata['widgets']
        data['metadata'] = metadata
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)  # Using indent=1 for minimal whitespace changes if possible
        print(f"Successfully updated {file_path}")
    else:
        print(f"No 'metadata.widgets' found in {file_path}")

if __name__ == "__main__":
    target_file = "/Users/architsinghai/assesment-jp/track_c_unsloth.ipynb"
    fix_notebook(target_file)
