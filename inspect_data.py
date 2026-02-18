
from datasets import load_dataset

def inspect_dataset():
    print("Loading archit11/new2...")
    try:
        ds = load_dataset("archit11/new2", split="train", streaming=True)
        print("Dataset keys:", next(iter(ds)).keys())
        print("Sample entry:")
        print(next(iter(ds)))
    except Exception as e:
        print(f"Error loading archit11/new2: {e}")

    print("\nLoading archit11/hyperswitch-filenames...")
    try:
        ds = load_dataset("archit11/hyperswitch-filenames", split="train", streaming=True)
        print("Dataset keys:", next(iter(ds)).keys())
        print("Sample entry:")
        print(next(iter(ds)))
    except Exception as e:
        print(f"Error loading archit11/hyperswitch-filenames: {e}")

if __name__ == "__main__":
    inspect_dataset()
