"""
Data Preparation Script
=======================
Clones the verl repository, extracts Python source files,
and prepares the code corpus for all three tracks.
"""

import os
import re
import json
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import random
import ast

# ── Config ──────────────────────────────────────────────────────────
REPO_URL = "https://github.com/volcengine/verl.git"
REPO_DIR = "data/verl_repo"
CORPUS_DIR = "data/code_corpus"
METADATA_FILE = "data/corpus_metadata.json"
FUNCTIONS_FILE = "data/extracted_functions.json"

MIN_LINES = 30          # Avoid __init__ and trivial files
MAX_LINES = 2000        # skip huge generated files
MAX_FILES = 1000        # Increased coverage
RANDOM_SEED = 42


# ── Helpers ─────────────────────────────────────────────────────────

def clone_repo(url: str, dest: str) -> None:
    """Clone repo if not already present."""
    if os.path.exists(dest):
        print(f"[INFO] Repo already exists at {dest}, skipping clone.")
        return
    print(f"[INFO] Cloning {url} → {dest}")
    subprocess.run(["git", "clone", "--depth", "1", url, dest], check=True)


def collect_python_files(repo_dir: str, min_lines: int, max_lines: int, max_files: int) -> List[Dict]:
    """Walk the repo and collect metadata for all qualifying .py files."""
    files = []
    repo_path = Path(repo_dir)

    all_py_files = sorted(repo_path.rglob("*.py"))
    for py_file in all_py_files:
        rel_path = str(py_file.relative_to(repo_path))

        if any(skip in rel_path for skip in ["test", "tests", "docs/", "examples/", ".git", "__init__.py"]):
            continue

        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        lines = content.split("\n")
        num_lines = len(lines)

        if num_lines < min_lines or num_lines > max_lines:
            continue

        # Skip files that are mostly comments/imports
        if "TODO" in content[:100]:
            pass # Keep strict filtering logic if desired, but size/line count is main filter now.

        # Count functions/classes via AST
        num_functions = 0
        num_classes = 0
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    num_functions += 1
                elif isinstance(node, ast.ClassDef):
                    num_classes += 1
        except:
            pass

        files.append({
            "rel_path": rel_path,
            "abs_path": str(py_file),
            "num_lines": num_lines,
            "num_functions": num_functions,
            "num_classes": num_classes,
            "size_bytes": len(content.encode("utf-8")),
        })

    # If too many files, prioritize larger/more complex ones (descending size)
    # This ensures we get the "meat" of the codebase first
    if len(files) > max_files:
        files = sorted(files, key=lambda x: x["size_bytes"], reverse=True)[:max_files]
        # Then sort by path for determinism
        files = sorted(files, key=lambda x: x["rel_path"])
    
    return files


def copy_corpus_files(file_metas: List[Dict], repo_dir: str, corpus_dir: str) -> None:
    """Copy qualifying files into a flat corpus directory."""
    os.makedirs(corpus_dir, exist_ok=True)
    for meta in file_metas:
        src = Path(meta["abs_path"])
        # Preserve directory structure using underscores
        safe_name = meta["rel_path"].replace("/", "__")
        dst = Path(corpus_dir) / safe_name
        shutil.copy2(src, dst)
        meta["corpus_filename"] = safe_name


def extract_functions_and_classes(file_metas: List[Dict], repo_dir: str) -> List[Dict]:
    """
    Parse each Python file with AST and extract individual functions/classes.
    This is used downstream by SFT and retrieval data generators.
    """
    extracted = []
    for meta in file_metas:
        filepath = Path(meta["abs_path"])
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(content)
        except (SyntaxError, UnicodeDecodeError):
            continue

        lines = content.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = node.lineno - 1
                end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else start + 10
                func_code = "\n".join(lines[start:end])

                if len(func_code.strip().split("\n")) < 3:
                    continue

                has_docstring = False
                if node.body:
                    first = node.body[0]
                    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                        has_docstring = True

                extracted.append({
                    "type": "function",
                    "name": node.name,
                    "file": meta["rel_path"],
                    "start_line": start + 1,
                    "end_line": end,
                    "code": func_code,
                    "num_lines": end - start,
                    "has_docstring": has_docstring,
                })

            elif isinstance(node, ast.ClassDef):
                start = node.lineno - 1
                end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else start + 20
                class_code = "\n".join(lines[start:end])

                if len(class_code.strip().split("\n")) < 5:
                    continue

                has_docstring = False
                if node.body:
                    first = node.body[0]
                    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                        has_docstring = True

                extracted.append({
                    "type": "class",
                    "name": node.name,
                    "file": meta["rel_path"],
                    "start_line": start + 1,
                    "end_line": end,
                    "code": class_code,
                    "num_lines": end - start,
                    "has_docstring": has_docstring,
                })

    return extracted


# ── Main ────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)

def main():
    random.seed(RANDOM_SEED)
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1. Clone the repo
    clone_repo(REPO_URL, REPO_DIR)

    # 2. Collect Python files
    print("[INFO] Scanning for Python files...")
    file_metas = collect_python_files(REPO_DIR, MIN_LINES, MAX_LINES, MAX_FILES)
    print(f"[INFO] Found {len(file_metas)} qualifying Python files.")

    # 3. Clear old corpus and copy new files
    if os.path.exists(CORPUS_DIR):
        shutil.rmtree(CORPUS_DIR)
    copy_corpus_files(file_metas, REPO_DIR, CORPUS_DIR)

    # 4. Save corpus metadata
    metadata = {
        "repo": REPO_URL,
        "language": "python",
        "total_files": len(file_metas),
        "total_lines": sum(m["num_lines"] for m in file_metas),
        "total_functions": sum(m["num_functions"] for m in file_metas),
        "total_classes": sum(m["num_classes"] for m in file_metas),
        "files": file_metas,
    }
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Corpus metadata saved to {METADATA_FILE}")

    # 5. Extract functions and classes for downstream use
    print("[INFO] Extracting functions and classes...")
    extracted = extract_functions_and_classes(file_metas, REPO_DIR)
    print(f"[INFO] Extracted {len(extracted)} functions/classes.")

    with open(FUNCTIONS_FILE, "w") as f:
        json.dump(extracted, f, indent=2)
    print(f"[INFO] Extracted items saved to {FUNCTIONS_FILE}")

    # 6. Print summary
    print("\n" + "=" * 60)
    print("CORPUS SUMMARY")
    print("=" * 60)
    print(f"  Repository:       {REPO_URL}")
    print(f"  Python files:     {len(file_metas)}")
    print(f"  Total lines:      {metadata['total_lines']:,}")
    print(f"  Total functions:  {metadata['total_functions']}")
    print(f"  Total classes:    {metadata['total_classes']}")
    print(f"  Extracted items:  {len(extracted)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
