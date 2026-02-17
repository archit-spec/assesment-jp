"""
Retrieval Data Generation Script (AST-based)
==============================================
Uses Python's ast module to generate natural language queries for code snippets.
Produces 200–400 text–code pairs for embedding fine-tuning (Track C).

No external API calls needed — queries are deterministically constructed
from AST analysis of function signatures, docstrings, and structure.
"""

import os
import json
import ast
import random
import textwrap
from typing import List, Dict, Optional
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────
FUNCTIONS_FILE = "data/extracted_functions.json"
OUTPUT_TRAIN = "data/retrieval_train.json"
OUTPUT_TEST = "data/retrieval_test.json"
OUTPUT_ALL = "data/retrieval_all.json"
DATA_CARD_FILE = "data/retrieval_data_card.json"

TARGET_PAIRS = 350       # aim for middle of 200–400
QUERIES_PER_FUNC = 3     # generate up to 3 diverse queries per function
TEST_RATIO = 0.20
RANDOM_SEED = 42


# ── Query Templates ────────────────────────────────────────────────

# Different query styles to create diversity
QUERY_STYLES = {
    "action": [
        "how to {action}",
        "{action} in httpx",
        "function that {action}",
        "code to {action}",
        "{action} implementation",
    ],
    "keyword": [
        "{keywords}",
        "{keywords} httpx",
        "{keywords} python",
    ],
    "question": [
        "how does httpx {action}?",
        "where is the {action} logic?",
        "what function handles {action}?",
        "how to {action} using httpx?",
    ],
    "natural": [
        "I need to {action}",
        "looking for the code that {action}",
        "find the function for {action}",
        "show me how to {action}",
    ],
}


# ── AST Helpers ─────────────────────────────────────────────────────

def get_docstring(node: ast.AST) -> Optional[str]:
    """Extract docstring from a function or class node."""
    if (node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
        return node.body[0].value.value
    return None


def name_to_action(name: str) -> str:
    """Convert function_name to a natural action phrase."""
    # Remove common prefixes
    for prefix in ["_", "get_", "set_", "is_", "has_", "do_", "make_"]:
        if name.startswith(prefix) and len(name) > len(prefix):
            name = name[len(prefix):]
            break

    # Convert snake_case to space-separated
    return name.replace("_", " ")


def extract_keywords(node: ast.FunctionDef, code: str) -> List[str]:
    """Extract meaningful keywords from the function."""
    keywords = set()

    # Function name parts
    for part in node.name.split("_"):
        if len(part) > 2 and part not in ("get", "set", "the", "and", "for", "def"):
            keywords.add(part)

    # Argument names
    for arg in node.args.args:
        if arg.arg not in ("self", "cls") and len(arg.arg) > 2:
            keywords.add(arg.arg.replace("_", " "))

    # Type annotation keywords
    for arg in node.args.args:
        if arg.annotation:
            ann = ast.unparse(arg.annotation)
            for part in ann.replace("[", " ").replace("]", " ").split():
                if len(part) > 2 and part[0].isupper():
                    keywords.add(part.lower())

    # Return type
    if node.returns:
        ret = ast.unparse(node.returns)
        for part in ret.replace("[", " ").replace("]", " ").split():
            if len(part) > 2:
                keywords.add(part.lower())

    return list(keywords)[:5]


def generate_queries_for_function(func: Dict) -> List[str]:
    """Generate diverse natural language queries for a function."""
    try:
        tree = ast.parse(textwrap.dedent(func["code"]))
        node = tree.body[0]
    except:
        return []

    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []

    queries = set()
    action = name_to_action(func["name"])
    keywords = extract_keywords(node, func["code"])
    docstring = get_docstring(node)

    # 1. Action-based queries from function name
    style = random.choice(list(QUERY_STYLES.keys()))
    templates = QUERY_STYLES[style]
    template = random.choice(templates)

    if "{action}" in template:
        queries.add(template.format(action=action))
    if "{keywords}" in template and keywords:
        queries.add(template.format(keywords=" ".join(keywords[:3])))

    # 2. Docstring-based query (first line of docstring)
    if docstring:
        first_line = docstring.strip().split("\n")[0].strip().rstrip(".")
        if len(first_line) > 10 and len(first_line) < 100:
            queries.add(first_line.lower())
            # Also make a question form
            queries.add(f"how to {first_line.lower()}")

    # 3. Keyword-based query
    if keywords:
        kw_query = " ".join(random.sample(keywords, min(3, len(keywords))))
        queries.add(kw_query)

    # 4. Natural language query based on args and return type
    args_info = []
    for arg in node.args.args:
        if arg.arg not in ("self", "cls"):
            if arg.annotation:
                args_info.append(f"{arg.arg}: {ast.unparse(arg.annotation)}")
            else:
                args_info.append(arg.arg)

    if args_info:
        arg_desc = " and ".join(args_info[:2])
        queries.add(f"function that takes {arg_desc}")

    ret_type = None
    if node.returns:
        ret_type = ast.unparse(node.returns)
        queries.add(f"function that returns {ret_type.lower()}")

    # 5. File-context query
    file_stem = func["file"].replace("/", " ").replace(".py", "").replace("_", " ")
    queries.add(f"{action} in {file_stem}")

    # Filter out very short/long queries
    queries = [q for q in queries if 5 < len(q) < 120]

    return list(queries)


# ── Main ────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)

    # Load extracted functions
    with open(FUNCTIONS_FILE) as f:
        all_funcs = json.load(f)

    # Filter to meaningful public functions
    funcs = [fn for fn in all_funcs
             if fn["type"] == "function"
             and 5 <= fn["num_lines"] <= 60
             and not fn["name"].startswith("__")
             and fn["name"] not in ("setUp", "tearDown", "setUpClass")]
    print(f"[INFO] {len(funcs)} functions available for retrieval data generation.")

    # Generate query-code pairs
    all_pairs = []
    success_count = 0
    fail_count = 0

    random.shuffle(funcs)

    for i, func in enumerate(funcs):
        if len(all_pairs) >= TARGET_PAIRS:
            break

        queries = generate_queries_for_function(func)
        if not queries:
            fail_count += 1
            continue

        # Take up to QUERIES_PER_FUNC queries
        for query in queries[:QUERIES_PER_FUNC]:
            all_pairs.append({
                "query": query,
                "code": func["code"],
                "function_name": func["name"],
                "file": func["file"],
                "num_lines": func["num_lines"],
            })

        success_count += 1

    print(f"[INFO] Generated {len(all_pairs)} query–code pairs from {success_count} functions ({fail_count} failures).")

    # Shuffle and split
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * (1 - TEST_RATIO))
    train_data = all_pairs[:split_idx]
    test_data = all_pairs[split_idx:]

    # Save
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_ALL, "w") as f:
        json.dump(all_pairs, f, indent=2)
    with open(OUTPUT_TRAIN, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(OUTPUT_TEST, "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"[INFO] Train: {len(train_data)}, Test: {len(test_data)}")
    print(f"[INFO] Saved to {OUTPUT_TRAIN}, {OUTPUT_TEST}")

    # Sample some pairs
    print("\n[SAMPLE] Example pairs:")
    for pair in all_pairs[:5]:
        print(f"  Query: \"{pair['query']}\"")
        print(f"  Code:  {pair['function_name']} ({pair['file']})")
        print()

    # Data Card
    data_card = {
        "name": "httpx Text–Code Retrieval Pairs (AST-generated)",
        "version": "1.0",
        "description": "Natural language query–code pairs programmatically generated from encode/httpx using AST analysis. No external API calls used.",
        "source_repo": "https://github.com/encode/httpx",
        "generation_method": "Python ast module — deterministic, reproducible",
        "total_pairs": len(all_pairs),
        "train_pairs": len(train_data),
        "test_pairs": len(test_data),
        "queries_per_function": QUERIES_PER_FUNC,
        "unique_functions": success_count,
        "query_styles": list(QUERY_STYLES.keys()),
        "query_construction": {
            "action": "Function name converted to action phrase (snake_case → natural language)",
            "docstring": "First line of docstring used as natural query",
            "keyword": "Keywords extracted from arg names, type annotations, return types",
            "signature": "Queries based on parameter types and return types",
            "context": "File path context added to queries",
        },
        "filtering_rules": [
            "Functions between 5–60 lines only",
            "Excluded dunder methods (__init__, __repr__, etc.)",
            "Excluded setUp/tearDown test methods",
            "Queries between 5–120 chars only",
        ],
        "random_seed": RANDOM_SEED,
        "reproducible": True,
    }
    with open(DATA_CARD_FILE, "w") as f:
        json.dump(data_card, f, indent=2)
    print(f"[INFO] Data card saved to {DATA_CARD_FILE}")


if __name__ == "__main__":
    main()
