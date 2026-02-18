"""
SFT Data Generation Script (AST-based)
========================================
Uses Python's ast module to programmatically generate instruction–response pairs
from the code corpus. Produces 300–600 pairs across 6 instruction categories.

No external API calls needed — all pairs are constructed deterministically
from AST analysis of the verl source code.
"""

import os
import json
import ast
import random
import textwrap
import inspect
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────
FUNCTIONS_FILE = "data/extracted_functions.json"
CORPUS_DIR = "data/code_corpus"
OUTPUT_TRAIN = "data/sft_train.json"
OUTPUT_TEST = "data/sft_test.json"
OUTPUT_ALL = "data/sft_all.json"
DATA_CARD_FILE = "data/sft_data_card.json"

TARGET_PAIRS = 500       # aim for middle of 300–600
TEST_RATIO = 0.15
RANDOM_SEED = 42


# ── AST Helpers ─────────────────────────────────────────────────────

def get_docstring(node: ast.AST) -> Optional[str]:
    """Extract docstring from a function or class node."""
    if (node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
        return node.body[0].value.value
    return None


def get_function_signature(node: ast.FunctionDef) -> str:
    """Extract function signature string from AST node."""
    args = []
    all_args = node.args

    # positional args
    defaults_offset = len(all_args.args) - len(all_args.defaults)
    for i, arg in enumerate(all_args.args):
        a = arg.arg
        if arg.annotation:
            a += f": {ast.unparse(arg.annotation)}"
        if i >= defaults_offset:
            default = all_args.defaults[i - defaults_offset]
            a += f" = {ast.unparse(default)}"
        args.append(a)

    # *args
    if all_args.vararg:
        a = f"*{all_args.vararg.arg}"
        if all_args.vararg.annotation:
            a += f": {ast.unparse(all_args.vararg.annotation)}"
        args.append(a)

    # keyword-only
    kw_defaults_map = {i: d for i, d in enumerate(all_args.kw_defaults) if d is not None}
    for i, arg in enumerate(all_args.kwonlyargs):
        a = arg.arg
        if arg.annotation:
            a += f": {ast.unparse(arg.annotation)}"
        if i in kw_defaults_map:
            a += f" = {ast.unparse(kw_defaults_map[i])}"
        args.append(a)

    # **kwargs
    if all_args.kwarg:
        a = f"**{all_args.kwarg.arg}"
        if all_args.kwarg.annotation:
            a += f": {ast.unparse(all_args.kwarg.annotation)}"
        args.append(a)

    ret = ""
    if node.returns:
        ret = f" -> {ast.unparse(node.returns)}"

    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({', '.join(args)}){ret}"


def get_return_type(node: ast.FunctionDef) -> Optional[str]:
    """Extract return type annotation."""
    if node.returns:
        return ast.unparse(node.returns)
    return None


def get_arg_info(node: ast.FunctionDef) -> List[Dict]:
    """Extract argument names, types, and defaults."""
    args_info = []
    all_args = node.args
    defaults_offset = len(all_args.args) - len(all_args.defaults)

    for i, arg in enumerate(all_args.args):
        if arg.arg == "self" or arg.arg == "cls":
            continue
        info = {"name": arg.arg}
        if arg.annotation:
            info["type"] = ast.unparse(arg.annotation)
        if i >= defaults_offset:
            info["default"] = ast.unparse(all_args.defaults[i - defaults_offset])
        args_info.append(info)

    for i, arg in enumerate(all_args.kwonlyargs):
        info = {"name": arg.arg, "keyword_only": True}
        if arg.annotation:
            info["type"] = ast.unparse(arg.annotation)
        args_info.append(info)

    return args_info


def count_branches(node: ast.AST) -> int:
    """Count if/elif/else branches in a function."""
    return sum(1 for n in ast.walk(node) if isinstance(n, (ast.If, ast.IfExp)))


def get_called_functions(node: ast.AST) -> List[str]:
    """Extract names of functions called within a node."""
    calls = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                calls.append(n.func.id)
            elif isinstance(n.func, ast.Attribute):
                calls.append(n.func.attr)
    return calls


def get_exceptions_raised(node: ast.AST) -> List[str]:
    """Extract exceptions raised in a function."""
    exceptions = []
    for n in ast.walk(node):
        if isinstance(n, ast.Raise) and n.exc:
            if isinstance(n.exc, ast.Call) and isinstance(n.exc.func, ast.Name):
                exceptions.append(n.exc.func.id)
            elif isinstance(n.exc, ast.Name):
                exceptions.append(n.exc.id)
    return exceptions


def strip_docstring(code: str) -> str:
    """Remove the docstring from function code."""
    try:
        tree = ast.parse(textwrap.dedent(code))
        func = tree.body[0]
        if get_docstring(func):
            func.body = func.body[1:]  # remove docstring node
        return ast.unparse(tree)
    except:
        return code


# ── Pair Generators ─────────────────────────────────────────────────

def generate_explain_pair(func_code: str, func_name: str, file: str) -> Optional[Dict]:
    """Generate an 'explain this function' pair using AST analysis."""
    try:
        tree = ast.parse(textwrap.dedent(func_code))
        node = tree.body[0]
    except:
        return None

    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None

    sig = get_function_signature(node)
    args = get_arg_info(node)
    ret_type = get_return_type(node)
    docstring = get_docstring(node)
    branches = count_branches(node)
    calls = get_called_functions(node)
    exceptions = get_exceptions_raised(node)

    # Build structured explanation
    explanation_parts = []

    # Purpose
    if docstring:
        first_line = docstring.strip().split("\n")[0]
        explanation_parts.append(f"**Purpose**: {first_line}")
    else:
        explanation_parts.append(f"**Purpose**: The `{func_name}` function from `{file}`.")

    # Signature
    explanation_parts.append(f"\n**Signature**: `{sig}`")

    # Parameters
    if args:
        params = []
        for a in args:
            p = f"- `{a['name']}`"
            if "type" in a:
                p += f" ({a['type']})"
            if "default" in a:
                p += f", defaults to `{a['default']}`"
            params.append(p)
        explanation_parts.append("\n**Parameters**:\n" + "\n".join(params))

    # Return type
    if ret_type:
        explanation_parts.append(f"\n**Returns**: `{ret_type}`")

    # Complexity hints
    details = []
    if branches > 0:
        details.append(f"Contains {branches} conditional branch(es)")
    if exceptions:
        details.append(f"Raises: {', '.join(set(exceptions))}")
    if calls:
        notable = [c for c in set(calls) if not c.startswith("_")][:5]
        if notable:
            details.append(f"Calls: {', '.join(notable)}")
    if isinstance(node, ast.AsyncFunctionDef):
        details.append("This is an async function")

    if details:
        explanation_parts.append("\n**Implementation Details**:\n" + "\n".join(f"- {d}" for d in details))

    # Full docstring if available
    if docstring and len(docstring.strip().split("\n")) > 1:
        explanation_parts.append(f"\n**Documentation**:\n{docstring.strip()}")

    instruction = f"Explain what the following Python function does:\n\n```python\n{func_code}\n```"
    response = "\n".join(explanation_parts)

    return {
        "instruction": instruction,
        "response": response,
        "category": "explain",
        "source_function": func_name,
        "source_file": file,
    }


def generate_docstring_pair(func_code: str, func_name: str, file: str) -> Optional[Dict]:
    """Generate a 'write a docstring' pair — strip the existing docstring, use it as the target."""
    try:
        tree = ast.parse(textwrap.dedent(func_code))
        node = tree.body[0]
    except:
        return None

    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None

    docstring = get_docstring(node)
    if not docstring or len(docstring.strip()) < 20:
        # No existing docstring — generate one from AST info
        args = get_arg_info(node)
        ret_type = get_return_type(node)
        exceptions = get_exceptions_raised(node)

        doc_lines = [f'"""', f"{func_name.replace('_', ' ').capitalize()}."]
        if args:
            doc_lines.append("")
            doc_lines.append("Args:")
            for a in args:
                t = a.get("type", "Any")
                doc_lines.append(f"    {a['name']} ({t}): Description.")
        if ret_type:
            doc_lines.append("")
            doc_lines.append("Returns:")
            doc_lines.append(f"    {ret_type}: Description.")
        if exceptions:
            doc_lines.append("")
            doc_lines.append("Raises:")
            for e in set(exceptions):
                doc_lines.append(f"    {e}: Description.")
        doc_lines.append('"""')
        docstring_text = "\n".join(doc_lines)
    else:
        docstring_text = f'"""{docstring}"""'

    # Strip docstring from the code shown to the model
    code_without_doc = strip_docstring(func_code)

    instruction = f"Write a comprehensive Google-style docstring for the following Python function:\n\n```python\n{code_without_doc}\n```"
    response = docstring_text

    return {
        "instruction": instruction,
        "response": response,
        "category": "docstring",
        "source_function": func_name,
        "source_file": file,
    }


def generate_complete_pair(func_code: str, func_name: str, file: str) -> Optional[Dict]:
    """Generate a 'complete the function' pair — show signature, use full body as target."""
    try:
        tree = ast.parse(textwrap.dedent(func_code))
        node = tree.body[0]
    except:
        return None

    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None

    lines = func_code.split("\n")
    if len(lines) < 4:
        return None

    # Keep def line + docstring (if any), truncate body
    sig = get_function_signature(node)
    docstring = get_docstring(node)

    partial_lines = [lines[0]]  # def line
    if docstring:
        # Find docstring end line
        in_doc = False
        for i, line in enumerate(lines[1:], 1):
            partial_lines.append(line)
            if '"""' in line or "'''" in line:
                if in_doc:
                    break
                else:
                    in_doc = True
                    if line.count('"""') >= 2 or line.count("'''") >= 2:
                        break
    # Add first 1-2 body lines as hint
    remaining_start = len(partial_lines)
    hint_lines = [l for l in lines[remaining_start:remaining_start + 2] if l.strip()]
    partial_lines.extend(hint_lines)
    partial_lines.append("    # ... complete this implementation")

    partial_code = "\n".join(partial_lines)

    instruction = f"Complete the following Python function from the verl library:\n\n```python\n{partial_code}\n```"
    response = f"```python\n{func_code}\n```"

    return {
        "instruction": instruction,
        "response": response,
        "category": "complete",
        "source_function": func_name,
        "source_file": file,
    }


def generate_bugfix_pair(func_code: str, func_name: str, file: str) -> Optional[Dict]:
    """Generate a 'fix the bug' pair by injecting a synthetic bug."""
    BUG_MUTATIONS = [
        # (description, find, replace, condition)
        ("Off-by-one error: removed '- 1' adjustment", " - 1", "", lambda c: " - 1" in c),
        ("Off-by-one error: removed '+ 1' adjustment", " + 1", "", lambda c: " + 1" in c),
        ("Wrong comparison: '==' changed to '!='", " == ", " != ", lambda c: " == " in c),
        ("Logic error: 'is not' changed to 'is'", " is not ", " is ", lambda c: " is not " in c),
        ("Logic error: 'not in' changed to 'in'", " not in ", " in ", lambda c: " not in " in c),
        ("Missing return: return statement commented out", "return ", "# return ", lambda c: c.count("return ") == 1),
        ("Wrong reference: 'self.' changed to 'cls.'", "self.", "cls.", lambda c: "self." in c and "cls." not in c),
        ("Boolean flip: True changed to False", "True", "False", lambda c: "True" in c and c.count("True") == 1),
        ("Boolean flip: False changed to True", "False", "True", lambda c: "False" in c and c.count("False") == 1),
        ("Missing None check: 'is None' changed to 'is not None'", " is None", " is not None", lambda c: " is None" in c),
    ]

    random.shuffle(BUG_MUTATIONS)

    for desc, find, replace, condition in BUG_MUTATIONS:
        if condition(func_code):
            buggy_code = func_code.replace(find, replace, 1)
            if buggy_code != func_code:
                instruction = f"The following Python code contains a bug. Identify the bug and provide the corrected version:\n\n```python\n{buggy_code}\n```"
                response = (
                    f"**Bug**: {desc}\n\n"
                    f"**Fix**: Restore the original logic.\n\n"
                    f"**Corrected Code**:\n```python\n{func_code}\n```"
                )
                return {
                    "instruction": instruction,
                    "response": response,
                    "category": "bugfix",
                    "source_function": func_name,
                    "source_file": file,
                }

    return None


def generate_unittest_pair(func_code: str, func_name: str, file: str) -> Optional[Dict]:
    """Generate a 'write unit tests' pair using AST-derived info."""
    try:
        tree = ast.parse(textwrap.dedent(func_code))
        node = tree.body[0]
    except:
        return None

    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None

    args = get_arg_info(node)
    ret_type = get_return_type(node)
    exceptions = get_exceptions_raised(node)
    is_method = any(a.arg in ("self", "cls") for a in node.args.args)

    # Build test stubs
    test_lines = ["import pytest", ""]

    # Determine import
    module = file.replace("/", ".").replace(".py", "")
    test_lines.append(f"# from {module} import {func_name}")
    test_lines.append("")

    # Happy path test
    test_lines.append(f"def test_{func_name}_basic():")
    test_lines.append(f'    """Test basic functionality of {func_name}."""')
    if args and not is_method:
        arg_str = ", ".join(f"{a['name']}={a.get('default', '...')}" for a in args[:3])
        test_lines.append(f"    result = {func_name}({arg_str})")
        test_lines.append(f"    assert result is not None")
    else:
        test_lines.append(f"    # TODO: Test basic {func_name} behavior")
        test_lines.append(f"    pass")
    test_lines.append("")

    # Edge case test
    test_lines.append(f"def test_{func_name}_edge_cases():")
    test_lines.append(f'    """Test edge cases for {func_name}."""')
    if args and not is_method:
        test_lines.append(f"    # Test with None/empty inputs")
        for a in args[:2]:
            t = a.get("type", "")
            if "str" in t.lower():
                test_lines.append(f'    # result = {func_name}({a["name"]}="")')
            elif "int" in t.lower() or "float" in t.lower():
                test_lines.append(f'    # result = {func_name}({a["name"]}=0)')
            elif "list" in t.lower():
                test_lines.append(f'    # result = {func_name}({a["name"]}=[])')
    test_lines.append(f"    pass")
    test_lines.append("")

    # Exception test
    if exceptions:
        test_lines.append(f"def test_{func_name}_raises():")
        test_lines.append(f'    """Test that {func_name} raises appropriate exceptions."""')
        for exc in set(exceptions):
            test_lines.append(f"    with pytest.raises({exc}):")
            test_lines.append(f"        # TODO: trigger {exc}")
            test_lines.append(f"        pass")
        test_lines.append("")

    test_code = "\n".join(test_lines)

    instruction = f"Generate unit tests for the following Python function using pytest:\n\n```python\n{func_code}\n```"
    response = f"```python\n{test_code}\n```"

    return {
        "instruction": instruction,
        "response": response,
        "category": "unit_test",
        "source_function": func_name,
        "source_file": file,
    }


def generate_improve_pair(func_code: str, func_name: str, file: str) -> Optional[Dict]:
    """Generate an 'improve this code' pair using AST-based analysis."""
    try:
        tree = ast.parse(textwrap.dedent(func_code))
        node = tree.body[0]
    except:
        return None

    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None

    suggestions = []
    improvements = []

    # Check: missing type hints
    args_without_types = [a.arg for a in node.args.args
                          if a.arg not in ("self", "cls") and a.annotation is None]
    if args_without_types:
        suggestions.append(
            f"**Add type hints**: Parameters `{'`, `'.join(args_without_types)}` lack type annotations. "
            f"Adding type hints improves readability and enables static analysis."
        )

    # Check: missing return type
    if not node.returns:
        suggestions.append(
            "**Add return type annotation**: The function is missing a return type hint. "
            "Adding `-> ReturnType` clarifies the function's contract."
        )

    # Check: missing docstring
    if not get_docstring(node):
        suggestions.append(
            "**Add docstring**: This function lacks documentation. "
            "A Google-style docstring would improve maintainability."
        )

    # Check: too many branches
    branches = count_branches(node)
    if branches > 3:
        suggestions.append(
            f"**Reduce complexity**: Function has {branches} conditional branches. "
            f"Consider extracting helper functions or using early returns to flatten the logic."
        )

    # Check: bare except
    for n in ast.walk(node):
        if isinstance(n, ast.ExceptHandler) and n.type is None:
            suggestions.append(
                "**Avoid bare except**: Use specific exception types instead of bare `except:` "
                "to prevent catching unexpected errors like KeyboardInterrupt."
            )
            break

    # Check: magic numbers
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            if n.value not in (0, 1, -1, 2, True, False, None) and abs(n.value) > 2:
                suggestions.append(
                    f"**Extract magic number**: The literal `{n.value}` should be extracted into "
                    f"a named constant for clarity."
                )
                break

    # Check: long function
    lines = func_code.split("\n")
    if len(lines) > 30:
        suggestions.append(
            f"**Consider splitting**: At {len(lines)} lines, this function is quite long. "
            f"Consider breaking it into smaller, focused helper functions."
        )

    if not suggestions:
        return None

    instruction = f"Review the following Python code and suggest improvements:\n\n```python\n{func_code}\n```"
    response = f"Here are suggestions to improve `{func_name}`:\n\n" + "\n\n".join(f"{i+1}. {s}" for i, s in enumerate(suggestions))

    return {
        "instruction": instruction,
        "response": response,
        "category": "improve",
        "source_function": func_name,
        "source_file": file,
    }


# ── Main ────────────────────────────────────────────────────────────

GENERATORS = {
    "explain": generate_explain_pair,
    "docstring": generate_docstring_pair,
    "complete": generate_complete_pair,
    "bugfix": generate_bugfix_pair,
    "unit_test": generate_unittest_pair,
    "improve": generate_improve_pair,
}


def main():
    random.seed(RANDOM_SEED)

    # Load extracted functions
    with open(FUNCTIONS_FILE) as f:
        all_funcs = json.load(f)

    # Filter to functions with reasonable size
    funcs = [fn for fn in all_funcs
             if fn["type"] == "function" and 5 <= fn["num_lines"] <= 80]
    print(f"[INFO] {len(funcs)} functions available for SFT data generation.")

    # Calculate pairs per category
    categories = list(GENERATORS.keys())
    pairs_per_category = TARGET_PAIRS // len(categories)
    print(f"[INFO] Target: {TARGET_PAIRS} pairs, ~{pairs_per_category} per category.")

    generated_pairs = []
    stats = {cat: {"attempted": 0, "success": 0} for cat in categories}

    for cat in categories:
        print(f"\n[CATEGORY] Generating '{cat}' pairs...")
        sampled = random.sample(funcs, min(pairs_per_category + 20, len(funcs)))

        for func in sampled:
            if stats[cat]["success"] >= pairs_per_category:
                break

            stats[cat]["attempted"] += 1
            generator = GENERATORS[cat]
            pair = generator(func["code"], func["name"], func["file"])

            if pair:
                generated_pairs.append(pair)
                stats[cat]["success"] += 1

    print(f"\n[INFO] Total pairs generated: {len(generated_pairs)}")

    # Shuffle and split
    random.shuffle(generated_pairs)
    split_idx = int(len(generated_pairs) * (1 - TEST_RATIO))
    train_data = generated_pairs[:split_idx]
    test_data = generated_pairs[split_idx:]

    # Save
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_ALL, "w") as f:
        json.dump(generated_pairs, f, indent=2)
    with open(OUTPUT_TRAIN, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(OUTPUT_TEST, "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"[INFO] Train: {len(train_data)}, Test: {len(test_data)}")
    print(f"[INFO] Saved to {OUTPUT_TRAIN}, {OUTPUT_TEST}")

    # Print stats
    print(f"\n{'Category':<15} {'Attempted':>10} {'Success':>10} {'Rate':>8}")
    print("-" * 45)
    for cat in categories:
        s = stats[cat]
        rate = s["success"] / s["attempted"] if s["attempted"] > 0 else 0
        print(f"{cat:<15} {s['attempted']:>10} {s['success']:>10} {rate:>7.1%}")

    # Data Card
    data_card = {
        "name": "verl SFT Instruction–Response Pairs (AST-generated)",
        "version": "1.0",
        "description": "Instruction–response pairs programmatically generated from volcengine/verl Python library using AST analysis. No external API calls used.",
        "source_repo": "https://github.com/volcengine/verl",
        "generation_method": "Python ast module — deterministic, reproducible",
        "total_pairs": len(generated_pairs),
        "train_pairs": len(train_data),
        "test_pairs": len(test_data),
        "categories": {cat: stats[cat] for cat in categories},
        "category_descriptions": {
            "explain": "AST-derived structured explanation of function purpose, signature, arguments, and implementation details",
            "docstring": "Existing docstrings stripped from source; used as target. Fallback: AST-generated Google-style docstring stubs",
            "complete": "Function signature + docstring shown as prompt; full implementation as target",
            "bugfix": "Deterministic bug injection (off-by-one, logic flip, missing return, etc.); original code as target",
            "unit_test": "AST-derived pytest test stubs covering happy path, edge cases, and exception handling",
            "improve": "AST-based code review: missing type hints, docstrings, complexity, magic numbers, function length",
        },
        "filtering_rules": [
            "Functions between 5–80 lines only",
            "Only ast.FunctionDef and ast.AsyncFunctionDef nodes",
            "Skipped functions that couldn't be parsed",
            "Bugfix: skipped functions where no mutation could be applied",
            "Improve: skipped functions with no suggestions",
        ],
        "random_seed": RANDOM_SEED,
        "reproducible": True,
    }
    with open(DATA_CARD_FILE, "w") as f:
        json.dump(data_card, f, indent=2)
    print(f"[INFO] Data card saved to {DATA_CARD_FILE}")


if __name__ == "__main__":
    main()
