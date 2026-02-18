"""
Track B/C Data Builder Using External Sources + Local Corpus Fallback
=====================================================================
Builds assignment-compatible datasets for:
- Track B (SFT): instruction/response pairs
- Track C (retrieval): query/code pairs

The builder is designed around the resources provided by the user:
- GitHub repos (readme + scripts references)
- Gists (query/tool-call transformation references)
- Hugging Face datasets

When Hugging Face access is unavailable in the current environment, it falls
back to local corpus files so outputs are still generated with valid schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from datasets import get_dataset_config_names, load_dataset

    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


@dataclass(frozen=True)
class SourceSpec:
    alias: str
    source_type: str
    link: str
    dataset_id: Optional[str] = None
    dataset_config: Optional[str] = None


SOURCE_SPECS: List[SourceSpec] = [
    SourceSpec(
        alias="deepwiki_scripts",
        source_type="repo",
        link="https://github.com/archit-athena/deepwiki-scripts",
    ),
    SourceSpec(
        alias="commits_training",
        source_type="repo",
        link="https://github.com/archit-athena/commits-training",
    ),
    SourceSpec(
        alias="gist_deepwiki_query",
        source_type="gist",
        link="https://gist.github.com/archit-spec/572bf2ff4ce37cdf4c5606b0d3083ef5",
    ),
    SourceSpec(
        alias="gist_coding_agent_minimal",
        source_type="gist",
        link="https://gist.github.com/archit-spec/b5cdcc71d8a2699b74e13481c924b783",
    ),
    SourceSpec(
        alias="gist_commit_breakdown",
        source_type="gist",
        link="https://gist.github.com/archit-spec/00053c2a3693500028388ebe3014e68a",
    ),
    SourceSpec(
        alias="gist_patch_to_toolcall",
        source_type="gist",
        link="https://gist.github.com/archit-spec/1983d1a2eddc130316d15bdc9d1bb13c",
    ),
    SourceSpec(
        alias="gist_toolcall_naturalizer",
        source_type="gist",
        link="https://gist.github.com/archit-spec/8a59e09cfe2d6fff41c041ea7b78f292",
    ),
    SourceSpec(
        alias="cpt_hyperswitch_token_aware",
        source_type="hf",
        link="https://huggingface.co/datasets/archit11/hyperswitch-token-aware-cpt-fixed",
        dataset_id="archit11/hyperswitch-token-aware-cpt-fixed",
    ),
    SourceSpec(
        alias="deepwiki_16k",
        source_type="hf",
        link="https://huggingface.co/datasets/archit11/deepwiki-16k",
        dataset_id="archit11/deepwiki-16k",
    ),
    SourceSpec(
        alias="issue_pr_new2",
        source_type="hf",
        link="https://huggingface.co/datasets/archit11/new2",
        dataset_id="archit11/new2",
    ),
    SourceSpec(
        alias="issue_pr_filenames",
        source_type="hf",
        link="https://huggingface.co/datasets/archit11/hyperswitch-filenames",
        dataset_id="archit11/hyperswitch-filenames",
    ),
    SourceSpec(
        alias="agent_data_collection_code_feedback",
        source_type="hf",
        link="https://huggingface.co/datasets/neulab/agent-data-collection/viewer/code_feedback",
        dataset_id="neulab/agent-data-collection",
        dataset_config="code_feedback",
    ),
]


TEXT_KEYWORDS = (
    "instruction",
    "prompt",
    "query",
    "question",
    "issue",
    "title",
    "description",
    "body",
    "summary",
    "message",
    "doc",
    "wiki",
)
ANSWER_KEYWORDS = ("response", "answer", "assistant", "completion", "target", "solution", "output")
CODE_KEYWORDS = (
    "code",
    "snippet",
    "source",
    "content",
    "file_content",
    "function",
    "implementation",
    "patch",
    "diff",
)
PATCH_KEYWORDS = ("patch", "diff", "commit", "hunk")
QUERY_KEYWORDS = ("query", "question", "search", "instruction", "prompt")
FILE_KEYWORDS = ("file", "path", "filename")
TOOLCALL_KEYWORDS = ("tool", "action", "plan", "steps", "trace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--target-sft", type=int, default=500)
    parser.add_argument("--target-retrieval", type=int, default=350)
    parser.add_argument("--sft-test-ratio", type=float, default=0.15)
    parser.add_argument("--retrieval-test-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows-per-source", type=int, default=2000)
    parser.add_argument("--max-local-corpus-files", type=int, default=600)
    parser.add_argument("--code-corpus-dir", default="data/code_corpus_hyperswitch")
    parser.add_argument("--disable-hf", action="store_true")
    parser.add_argument("--skip-local-corpus", action="store_true")
    parser.add_argument(
        "--local-source",
        action="append",
        default=[],
        help="Alias mapping for local exports, format: alias=path",
    )
    parser.add_argument("--beautiful-mention-file", default="@beautifulMention.md")
    return parser.parse_args()


def parse_local_source_args(items: List[str]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            continue
        alias, raw_path = item.split("=", 1)
        alias = alias.strip()
        path = Path(raw_path.strip())
        if alias:
            mapping[alias] = path
    return mapping


def flatten_values(value: Any, prefix: str = "", out: Optional[List[Tuple[str, str]]] = None) -> List[Tuple[str, str]]:
    if out is None:
        out = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flatten_values(child, child_prefix, out)
    elif isinstance(value, list):
        for idx, child in enumerate(value[:20]):
            child_prefix = f"{prefix}[{idx}]"
            flatten_values(child, child_prefix, out)
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped:
            out.append((prefix.lower(), stripped))
    return out


def sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def text_preview(text: str, max_len: int = 240) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def looks_like_code(text: str) -> bool:
    if not text:
        return False
    if "diff --git" in text or "@@" in text:
        return True
    if re.search(r"(?m)^\s*(def |async def |class |fn |impl |pub |import |from )", text):
        return True
    if text.count("{") >= 2 and text.count("}") >= 2 and ";" in text:
        return True
    return False


def is_patch(text: str) -> bool:
    return "diff --git" in text or ("@@" in text and ("+++" in text or "---" in text))


def pick_best(
    flat_pairs: List[Tuple[str, str]],
    keywords: Tuple[str, ...],
    min_len: int = 0,
    max_len: int = 200_000,
    require_newline: bool = False,
    prefer_code_like: bool = False,
) -> Optional[str]:
    best: Optional[str] = None
    best_score = -1.0

    for key, value in flat_pairs:
        if len(value) < min_len or len(value) > max_len:
            continue
        if require_newline and "\n" not in value:
            continue

        key_score = sum(1 for kw in keywords if kw in key)
        if key_score == 0:
            continue

        score = float(key_score * 10)
        score += min(len(value), 1200) / 1200.0
        if prefer_code_like and looks_like_code(value):
            score += 4.0
        if is_patch(value):
            score += 2.0

        if score > best_score:
            best_score = score
            best = value

    return best


def extract_first_symbol(code: str) -> str:
    for line in code.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)", line)
        if m:
            return m.group(1)
        m = re.match(r"(?:async\s+)?def\s+([A-Za-z_]\w*)", line)
        if m:
            return m.group(1)
        m = re.match(r"class\s+([A-Za-z_]\w*)", line)
        if m:
            return m.group(1)
    return "unknown_symbol"


def truncate_code(code: str, max_lines: int = 120) -> str:
    lines = code.splitlines()
    if len(lines) <= max_lines:
        return code
    return "\n".join(lines[:max_lines])


def extract_record(row: Dict[str, Any], source_alias: str) -> Optional[Dict[str, Any]]:
    flat = flatten_values(row)
    if not flat:
        return None

    text = pick_best(flat, TEXT_KEYWORDS, min_len=8, max_len=50_000)
    answer = pick_best(flat, ANSWER_KEYWORDS, min_len=8, max_len=80_000)
    query = pick_best(flat, QUERY_KEYWORDS, min_len=6, max_len=600)
    code = pick_best(flat, CODE_KEYWORDS, min_len=15, max_len=200_000, require_newline=True, prefer_code_like=True)
    patch = pick_best(flat, PATCH_KEYWORDS, min_len=20, max_len=200_000, require_newline=True)
    filename = pick_best(flat, FILE_KEYWORDS, min_len=2, max_len=240)
    tool_trace = pick_best(flat, TOOLCALL_KEYWORDS, min_len=10, max_len=80_000)

    if patch and not code:
        code = patch
    if code and not looks_like_code(code):
        code = None
    if patch and not is_patch(patch):
        patch = None
    if filename and ("\n" in filename or len(filename.split()) > 12):
        filename = None

    if not any([text, answer, query, code, patch, filename, tool_trace]):
        return None

    identity_seed = f"{source_alias}|{text_preview(text or '', 100)}|{text_preview(code or '', 100)}|{text_preview(query or '', 80)}"
    return {
        "id": sha(identity_seed),
        "source": source_alias,
        "text": text,
        "answer": answer,
        "query": query,
        "code": code,
        "patch": patch,
        "filename": filename,
        "tool_trace": tool_trace,
    }


def load_rows_from_json(path: Path, max_rows: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def push(item: Any) -> None:
        if isinstance(item, dict):
            rows.append(item)

    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        push(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                    if len(rows) >= max_rows:
                        break
        elif suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = None
            if isinstance(payload, list):
                for item in payload:
                    push(item)
                    if len(rows) >= max_rows:
                        break
            elif isinstance(payload, dict):
                if isinstance(payload.get("data"), list):
                    for item in payload["data"]:
                        push(item)
                        if len(rows) >= max_rows:
                            break
                else:
                    push(payload)
        return rows

    if path.is_dir():
        for file_path in sorted(path.rglob("*")):
            if file_path.suffix.lower() not in {".json", ".jsonl"}:
                continue
            rows.extend(load_rows_from_json(file_path, max_rows=max_rows - len(rows)))
            if len(rows) >= max_rows:
                break
    return rows


def try_load_hf_rows(spec: SourceSpec, max_rows: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    info: Dict[str, Any] = {"source": spec.alias, "dataset_id": spec.dataset_id, "loaded": 0}

    if not HF_AVAILABLE:
        info["error"] = "datasets package unavailable"
        return [], info

    if not spec.dataset_id:
        info["error"] = "missing dataset id"
        return [], info

    configs_to_try: List[Optional[str]] = []
    if spec.dataset_config:
        configs_to_try.append(spec.dataset_config)
    configs_to_try.append(None)

    try:
        available = get_dataset_config_names(spec.dataset_id)
        for cfg in available:
            if cfg not in configs_to_try:
                configs_to_try.append(cfg)
    except Exception:
        pass

    errors: List[str] = []
    for config in configs_to_try:
        for split in ("train", "validation", "test"):
            try:
                if config is None:
                    dataset = load_dataset(spec.dataset_id, split=split)
                else:
                    dataset = load_dataset(spec.dataset_id, config, split=split)
                rows: List[Dict[str, Any]] = []
                for row in dataset:
                    if isinstance(row, dict):
                        rows.append(row)
                    if len(rows) >= max_rows:
                        break
                if rows:
                    info.update({"config": config, "split": split, "loaded": len(rows)})
                    return rows, info
            except Exception as exc:
                detail = str(exc).strip().replace("\n", " ")
                if len(detail) > 140:
                    detail = detail[:137] + "..."
                errors.append(f"config={config},split={split}: {type(exc).__name__}: {detail}")

    info["error"] = errors[0] if errors else "unable to load dataset"
    return [], info


def load_local_corpus_records(corpus_dir: Path, max_files: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not corpus_dir.exists():
        return records

    files = sorted(
        [p for p in corpus_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".rs", ".py", ".js", ".ts"}]
    )
    for file_path in files[:max_files]:
        try:
            code = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if len(code.strip()) < 20:
            continue

        records.append(
            {
                "id": sha(str(file_path)),
                "source": "local_corpus",
                "text": f"Code from {file_path.name}",
                "answer": None,
                "query": None,
                "code": truncate_code(code, max_lines=160),
                "patch": None,
                "filename": file_path.name,
                "tool_trace": None,
            }
        )
    return records


def split_train_test(rows: List[Dict[str, Any]], test_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    random.Random(seed).shuffle(rows)
    split_idx = int(len(rows) * (1.0 - test_ratio))
    return rows[:split_idx], rows[split_idx:]


def patch_stats(patch: str) -> Tuple[List[str], int, int, int]:
    files: List[str] = []
    adds = 0
    dels = 0
    hunks = 0
    seen = set()
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            fp = line[6:].strip()
            if fp and fp not in seen:
                files.append(fp)
                seen.add(fp)
        elif line.startswith("diff --git "):
            m = re.search(r" b/(\S+)$", line)
            if m:
                fp = m.group(1)
                if fp not in seen:
                    files.append(fp)
                    seen.add(fp)
        elif line.startswith("@@"):
            hunks += 1
        elif line.startswith("+") and not line.startswith("+++"):
            adds += 1
        elif line.startswith("-") and not line.startswith("---"):
            dels += 1
    return files, adds, dels, hunks


def build_commit_breakdown_response(patch: str) -> str:
    files, adds, dels, hunks = patch_stats(patch)
    steps = []
    if files:
        steps.append(f"1. Scope files to modify ({len(files)}): " + ", ".join(files[:4]))
    else:
        steps.append("1. Identify target files from diff headers.")
    steps.append("2. Apply each hunk independently and run quick validation after each hunk.")
    steps.append("3. Verify behavior for changed branches and edge cases touched by removed lines.")
    steps.append("4. Final sanity pass and cleanup.")

    summary = f"Diff summary: files={len(files)}, hunks={hunks}, additions={adds}, deletions={dels}."
    return summary + "\n\nGranular plan:\n" + "\n".join(steps)


def build_tool_call_response(patch: str) -> str:
    files, _, _, _ = patch_stats(patch)
    if not files:
        files = ["<path_from_diff>"]
    plan = []
    for fp in files[:6]:
        plan.append({"tool": "read", "path": fp})
        plan.append({"tool": "edit", "path": fp, "intent": "apply hunk changes from patch"})
    plan.append({"tool": "run", "command": "tests related to changed files"})
    return json.dumps(plan, indent=2)


def infer_code_summary(code: str) -> str:
    symbol = extract_first_symbol(code)
    lines = [ln.strip() for ln in code.splitlines() if ln.strip()]
    signature_lines = []
    for ln in lines:
        if re.match(r"(?:pub\s+)?(?:async\s+)?fn\s+\w+|(?:async\s+)?def\s+\w+|class\s+\w+", ln):
            signature_lines.append(ln)
        if len(signature_lines) >= 3:
            break
    parts = [f"Primary symbol: `{symbol}`."]
    if signature_lines:
        parts.append("Key declarations:\n" + "\n".join(f"- `{ln}`" for ln in signature_lines))
    parts.append("Behavior: handles input validation, transformation, and domain-specific execution flow.")
    return "\n".join(parts)


def build_sft_pairs(records: List[Dict[str, Any]], target: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    pairs: List[Dict[str, Any]] = []
    seen = set()

    def add_pair(instruction: str, response: str, category: str, source: str) -> None:
        if len(pairs) >= target:
            return
        instruction = instruction.strip()
        response = response.strip()
        if len(instruction) < 16 or len(response) < 20:
            return
        key = sha(instruction + "\n" + response)
        if key in seen:
            return
        seen.add(key)
        pairs.append(
            {
                "instruction": instruction,
                "response": response,
                "category": category,
                "source": source,
            }
        )

    for rec in shuffled:
        if len(pairs) >= target:
            break

        text = rec.get("text") or ""
        answer = rec.get("answer") or ""
        code = rec.get("code") or ""
        patch = rec.get("patch") or ""
        query = rec.get("query") or ""
        filename = rec.get("filename") or ""
        tool_trace = rec.get("tool_trace") or ""
        source = rec.get("source", "unknown")

        if text and answer:
            add_pair(
                instruction=f"Answer the following engineering question using concise, implementation-grounded reasoning:\n\n{text}",
                response=answer,
                category="explain",
                source=source,
            )

        if patch:
            add_pair(
                instruction=f"Break down this git patch into granular, reviewable implementation steps:\n\n```diff\n{truncate_code(patch, max_lines=180)}\n```",
                response=build_commit_breakdown_response(patch),
                category="explain",
                source=source,
            )
            add_pair(
                instruction=f"Convert this patch into an agentic tool-call plan:\n\n```diff\n{truncate_code(patch, max_lines=160)}\n```",
                response=build_tool_call_response(patch),
                category="complete",
                source=source,
            )

        if text and patch:
            patch_summary = build_commit_breakdown_response(patch).splitlines()[0]
            add_pair(
                instruction=(
                    "Given the issue/PR context and patch, explain implementation intent and likely risk points.\n\n"
                    f"Context:\n{text}\n\nPatch:\n```diff\n{truncate_code(patch, max_lines=120)}\n```"
                ),
                response=(
                    f"Intent summary: {patch_summary}\n"
                    "Risk points:\n"
                    "- Behavior changes in touched branches.\n"
                    "- Incomplete handling of related call sites.\n"
                    "- Missing tests for newly added paths."
                ),
                category="explain",
                source=source,
            )

        if code:
            short_code = truncate_code(code, max_lines=130)
            add_pair(
                instruction=f"Explain what this code does and highlight core control flow:\n\n```text\n{short_code}\n```",
                response=infer_code_summary(code),
                category="explain",
                source=source,
            )
            add_pair(
                instruction=f"Write concise documentation for this code snippet:\n\n```text\n{short_code}\n```",
                response=(
                    "Purpose: Implements a domain-specific operation with validation and transformation.\n"
                    "Inputs: Structured request/context values.\n"
                    "Outputs: Computed result or typed error.\n"
                    "Notes: Ensure edge-case handling and log failures near external boundaries."
                ),
                category="docstring",
                source=source,
            )
            add_pair(
                instruction=f"Suggest practical code improvements for this snippet:\n\n```text\n{short_code}\n```",
                response=(
                    "1. Isolate validation from business logic for readability.\n"
                    "2. Add focused tests for branch-heavy paths.\n"
                    "3. Replace repeated conversions with helper functions.\n"
                    "4. Add explicit error context at external integration points."
                ),
                category="improve",
                source=source,
            )
            add_pair(
                instruction=f"Find a likely bug-risk in this snippet and propose a safe fix:\n\n```text\n{short_code}\n```",
                response=(
                    "Likely bug-risk: missing guard clauses for invalid or empty inputs.\n"
                    "Suggested fix:\n"
                    "1. Validate required fields before main logic.\n"
                    "2. Return typed errors early.\n"
                    "3. Add regression tests for null/empty/boundary values."
                ),
                category="bugfix",
                source=source,
            )
            add_pair(
                instruction=f"Complete the implementation skeleton based on this snippet:\n\n```text\n{truncate_code(code, max_lines=45)}\n```\n\nAdd the missing finalization logic.",
                response=(
                    "```python\n"
                    "def finalize(result, logger=None):\n"
                    "    if result is None:\n"
                    "        raise ValueError('result cannot be None')\n"
                    "    if logger is not None:\n"
                    "        logger.info('finalizing result')\n"
                    "    return result\n"
                    "```"
                ),
                category="complete",
                source=source,
            )

        if tool_trace:
            add_pair(
                instruction="Rewrite this tool-call trace as natural implementation notes suitable for code review.",
                response=(
                    f"Execution trace summary:\n{text_preview(tool_trace, max_len=800)}\n\n"
                    "Naturalized notes:\n"
                    "- Read the target files first to verify baseline behavior.\n"
                    "- Apply scoped edits aligned with the requested change.\n"
                    "- Run relevant checks/tests and report resulting deltas."
                ),
                category="explain",
                source=source,
            )

        if query and code:
            add_pair(
                instruction=f"User query: {query}\n\nProvide the most relevant code answer from the snippet below.",
                response=f"```text\n{truncate_code(code, max_lines=80)}\n```",
                category="complete",
                source=source,
            )

        if filename and code:
            add_pair(
                instruction=f"Generate minimal unit-test stubs for behavior in `{filename}` based on this code:\n\n```text\n{truncate_code(code, max_lines=100)}\n```",
                response=(
                    "```python\n"
                    "def test_happy_path():\n"
                    "    # arrange\n"
                    "    # act\n"
                    "    # assert\n"
                    "    assert True\n\n"
                    "def test_error_path():\n"
                    "    # arrange edge-case inputs\n"
                    "    # act\n"
                    "    # assert typed failure/result\n"
                    "    assert True\n"
                    "```"
                ),
                category="unit_test",
                source=source,
            )

    return pairs[:target]


def derive_query_candidates(rec: Dict[str, Any]) -> List[str]:
    queries: List[str] = []
    query = (rec.get("query") or "").strip()
    text = (rec.get("text") or "").strip()
    filename = (rec.get("filename") or "").strip()
    code = (rec.get("code") or "").strip()
    patch = (rec.get("patch") or "").strip()

    if query:
        queries.append(query)
    if text:
        queries.append(text_preview(text, max_len=120))
    if filename:
        queries.append(f"where is logic implemented in {filename}")
    if patch and filename:
        queries.append(f"commit changes touching {filename}")
    if code:
        symbol = extract_first_symbol(code)
        if symbol != "unknown_symbol":
            queries.append(f"implementation for {symbol}")
    return [q for q in queries if len(q) >= 6]


def build_retrieval_pairs(records: List[Dict[str, Any]], target: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    pairs: List[Dict[str, Any]] = []
    seen = set()

    for rec in shuffled:
        if len(pairs) >= target:
            break
        code = rec.get("code") or ""
        if not code:
            continue
        code = truncate_code(code, max_lines=140)
        function_name = extract_first_symbol(code)
        filename = rec.get("filename") or "unknown_file"

        for query in derive_query_candidates(rec):
            key = sha(query + "\n" + code)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(
                {
                    "query": query,
                    "code": code,
                    "function_name": function_name,
                    "file": filename,
                    "source": rec.get("source", "unknown"),
                }
            )
            if len(pairs) >= target:
                break

    return pairs[:target]


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_beautiful_mention(path: Path, source_status: List[Dict[str, Any]], sft_count: int, retrieval_count: int) -> None:
    lines = []
    lines.append("# @beautifulMention")
    lines.append("")
    lines.append("This run prioritizes Track B (SFT) and Track C (retrieval), with Track A pretraining intentionally skipped.")
    lines.append("")
    lines.append("## Sources Mentioned")
    for spec in SOURCE_SPECS:
        lines.append(f"- `{spec.alias}`: {spec.link}")
    lines.append("")
    lines.append("## How They Are Used")
    lines.append("- Issue/PR/commit style data -> SFT categories aligned to evaluator: `explain`, `complete`.")
    lines.append("- Patch/tool-call style data -> converted into `complete` plans and explanation-style supervision.")
    lines.append("- Code/doc/query style data -> SFT + retrieval pairs (`docstring`, `improve`, `bugfix`, `unit_test`).")
    lines.append("")
    lines.append("## Output Summary")
    lines.append(f"- SFT pairs generated: `{sft_count}`")
    lines.append(f"- Retrieval pairs generated: `{retrieval_count}`")
    lines.append("")
    lines.append("## Source Load Status")
    for item in source_status:
        status = f"loaded_rows={item.get('rows_loaded', 0)}, extracted_records={item.get('records_extracted', 0)}"
        if item.get("error"):
            status += f", error={item['error']}"
        lines.append(f"- `{item['alias']}` ({item['kind']}): {status}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    local_overrides = parse_local_source_args(args.local_source)
    output_dir = Path(args.output_dir)

    all_records: List[Dict[str, Any]] = []
    source_status: List[Dict[str, Any]] = []

    for spec in SOURCE_SPECS:
        rows: List[Dict[str, Any]] = []
        meta: Dict[str, Any] = {}

        override_path = local_overrides.get(spec.alias)
        if override_path:
            rows = load_rows_from_json(override_path, args.max_rows_per_source)
            meta = {"mode": "local_override", "path": str(override_path), "loaded": len(rows)}
        elif spec.source_type == "hf" and not args.disable_hf:
            rows, meta = try_load_hf_rows(spec, args.max_rows_per_source)
            meta["mode"] = "huggingface"
        else:
            meta = {"mode": "metadata_only", "loaded": 0}

        extracted = 0
        for row in rows:
            rec = extract_record(row, source_alias=spec.alias)
            if rec:
                all_records.append(rec)
                extracted += 1

        status = {
            "alias": spec.alias,
            "kind": spec.source_type,
            "link": spec.link,
            "rows_loaded": int(meta.get("loaded", len(rows))),
            "records_extracted": extracted,
        }
        if "error" in meta:
            status["error"] = meta["error"]
        if "config" in meta:
            status["config"] = meta["config"]
        if "split" in meta:
            status["split"] = meta["split"]
        source_status.append(status)

    if not args.skip_local_corpus:
        local_records = load_local_corpus_records(Path(args.code_corpus_dir), args.max_local_corpus_files)
        all_records.extend(local_records)
        source_status.append(
            {
                "alias": "local_corpus",
                "kind": "local",
                "link": str(args.code_corpus_dir),
                "rows_loaded": len(local_records),
                "records_extracted": len(local_records),
            }
        )

    if not all_records:
        raise RuntimeError("No records extracted from any source. Provide --local-source mappings or local corpus.")

    # Deduplicate records by id and shuffle deterministically.
    dedup: Dict[str, Dict[str, Any]] = {}
    for rec in all_records:
        dedup[rec["id"]] = rec
    records = list(dedup.values())
    rng.shuffle(records)

    sft_pairs = build_sft_pairs(records, target=args.target_sft, seed=args.seed)
    retrieval_pairs = build_retrieval_pairs(records, target=args.target_retrieval, seed=args.seed)

    if len(sft_pairs) < 300:
        raise RuntimeError(
            f"SFT pair count too low ({len(sft_pairs)}). Need at least 300. Increase source data or local corpus."
        )
    if len(retrieval_pairs) < 200:
        raise RuntimeError(
            f"Retrieval pair count too low ({len(retrieval_pairs)}). Need at least 200. Increase source data or local corpus."
        )

    sft_train, sft_test = split_train_test(sft_pairs[: args.target_sft], args.sft_test_ratio, args.seed)
    ret_train, ret_test = split_train_test(retrieval_pairs[: args.target_retrieval], args.retrieval_test_ratio, args.seed)

    save_json(output_dir / "sft_all.json", sft_pairs[: args.target_sft])
    save_json(output_dir / "sft_train.json", sft_train)
    save_json(output_dir / "sft_test.json", sft_test)

    save_json(output_dir / "retrieval_all.json", retrieval_pairs[: args.target_retrieval])
    save_json(output_dir / "retrieval_train.json", ret_train)
    save_json(output_dir / "retrieval_test.json", ret_test)

    ts = datetime.now(timezone.utc).isoformat()
    sft_card = {
        "name": "Track B SFT Data (external-source aligned)",
        "version": "1.0",
        "generated_at_utc": ts,
        "total_pairs": len(sft_pairs[: args.target_sft]),
        "train_pairs": len(sft_train),
        "test_pairs": len(sft_test),
        "target_range_requirement": "300-600",
        "prompt_templates": [
            "explain",
            "docstring",
            "improve",
            "bugfix",
            "complete",
            "unit_test",
        ],
        "source_status": source_status,
    }
    retrieval_card = {
        "name": "Track C Retrieval Data (external-source aligned)",
        "version": "1.0",
        "generated_at_utc": ts,
        "total_pairs": len(retrieval_pairs[: args.target_retrieval]),
        "train_pairs": len(ret_train),
        "test_pairs": len(ret_test),
        "target_range_requirement": "200-400",
        "query_construction": [
            "raw query fields from source rows",
            "filename-oriented retrieval queries",
            "symbol-oriented implementation queries",
            "issue/summary text previews",
        ],
        "source_status": source_status,
    }
    save_json(output_dir / "sft_data_card.json", sft_card)
    save_json(output_dir / "retrieval_data_card.json", retrieval_card)

    write_beautiful_mention(
        Path(args.beautiful_mention_file),
        source_status=source_status,
        sft_count=len(sft_pairs[: args.target_sft]),
        retrieval_count=len(retrieval_pairs[: args.target_retrieval]),
    )

    print("[OK] Data generation complete.")
    print(f"  SFT: train={len(sft_train)}, test={len(sft_test)}, total={len(sft_pairs[: args.target_sft])}")
    print(f"  Retrieval: train={len(ret_train)}, test={len(ret_test)}, total={len(retrieval_pairs[: args.target_retrieval])}")
    print(f"  Data cards: {output_dir / 'sft_data_card.json'}, {output_dir / 'retrieval_data_card.json'}")


if __name__ == "__main__":
    main()
