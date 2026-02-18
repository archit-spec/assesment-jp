"""
AI-Powered SFT & Retrieval Data Generation
===========================================
Uses an LLM (via OpenAI-compatible API) to:
  1. Read real code files from the corpus (hyperswitch Rust + verl Python)
  2. Mine hard developer questions — high-level (architecture/design) and
     low-level (function behavior, edge cases) — plus mermaid diagram tasks
  3. Answer each question with the code as context
  4. Produce SFT pairs (instruction → response) and retrieval pairs (query → code)

Output files (drop-in compatible with existing training scripts):
  data/ai_sft_train.json, data/ai_sft_test.json
  data/ai_retrieval_train.json, data/ai_retrieval_test.json
  data/ai_sft_data_card.json, data/ai_retrieval_data_card.json

Usage:
  python generate_ai_data.py                    # full run, both corpora
  python generate_ai_data.py --max-files 10     # quick test
  python generate_ai_data.py --repo hyperswitch # single repo
  python generate_ai_data.py --dry-run          # mine questions only, no answers
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI

# ── Load env ────────────────────────────────────────────────────────────────
load_dotenv()

# Use OpenRouter — falls back to OPENAI_BASE_URL / OPENAI_API_KEY if not set
API_KEY    = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", "")
API_BASE   = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/")
MODEL      = os.getenv("OPENROUTER_MODEL", "qwen/qwen3.5-397b-a17b")

# ── Config ───────────────────────────────────────────────────────────────────
CORPUS_CONFIGS = {
    "hyperswitch": {
        "corpus_dir": "data/code_corpus_hyperswitch",
        "language": "Rust",
        "repo": "juspay/hyperswitch",
        "description": "a payment orchestration platform",
    },
    "verl": {
        "corpus_dir": "data/code_corpus_verl",
        "language": "Python",
        "repo": "volcengine/verl",
        "description": "a reinforcement learning from human feedback (RLHF) training framework",
    },
}

OUTPUT_DIR          = "data"
SFT_TRAIN_OUT       = "data/ai_sft_train.json"
SFT_TEST_OUT        = "data/ai_sft_test.json"
SFT_ALL_OUT         = "data/ai_sft_all.json"
SFT_CARD_OUT        = "data/ai_sft_data_card.json"
RETRIEVAL_TRAIN_OUT = "data/ai_retrieval_train.json"
RETRIEVAL_TEST_OUT  = "data/ai_retrieval_test.json"
RETRIEVAL_ALL_OUT   = "data/ai_retrieval_all.json"
RETRIEVAL_CARD_OUT  = "data/ai_retrieval_data_card.json"

MAX_FILE_CHARS      = 6000   # truncate files to this many chars before sending
MAX_CONCURRENT      = 4      # max parallel LLM calls (OpenRouter supports higher concurrency)
RANDOM_SEED         = 42
SFT_TEST_RATIO      = 0.15
RETRIEVAL_TEST_RATIO= 0.20

# Questions per file
HIGH_LEVEL_Q        = 3
LOW_LEVEL_Q         = 3
MERMAID_Q           = 1      # mermaid diagram tasks per file


# ── Prompts ──────────────────────────────────────────────────────────────────

MINE_SYSTEM = """\
You are a senior software engineer doing a thorough code review of a production codebase.
Your job is to generate the kind of hard, insightful questions that a new developer joining
the team would genuinely need answered to understand and contribute to this code.
Always return valid JSON — no markdown fences, no extra text."""

MINE_USER_TMPL = """\
Here is a {language} file from {repo} ({description}):

File: {filename}
```{lang_lower}
{code}
```

Generate exactly the following questions about this file:
- {high_n} HIGH-LEVEL questions: about architecture, design decisions, module responsibilities,
  data flow, why this abstraction exists, how it fits into the larger system.
- {low_n} LOW-LEVEL questions: about specific functions, edge cases, error handling,
  performance implications, subtle bugs, or tricky implementation details.
- {mermaid_n} MERMAID task(s): ask to "Draw a mermaid diagram showing [something specific
  and interesting about this file's structure, flow, or relationships]".

Return ONLY this JSON (no markdown fences):
{{
  "high_level": ["question 1", "question 2", ...],
  "low_level": ["question 1", "question 2", ...],
  "mermaid": ["Draw a mermaid diagram showing ..."]
}}"""

ANSWER_SYSTEM = """\
You are an expert {language} developer with deep knowledge of {repo} ({description}).
Answer the developer's question concisely, accurately, and with concrete references to the code.
For mermaid diagram tasks, produce a valid mermaid diagram inside a ```mermaid ... ``` code block,
followed by a brief explanation."""

ANSWER_USER_TMPL = """\
Code context — {language} file `{filename}` from {repo}:
```{lang_lower}
{code}
```

Question / Task:
{question}"""

LABEL_SYSTEM = """\
You are a senior engineer writing a semantic search index for a codebase.
Your job is to produce short, precise labels that describe what a piece of code DOES,
so that a developer can find it by searching for those labels.
Always return valid JSON — no markdown fences, no extra text."""

LABEL_USER_TMPL = """\
Here is a {language} file from {repo} ({description}):

File: {filename}
```{lang_lower}
{code}
```

Generate 4-6 short semantic labels for this file. Each label should be a concise phrase
(3-8 words) that a developer might type into a search box to find this code.
Cover different aspects: what it does, what domain it belongs to, key patterns used.

Return ONLY this JSON (no markdown fences):
{{"labels": ["label 1", "label 2", "label 3", ...]}}"""

# ── Commit Mining Prompts ─────────────────────────────────────────────────────

COMMIT_MINE_SYSTEM = """\
You are a senior software engineer reviewing a commit to a production codebase.
Your job is to generate insightful questions and answers that a developer would
need to understand this change — why it was made, what it does, and its implications.
Always return valid JSON — no markdown fences, no extra text."""

COMMIT_MINE_USER_TMPL = """\
Here is a commit to the {repo} codebase:

Commit message:
{commit_msg}

Files changed:
{files_changed}

Generate 3 insightful question-answer pairs about this commit. Questions should cover:
- WHY this change was made (motivation, design decision)
- WHAT the change does (technical detail)
- HOW it affects the system (implications, risks, related components)

Return ONLY this JSON (no markdown fences):
{{
  "pairs": [
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}}
  ]
}}"""

SWEBENCH_MINE_SYSTEM = """\
You are a senior software engineer reviewing a bug fix or feature implementation.
Your job is to generate insightful question-answer pairs that help a developer
understand the problem, the solution, and the reasoning behind the approach.
Always return valid JSON — no markdown fences, no extra text."""

SWEBENCH_MINE_USER_TMPL = """\
Here is a GitHub issue and its fix from the {repo} codebase:

Problem statement:
{problem_statement}

Patch (diff):
```diff
{patch}
```

Generate 3 insightful question-answer pairs about this issue and fix. Cover:
- What the root cause of the bug/need was
- How the patch addresses it and why this approach was chosen
- What edge cases or follow-up concerns remain

Return ONLY this JSON (no markdown fences):
{{
  "pairs": [
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}}
  ]
}}"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def truncate_code(code: str, max_chars: int = MAX_FILE_CHARS) -> str:
    if len(code) <= max_chars:
        return code
    # Keep first 2/3 and last 1/3 to preserve both header and tail
    head = int(max_chars * 0.67)
    tail = max_chars - head
    return code[:head] + f"\n\n... [truncated {len(code)-max_chars} chars] ...\n\n" + code[-tail:]


def pair_hash(instruction: str, source_file: str) -> str:
    return hashlib.md5(f"{instruction}|{source_file}".encode()).hexdigest()


def _is_quality_code_file(code: str, language: str) -> bool:
    """Return True if the file has enough real code to be worth processing."""
    lines = code.splitlines()
    total = len(lines)
    if total < 30:
        return False
    if language == "Rust":
        comment_prefixes = ("//", "/*", "*", "#!")
        code_keywords = ("fn ", "impl ", "struct ", "enum ", "trait ", "pub ", "let ", "match ")
    else:
        comment_prefixes = ("#",)
        code_keywords = ("def ", "class ", "import ", "from ", "return ", "if ", "for ", "async ", "await ")
    code_lines = [l for l in lines if l.strip() and not l.strip().startswith(comment_prefixes)]
    if len(code_lines) < 15:
        return False
    comment_ratio = (total - len(code_lines)) / max(total, 1)
    if comment_ratio > 0.7:
        return False
    has_definition = any(any(kw in l for kw in code_keywords) for l in code_lines)
    return has_definition


def load_corpus_files(corpus_dir: str, max_files: Optional[int] = None) -> List[Dict]:
    """Load quality code files from a corpus directory."""
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        print(f"[WARN] Corpus dir not found: {corpus_dir}")
        return []

    # Infer language from corpus dir name
    language = "Rust" if "hyperswitch" in corpus_dir else "Python"

    files = []
    skipped = 0
    for p in sorted(corpus_path.iterdir()):
        if p.is_file() and not p.name.startswith("."):
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                if not content.strip():
                    continue
                if not _is_quality_code_file(content, language):
                    skipped += 1
                    continue
                files.append({"filename": p.name, "code": content, "path": str(p)})
            except Exception as e:
                print(f"[WARN] Could not read {p}: {e}")

    if skipped:
        print(f"[FILTER] Skipped {skipped} low-quality files, kept {len(files)}")
    random.shuffle(files)
    if max_files:
        files = files[:max_files]
    return files


def extract_json_block(text: str) -> Optional[Dict]:
    """Try to extract a JSON object from LLM output (handles markdown fences)."""
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # Find first { ... } block
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end+1])
    except json.JSONDecodeError:
        return None


# ── LLM Calls ────────────────────────────────────────────────────────────────

async def mine_questions(
    client: AsyncOpenAI,
    file_info: Dict,
    cfg: Dict,
    semaphore: asyncio.Semaphore,
    retries: int = 2,
) -> Optional[Dict]:
    """Ask the LLM to generate questions for a code file."""
    code = truncate_code(file_info["code"])
    user_msg = MINE_USER_TMPL.format(
        language=cfg["language"],
        lang_lower=cfg["language"].lower(),
        repo=cfg["repo"],
        description=cfg["description"],
        filename=file_info["filename"],
        code=code,
        high_n=HIGH_LEVEL_Q,
        low_n=LOW_LEVEL_Q,
        mermaid_n=MERMAID_Q,
    )

    async with semaphore:
        for attempt in range(retries + 1):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": MINE_SYSTEM},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                )
                raw = resp.choices[0].message.content or ""
                parsed = extract_json_block(raw)
                if parsed and isinstance(parsed.get("high_level"), list):
                    return parsed
                print(f"  [WARN] Bad JSON from mine_questions for {file_info['filename']}, attempt {attempt+1}")
            except Exception as e:
                print(f"  [ERROR] mine_questions {file_info['filename']} attempt {attempt+1}: {e}")
                if attempt < retries:
                    delay = 2 * (2 ** attempt)  # 2s, 4s
                    await asyncio.sleep(delay)
    return None


async def answer_question(
    client: AsyncOpenAI,
    question: str,
    file_info: Dict,
    cfg: Dict,
    semaphore: asyncio.Semaphore,
    retries: int = 2,
) -> Optional[str]:
    """Ask the LLM to answer a question about a code file."""
    code = truncate_code(file_info["code"])
    user_msg = ANSWER_USER_TMPL.format(
        language=cfg["language"],
        lang_lower=cfg["language"].lower(),
        repo=cfg["repo"],
        description=cfg["description"],
        filename=file_info["filename"],
        code=code,
        question=question,
    )

    async with semaphore:
        for attempt in range(retries + 1):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": ANSWER_SYSTEM.format(
                            language=cfg["language"],
                            repo=cfg["repo"],
                            description=cfg["description"],
                        )},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.3,
                    max_tokens=1024,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                print(f"  [ERROR] answer_question attempt {attempt+1}: {e}")
                if attempt < retries:
                    delay = 2 * (2 ** attempt)
                    await asyncio.sleep(delay)
    return None


async def label_code_chunk(
    client: AsyncOpenAI,
    file_info: Dict,
    cfg: Dict,
    semaphore: asyncio.Semaphore,
    retries: int = 2,
) -> List[str]:
    """Ask the LLM to generate short semantic labels for a code file."""
    code = truncate_code(file_info["code"], max_chars=3000)
    user_msg = LABEL_USER_TMPL.format(
        language=cfg["language"],
        lang_lower=cfg["language"].lower(),
        repo=cfg["repo"],
        description=cfg["description"],
        filename=file_info["filename"],
        code=code,
    )

    async with semaphore:
        for attempt in range(retries + 1):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": LABEL_SYSTEM},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=0.5,
                    max_tokens=256,
                )
                raw = resp.choices[0].message.content or ""
                parsed = extract_json_block(raw)
                if parsed and isinstance(parsed.get("labels"), list):
                    # Filter: keep labels between 3 and 80 chars
                    labels = [l.strip() for l in parsed["labels"]
                              if isinstance(l, str) and 3 <= len(l.strip()) <= 80]
                    return labels[:6]
                print(f"  [WARN] Bad JSON from label_code_chunk for {file_info['filename']}, attempt {attempt+1}")
            except Exception as e:
                print(f"  [ERROR] label_code_chunk {file_info['filename']} attempt {attempt+1}: {e}")
                if attempt < retries:
                    delay = 2 * (2 ** attempt)
                    await asyncio.sleep(delay)
    return []


# ── Commit Mining ─────────────────────────────────────────────────────────────

def load_commit_files(paths: List[str], max_commits: Optional[int] = None) -> List[Dict]:
    """Load commit records from JSON files. Handles both formats:
    - [{prompt, response}, ...] — commit message + files changed
    - {problem_statement, patch, repo, ...} — SWE-bench style (single object or list)
    """
    records = []
    for path_str in paths:
        p = Path(path_str)
        if not p.exists():
            print(f"[WARN] Commit file not found: {path_str}")
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Could not parse {path_str}: {e}")
            continue

        if isinstance(data, list):
            for item in data:
                if "prompt" in item and "response" in item:
                    # commit-message → files-changed format
                    records.append({"format": "commit", "source": p.name, **item})
                elif "problem_statement" in item:
                    records.append({"format": "swebench", "source": p.name, **item})
        elif isinstance(data, dict):
            if "problem_statement" in data:
                records.append({"format": "swebench", "source": p.name, **data})

    random.shuffle(records)
    if max_commits:
        records = records[:max_commits]
    print(f"[COMMITS] Loaded {len(records)} commit records from {len(paths)} file(s)")
    return records


async def mine_commit_pairs(
    client: AsyncOpenAI,
    record: Dict,
    semaphore: asyncio.Semaphore,
    retries: int = 2,
) -> List[Dict]:
    """Generate SFT pairs from a single commit record."""
    if record["format"] == "commit":
        repo = "juspay/hyperswitch"
        commit_msg = record["prompt"][:1000]  # truncate long messages
        files_changed = record["response"][:2000]
        user_msg = COMMIT_MINE_USER_TMPL.format(
            repo=repo,
            commit_msg=commit_msg,
            files_changed=files_changed,
        )
        system_msg = COMMIT_MINE_SYSTEM
        source_id = commit_msg[:80]
    else:  # swebench
        repo = record.get("repo", "juspay/hyperswitch")
        problem = record.get("problem_statement", "")[:1500]
        patch = record.get("patch", "")[:1500]
        user_msg = SWEBENCH_MINE_USER_TMPL.format(
            repo=repo,
            problem_statement=problem,
            patch=patch,
        )
        system_msg = SWEBENCH_MINE_SYSTEM
        source_id = record.get("instance_id", problem[:60])

    async with semaphore:
        for attempt in range(retries + 1):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=0.4,
                    max_tokens=1024,
                )
                raw = resp.choices[0].message.content or ""
                parsed = extract_json_block(raw)
                if parsed and isinstance(parsed.get("pairs"), list):
                    pairs = []
                    for item in parsed["pairs"]:
                        q = item.get("question", "").strip()
                        a = item.get("answer", "").strip()
                        if q and a:
                            pairs.append({
                                "instruction": q,
                                "response":    a,
                                "category":    f"commit_{record['format']}",
                                "source_function": "",
                                "source_file": source_id,
                                "repo":        repo,
                                "language":    "Rust",  # hyperswitch is Rust
                            })
                    return pairs
                print(f"  [WARN] Bad JSON from mine_commit_pairs, attempt {attempt+1}")
            except Exception as e:
                print(f"  [ERROR] mine_commit_pairs attempt {attempt+1}: {e}")
                if attempt < retries:
                    delay = 2 * (2 ** attempt)
                    await asyncio.sleep(delay)
    return []


async def run_commit_pipeline(
    commit_files: List[str],
    max_commits: Optional[int],
    dry_run: bool,
) -> List[Dict]:
    """Mine SFT pairs from commit records."""
    records = load_commit_files(commit_files, max_commits)
    if not records:
        return []

    client    = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    print(f"\n{'='*60}")
    print(f"[COMMITS] Mining questions from {len(records)} commit records...")
    print(f"{'='*60}")

    all_pairs: List[Dict] = []
    for i, record in enumerate(records):
        fmt = record["format"]
        src = record["source"]
        label = record.get("prompt", record.get("instance_id", "?"))[:60]
        print(f"[{i+1}/{len(records)}] [{fmt}] {src}: {label}")

        if dry_run:
            print(f"  [DRY RUN] Would mine 3 Q&A pairs from this commit")
            continue

        pairs = await mine_commit_pairs(client, record, semaphore)
        all_pairs.extend(pairs)
        print(f"  [DONE] {len(pairs)} SFT pairs")

    print(f"[COMMITS] Total: {len(all_pairs)} SFT pairs from commits")
    return all_pairs


async def process_file(
    client: AsyncOpenAI,
    file_info: Dict,
    cfg: Dict,
    repo_name: str,
    semaphore: asyncio.Semaphore,
    dry_run: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Process one file:
      1. Mine questions (high-level, low-level, mermaid)
      2. Answer each question
      3. Return (sft_pairs, retrieval_pairs)
    """
    sft_pairs       = []
    retrieval_pairs = []

    print(f"  [MINE+LABEL] {repo_name}/{file_info['filename']}")
    # Run question mining first, then labeling (sequential to respect rate limit)
    questions_data = await mine_questions(client, file_info, cfg, semaphore)
    labels = await label_code_chunk(client, file_info, cfg, semaphore)
    if not questions_data:
        print(f"  [SKIP] Could not mine questions for {file_info['filename']}")
        return [], []

    high_qs   = questions_data.get("high_level", [])[:HIGH_LEVEL_Q]
    low_qs    = questions_data.get("low_level",  [])[:LOW_LEVEL_Q]
    mermaid_qs= questions_data.get("mermaid",    [])[:MERMAID_Q]

    all_questions = [
        (q, "high_level") for q in high_qs
    ] + [
        (q, "low_level")  for q in low_qs
    ] + [
        (q, "mermaid")    for q in mermaid_qs
    ]

    if dry_run:
        # In dry-run mode, just show the questions without answering
        for q, cat in all_questions:
            print(f"    [{cat}] {q}")
        return [], []

    # Answer all questions concurrently (still within semaphore per call)
    answer_tasks = [
        answer_question(client, q, file_info, cfg, semaphore)
        for q, _ in all_questions
    ]
    answers = await asyncio.gather(*answer_tasks)

    seen_hashes = set()
    code_snippet = truncate_code(file_info["code"], max_chars=2000)

    # ── Label-based retrieval pairs (one per label) ──────────────────────────
    for label in labels:
        h = pair_hash(label, file_info["filename"])
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        retrieval_pairs.append({
            "query":         label,
            "code":          code_snippet,
            "function_name": file_info["filename"].replace(".py", "").replace(".rs", ""),
            "file":          file_info["filename"],
            "repo":          repo_name,
            "language":      cfg["language"],
            "num_lines":     len(file_info["code"].splitlines()),
            "label_source":  "synthetic_label",
        })

    for (question, category), answer in zip(all_questions, answers):
        if not answer:
            continue

        h = pair_hash(question, file_info["filename"])
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        # SFT pair
        sft_pairs.append({
            "instruction": question,
            "response":    answer,
            "category":    category,
            "source_function": "",
            "source_file": file_info["filename"],
            "repo":        repo_name,
            "language":    cfg["language"],
        })

        # Retrieval pair — use question as query, code snippet as the document
        retrieval_pairs.append({
            "query":         question,
            "code":          code_snippet,
            "function_name": file_info["filename"].replace(".py", "").replace(".rs", ""),
            "file":          file_info["filename"],
            "repo":          repo_name,
            "language":      cfg["language"],
            "num_lines":     len(file_info["code"].splitlines()),
        })

    print(f"  [DONE] {file_info['filename']}: {len(sft_pairs)} SFT pairs, "
          f"{len(retrieval_pairs)} retrieval pairs ({len(labels)} labels)")
    return sft_pairs, retrieval_pairs


async def run_pipeline(
    repos: List[str],
    max_files_per_repo: Optional[int],
    dry_run: bool,
) -> Tuple[List[Dict], List[Dict]]:
    """Main async pipeline across all repos. Files processed sequentially
    to respect the model's single-concurrent-request limit."""
    client    = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    all_sft        = []
    all_retrieval  = []

    for repo_name in repos:
        cfg = CORPUS_CONFIGS[repo_name]
        print(f"\n{'='*60}")
        print(f"[REPO] {repo_name} ({cfg['language']}) — {cfg['corpus_dir']}")
        print(f"{'='*60}")

        files = load_corpus_files(cfg["corpus_dir"], max_files=max_files_per_repo)
        if not files:
            print(f"[WARN] No files found in {cfg['corpus_dir']}, skipping.")
            continue

        print(f"[INFO] Processing {len(files)} files from {repo_name}...")

        repo_sft = []
        repo_ret = []

        # Process files one at a time (model allows only 1 concurrent request)
        for i, f in enumerate(files):
            print(f"[{i+1}/{len(files)}] {repo_name}/{f['filename']}")
            sft_pairs, ret_pairs = await process_file(
                client, f, cfg, repo_name, semaphore, dry_run=dry_run
            )
            repo_sft.extend(sft_pairs)
            repo_ret.extend(ret_pairs)

        all_sft.extend(repo_sft)
        all_retrieval.extend(repo_ret)

        print(f"[INFO] {repo_name}: {len(repo_sft)} SFT pairs, "
              f"{len(repo_ret)} retrieval pairs")

    return all_sft, all_retrieval


# ── Save Outputs ──────────────────────────────────────────────────────────────

def save_outputs(
    sft_pairs: List[Dict],
    retrieval_pairs: List[Dict],
    repos: List[str],
    max_files_per_repo: Optional[int],
    dry_run: bool,
) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Deduplicate by hash
    def dedup(pairs: List[Dict], key: str) -> List[Dict]:
        seen = set()
        out  = []
        for p in pairs:
            h = pair_hash(p[key], p.get("source_file", p.get("file", "")))
            if h not in seen:
                seen.add(h)
                out.append(p)
        return out

    sft_pairs       = dedup(sft_pairs, "instruction")
    retrieval_pairs = dedup(retrieval_pairs, "query")

    # Shuffle + split SFT
    random.shuffle(sft_pairs)
    sft_split = int(len(sft_pairs) * (1 - SFT_TEST_RATIO))
    sft_train = sft_pairs[:sft_split]
    sft_test  = sft_pairs[sft_split:]

    # Shuffle + split retrieval
    random.shuffle(retrieval_pairs)
    ret_split = int(len(retrieval_pairs) * (1 - RETRIEVAL_TEST_RATIO))
    ret_train = retrieval_pairs[:ret_split]
    ret_test  = retrieval_pairs[ret_split:]

    if not dry_run:
        for path, data in [
            (SFT_ALL_OUT,        sft_pairs),
            (SFT_TRAIN_OUT,      sft_train),
            (SFT_TEST_OUT,       sft_test),
            (RETRIEVAL_ALL_OUT,  retrieval_pairs),
            (RETRIEVAL_TRAIN_OUT,ret_train),
            (RETRIEVAL_TEST_OUT, ret_test),
        ]:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[SAVED] {path} ({len(data)} records)")

        # Data cards
        sft_card = {
            "name": "AI-Mined SFT Instruction–Response Pairs",
            "version": "1.0",
            "description": (
                "Instruction–response pairs generated by an LLM mining hard developer questions "
                "from real production code (hyperswitch Rust + verl Python). "
                "Covers high-level architecture questions, low-level implementation questions, "
                "and mermaid diagram generation tasks."
            ),
            "model": MODEL,
            "repos": repos,
            "total_pairs": len(sft_pairs),
            "train_pairs": len(sft_train),
            "test_pairs":  len(sft_test),
            "categories": {
                "high_level": sum(1 for p in sft_pairs if p["category"] == "high_level"),
                "low_level":  sum(1 for p in sft_pairs if p["category"] == "low_level"),
                "mermaid":    sum(1 for p in sft_pairs if p["category"] == "mermaid"),
            },
            "by_repo": {
                r: sum(1 for p in sft_pairs if p.get("repo") == r) for r in repos
            },
            "generation_method": "LLM question mining + LLM answering with code context",
            "prompt_strategy": {
                "mining": f"{HIGH_LEVEL_Q} high-level + {LOW_LEVEL_Q} low-level + {MERMAID_Q} mermaid per file",
                "answering": "Code context + question → LLM answer",
                "temperature_mine": 0.7,
                "temperature_answer": 0.3,
            },
            "max_file_chars": MAX_FILE_CHARS,
            "random_seed": RANDOM_SEED,
        }
        with open(SFT_CARD_OUT, "w") as f:
            json.dump(sft_card, f, indent=2)
        print(f"[SAVED] {SFT_CARD_OUT}")

        ret_card = {
            "name": "AI-Mined Text–Code Retrieval Pairs",
            "version": "1.0",
            "description": (
                "Natural language query–code pairs where queries are hard developer questions "
                "mined by an LLM from real production code. "
                "Queries span architecture, implementation, and diagram tasks."
            ),
            "model": MODEL,
            "repos": repos,
            "total_pairs": len(retrieval_pairs),
            "train_pairs": len(ret_train),
            "test_pairs":  len(ret_test),
            "by_repo": {
                r: sum(1 for p in retrieval_pairs if p.get("repo") == r) for r in repos
            },
            "generation_method": "LLM question mining; code snippet = file truncated to 2000 chars",
            "random_seed": RANDOM_SEED,
        }
        with open(RETRIEVAL_CARD_OUT, "w") as f:
            json.dump(ret_card, f, indent=2)
        print(f"[SAVED] {RETRIEVAL_CARD_OUT}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  SFT pairs total   : {len(sft_pairs)}")
    print(f"    train / test     : {len(sft_train)} / {len(sft_test)}")
    cat_counts = {}
    for p in sft_pairs:
        cat_counts[p["category"]] = cat_counts.get(p["category"], 0) + 1
    for cat, cnt in sorted(cat_counts.items()):
        print(f"    [{cat}]: {cnt}")
    print(f"  Retrieval pairs   : {len(retrieval_pairs)}")
    print(f"    train / test     : {len(ret_train)} / {len(ret_test)}")
    if dry_run:
        print("\n  [DRY RUN] No files written.")
    print(f"{'='*60}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI-powered SFT & retrieval data generation")
    p.add_argument(
        "--repo",
        choices=["both", "hyperswitch", "verl"],
        default="both",
        help="Which corpus to process (default: both)",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Max files per repo (default: all files)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Mine questions only — do not call answer API, do not write files",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed (default: 42)",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=MAX_CONCURRENT,
        help=f"Max concurrent LLM calls (default: {MAX_CONCURRENT})",
    )
    p.add_argument(
        "--commits",
        action="store_true",
        help="Also mine SFT pairs from commit data in commits-training/",
    )
    p.add_argument(
        "--commit-files",
        nargs="+",
        default=[
            "commits-training/sample_data.json",
            "commits-training/juspay_sample.json",
        ],
        help="Commit JSON files to mine from (default: sample_data.json + juspay_sample.json)",
    )
    p.add_argument(
        "--max-commits",
        type=int,
        default=None,
        help="Max commit records to process (default: all)",
    )
    return p.parse_args()


async def main_async() -> None:
    args = parse_args()
    random.seed(args.seed)

    global MAX_CONCURRENT
    MAX_CONCURRENT = args.concurrency

    if not API_KEY:
        print("[ERROR] No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY in .env")
        sys.exit(1)

    repos = ["hyperswitch", "verl"] if args.repo == "both" else [args.repo]

    print(f"[CONFIG] Model      : {MODEL}")
    print(f"[CONFIG] API base   : {API_BASE}")
    print(f"[CONFIG] Repos      : {repos}")
    print(f"[CONFIG] Max files  : {args.max_files or 'all'}")
    print(f"[CONFIG] Concurrency: {MAX_CONCURRENT}")
    print(f"[CONFIG] Dry run    : {args.dry_run}")
    if args.commits:
        print(f"[CONFIG] Commits    : {args.commit_files}")

    t0 = time.time()
    sft_pairs, retrieval_pairs = await run_pipeline(
        repos=repos,
        max_files_per_repo=args.max_files,
        dry_run=args.dry_run,
    )

    # Optionally mine from commits
    if args.commits:
        commit_sft = await run_commit_pipeline(
            commit_files=args.commit_files,
            max_commits=args.max_commits,
            dry_run=args.dry_run,
        )
        sft_pairs.extend(commit_sft)

    elapsed = time.time() - t0
    print(f"\n[INFO] Pipeline completed in {elapsed:.1f}s")

    save_outputs(sft_pairs, retrieval_pairs, repos, args.max_files, args.dry_run)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
