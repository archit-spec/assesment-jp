"""
Code-to-Doc Embedding Dataset Generator
========================================
Generates high-quality (code, document) pairs for training code embedding models.

For each code file / chunk the LLM produces:
  1. A rich natural-language DOCUMENTATION (what the code does, key concepts, usage)
  2. 3-5 QUERY VARIANTS a developer might type to find this code
  3. A SHORT LABEL (3-8 words, for BM25 / dense retrieval)

Output schema (one JSON object per line — HuggingFace JSONL):
  {
    "anchor":       "<code snippet>",
    "positive":     "<generated documentation>",
    "queries":      ["query 1", "query 2", ...],
    "label":        "short semantic label",
    "repo":         "volcengine/verl",
    "language":     "Python",
    "filename":     "verl__protocol.py",
    "split":        "train" | "test"
  }

Usage:
  python generate_embedding_data.py                        # both repos, all files
  python generate_embedding_data.py --max-files 20        # quick test
  python generate_embedding_data.py --repo verl           # single repo
  python generate_embedding_data.py --upload --hf-repo YOUR_HF_REPO_ID
  python generate_embedding_data.py --dry-run             # show prompts only

HuggingFace upload:
  Set HF_TOKEN in .env (or export HF_TOKEN=...) then pass --upload --hf-repo <repo_id>
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
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")

# ── API Providers ─────────────────────────────────────────────────────────────

@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str
    model: str
    concurrency: int
    semaphore: asyncio.Semaphore = field(init=False)

    def __post_init__(self):
        self.semaphore = asyncio.Semaphore(self.concurrency)


def load_providers(force_provider: Optional[str] = None, base_concurrency: int = 4) -> List[Provider]:
    providers = []

    # OpenRouter
    or_key = os.getenv("OPENROUTER_API_KEY")
    if or_key:
        providers.append(Provider(
            name="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key=or_key,
            model="z-ai/glm-4.7-flash",
            concurrency=base_concurrency,  # OpenRouter handles parallel well
        ))

    # Modal / OpenAI-compatible
    modal_key = os.getenv("OPENAI_API_KEY")
    modal_base = os.getenv("OPENAI_BASE_URL")
    if modal_key and modal_base:
        providers.append(Provider(
            name="modal",
            base_url=modal_base,
            api_key=modal_key,
            model=os.getenv("OPENAI_MODEL", "zai-org/GLM-5-FP8"),
            concurrency=2,  # Modal usually stricter/lower limits
        ))

    if force_provider and force_provider != "auto":
        providers = [p for p in providers if p.name == force_provider]

    if not providers:
        print("[ERROR] No API providers found. Check .env")
        sys.exit(1)
        
    print(f"[CONFIG] Active providers: {', '.join(p.name for p in providers)}")
    return providers

# ── Corpus configs ────────────────────────────────────────────────────────────
CORPUS_CONFIGS = {
    "hyperswitch": {
        "corpus_dir": "data/code_corpus_hyperswitch",
        "language": "Rust",
        "repo": "juspay/hyperswitch",
        "description": "a payment orchestration platform",
        "lang_lower": "rust",
    },
    "verl": {
        "corpus_dir": "data/code_corpus_verl",
        "language": "Python",
        "repo": "volcengine/verl",
        "description": "a reinforcement learning from human feedback (RLHF) training framework",
        "lang_lower": "python",
    },
}

# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUT_DIR       = "data/embedding_dataset"
TRAIN_JSONL      = f"{OUTPUT_DIR}/train.jsonl"
TEST_JSONL       = f"{OUTPUT_DIR}/test.jsonl"
ALL_JSONL        = f"{OUTPUT_DIR}/all.jsonl"
DATASET_CARD     = f"{OUTPUT_DIR}/README.md"

MAX_FILE_CHARS   = 5000
MAX_CONCURRENT   = 4
RANDOM_SEED      = 42
TEST_RATIO       = 0.15
QUERIES_PER_FILE = 4   # number of query variants to generate per file


# ── Prompts ───────────────────────────────────────────────────────────────────

DOC_SYSTEM = """\
You are a technical writer and senior software engineer.
Your job is to write clear, accurate documentation for code that will be used
to train a code search / embedding model.
Always return valid JSON — no markdown fences, no extra text."""

DOC_USER_TMPL = """\
Here is a {language} file from {repo} ({description}):

File: {filename}
```{lang_lower}
{code}
```

Produce the following for this file. Be concise — your entire response must fit in 512 tokens.

1. DOCUMENTATION (80-120 words max): A tight natural-language description covering:
   - What this module/file does and its purpose in the system
   - Key classes, functions, or data structures defined
   - How a developer would typically use or interact with this code

2. QUERIES ({n_queries} items): Short natural-language queries a developer might type
   into a code search engine to find this file. Vary the style:
   - High-level purpose ("how does X work")
   - Specific function or class
   - Task-phrased ("how to implement Y")
   - Domain-specific terminology

3. LABEL: A single short phrase (3-8 words) describing what this code does.

Return ONLY this JSON (no markdown fences, stay within 512 tokens total):
{{
  "documentation": "...",
  "queries": ["query 1", "query 2", "query 3", "query 4"],
  "label": "short label here"
}}"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def truncate_code(code: str, max_chars: int = MAX_FILE_CHARS) -> str:
    if len(code) <= max_chars:
        return code
    head = int(max_chars * 0.67)
    tail = max_chars - head
    return code[:head] + f"\n\n... [truncated {len(code)-max_chars} chars] ...\n\n" + code[-tail:]


def pair_hash(anchor: str, filename: str) -> str:
    return hashlib.md5(f"{anchor[:200]}|{filename}".encode()).hexdigest()


def extract_json_block(text: str) -> Optional[Dict]:
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end+1])
    except json.JSONDecodeError:
        return None


def message_content_to_text(content: Any) -> str:
    """Normalize provider-specific content payloads to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text_val = item.get("text") or item.get("content")
                if isinstance(text_val, str):
                    parts.append(text_val)
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


def is_quality_code_file(code: str, language: str) -> Tuple[bool, str]:
    """Return (passes, reason) — filter out config/stub/trivial files."""
    lines = code.splitlines()
    total = len(lines)

    # Too short
    if total < 30:
        return False, f"too short ({total} lines)"

    # Count meaningful code lines (non-blank, non-comment)
    if language == "Rust":
        comment_prefixes = ("//", "/*", "*", "#!")
        code_keywords = ("fn ", "impl ", "struct ", "enum ", "trait ", "pub ", "use ", "mod ", "let ", "match ")
    else:  # Python
        comment_prefixes = ("#",)
        code_keywords = ("def ", "class ", "import ", "from ", "return ", "if ", "for ", "with ", "async ", "await ")

    code_lines = [l for l in lines if l.strip() and not l.strip().startswith(comment_prefixes)]
    comment_lines = [l for l in lines if l.strip() and l.strip().startswith(comment_prefixes)]

    # Mostly comments / docstrings
    if len(code_lines) < 15:
        return False, f"too few code lines ({len(code_lines)})"

    comment_ratio = len(comment_lines) / max(total, 1)
    if comment_ratio > 0.7:
        return False, f"mostly comments ({comment_ratio:.0%})"

    # Must have at least one real definition
    has_definition = any(
        any(kw in l for kw in code_keywords)
        for l in code_lines
    )
    if not has_definition:
        return False, "no function/struct/class definitions found"

    # Skip files that look like pure config / generated code
    lower = code.lower()
    config_signals = ["#[allow(dead_code)]", "#[derive(", "serde_json::json!(", "toml", ".yaml", ".json"]
    config_count = sum(1 for sig in config_signals if sig in lower)
    # Only skip if it's MOSTLY config-like with very little logic
    if config_count >= 3 and len(code_lines) < 40:
        return False, f"likely config/generated file ({config_count} config signals, {len(code_lines)} code lines)"

    return True, "ok"


def load_corpus_files(
    corpus_dir: str,
    language: str = "Python",
    max_files: Optional[int] = None,
    skip_sandbox_files: bool = True,
) -> List[Dict]:
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        print(f"[WARN] Corpus dir not found: {corpus_dir}")
        return []
    files = []
    skipped = 0
    skipped_sandbox = 0
    for p in sorted(corpus_path.iterdir()):
        if p.is_file() and not p.name.startswith("."):
            if skip_sandbox_files and "sandbox" in p.name.lower():
                skipped_sandbox += 1
                continue
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                if not content.strip():
                    continue
                passes, reason = is_quality_code_file(content, language)
                if not passes:
                    skipped += 1
                    continue
                files.append({"filename": p.name, "code": content, "path": str(p)})
            except Exception as e:
                print(f"[WARN] Could not read {p}: {e}")
    if skipped:
        print(f"[FILTER] Skipped {skipped} low-quality files, kept {len(files)}")
    if skipped_sandbox:
        print(f"[FILTER] Skipped {skipped_sandbox} sandbox files, kept {len(files)}")
    random.shuffle(files)
    if max_files:
        files = files[:max_files]
    return files


# ── LLM call ─────────────────────────────────────────────────────────────────

async def generate_code_doc(
    client: AsyncOpenAI,
    provider: Provider,
    file_info: Dict,
    cfg: Dict,
    retries: int = 2,
) -> Optional[Dict]:
    """Call LLM to generate documentation + queries + label for a code file."""
    code = truncate_code(file_info["code"])
    user_msg = DOC_USER_TMPL.format(
        language=cfg["language"],
        lang_lower=cfg["lang_lower"],
        repo=cfg["repo"],
        description=cfg["description"],
        filename=file_info["filename"],
        code=code,
        n_queries=QUERIES_PER_FILE,
    )

    async with provider.semaphore:
        for attempt in range(retries + 1):
            try:
                # Re-instantiate client if base_url changes per request (not efficient but safe)
                # or just use the passed client if we manage clients per provider.
                # Actually, standard AsyncOpenAI client is tied to one base_url.
                # So process_file will need to pick a provider and create/use its client.
                
                req = {
                    "model": provider.model,
                    "messages": [
                        {"role": "system", "content": DOC_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1024,
                }
                try:
                    resp = await client.chat.completions.create(
                        **req,
                        response_format={"type": "json_object"},
                        extra_body={"reasoning": {"enabled": False}},
                    )
                except Exception as param_err:
                    err_text = str(param_err).lower()
                    if "response_format" in err_text or "reasoning" in err_text:
                        resp = await client.chat.completions.create(**req)
                    else:
                        raise
                msg = resp.choices[0].message
                raw = message_content_to_text(msg.content)
                if not raw:
                    raw = message_content_to_text(getattr(msg, "reasoning", None))
                if not raw:
                    raw = message_content_to_text(getattr(msg, "refusal", None))
                if not raw:
                    try:
                        resp_preview = json.dumps(resp.model_dump(), ensure_ascii=False)[:400]
                    except Exception:
                        resp_preview = "<could not serialize response>"
                    print(f"  [WARN] [{provider.name}] Empty response content for {file_info['filename']}")
                    print(f"         RESP: {resp_preview}")
                parsed = extract_json_block(raw)
                if (
                    parsed
                    and isinstance(parsed.get("documentation"), str)
                    and isinstance(parsed.get("queries"), list)
                    and isinstance(parsed.get("label"), str)
                    and len(parsed["documentation"]) > 50
                ):
                    return parsed
                print(f"  [WARN] [{provider.name}] Bad JSON for {file_info['filename']}, attempt {attempt+1}")
                print(f"         RAW: {raw[:200]} ... {raw[-200:] if len(raw)>200 else ''}")
            except Exception as e:
                print(f"  [ERROR] [{provider.name}] {file_info['filename']} attempt {attempt+1}: {e}")
                if attempt < retries:
                    delay = 2 * (2 ** attempt)
                    await asyncio.sleep(delay)
    return None


# ── Pipeline ──────────────────────────────────────────────────────────────────

async def process_file(
    provider: Provider,
    file_info: Dict,
    cfg: Dict,
    repo_name: str,
    dry_run: bool = False,
) -> Optional[Dict]:
    """Process one file → one embedding record."""
    if dry_run:
        print(f"  [DRY RUN] [{provider.name}] Would generate doc+queries for {file_info['filename']}")
        return None

    # Create a client for this specific provider (lightweight)
    client = AsyncOpenAI(api_key=provider.api_key, base_url=provider.base_url, timeout=60.0)
    
    try:
        result = await generate_code_doc(client, provider, file_info, cfg)
        await client.close()
    except Exception:
        await client.close()
        return None

    if not result:
        print(f"  [SKIP] {file_info['filename']}: could not generate doc")
        return None

    code_snippet = truncate_code(file_info["code"], max_chars=3000)

    # Clean up queries
    queries = [q.strip() for q in result["queries"] if isinstance(q, str) and len(q.strip()) > 5]
    queries = queries[:QUERIES_PER_FILE]

    record = {
        "anchor":    code_snippet,
        "positive":  result["documentation"].strip(),
        "queries":   queries,
        "label":     result["label"].strip()[:100],
        "repo":      cfg["repo"],
        "language":  cfg["language"],
        "filename":  file_info["filename"],
        "num_lines": len(file_info["code"].splitlines()),
    }
    print(f"  [DONE] {file_info['filename']}: {len(queries)} queries, doc={len(record['positive'])} chars")
    return record


async def run_pipeline(
    repos: List[str],
    max_files_per_repo: Optional[int],
    dry_run: bool,
    active_providers: List[Provider],
    skip_sandbox_files: bool = True,
) -> List[Dict]:
    all_records: List[Dict] = []

    for repo_name in repos:
        cfg = CORPUS_CONFIGS[repo_name]
        print(f"\n{'='*60}")
        print(f"[REPO] {repo_name} ({cfg['language']}) — {cfg['corpus_dir']}")
        print(f"{'='*60}")

        files = load_corpus_files(
            cfg["corpus_dir"],
            language=cfg["language"],
            max_files=max_files_per_repo,
            skip_sandbox_files=skip_sandbox_files,
        )
        if not files:
            print(f"[WARN] No files found in {cfg['corpus_dir']}, skipping.")
            continue

        print(f"[INFO] Processing {len(files)} files from {repo_name} using {len(active_providers)} providers...")

        # Distribute files active providers in round-robin fashion
        tasks = []
        for i, f in enumerate(files):
            provider = active_providers[i % len(active_providers)]
            tasks.append(process_file(provider, f, cfg, repo_name, dry_run=dry_run))

        results = await asyncio.gather(*tasks)
        repo_records = [r for r in results if r is not None]
        all_records.extend(repo_records)

        print(f"[INFO] {repo_name}: {len(repo_records)} records")

    return all_records


# ── Save + Upload ─────────────────────────────────────────────────────────────

def save_outputs(records: List[Dict], repos: List[str], dry_run: bool, hf_repo: Optional[str] = None) -> None:
    if dry_run:
        print(f"\n[DRY RUN] Would save {len(records)} records. No files written.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Deduplicate
    seen = set()
    deduped = []
    for r in records:
        h = pair_hash(r["anchor"], r["filename"])
        if h not in seen:
            seen.add(h)
            deduped.append(r)
    records = deduped

    # Shuffle + split
    random.shuffle(records)
    split_idx = int(len(records) * (1 - TEST_RATIO))
    train_records = [{"split": "train", **r} for r in records[:split_idx]]
    test_records  = [{"split": "test",  **r} for r in records[split_idx:]]
    all_records   = train_records + test_records

    # Write JSONL
    for path, data in [
        (ALL_JSONL,   all_records),
        (TRAIN_JSONL, train_records),
        (TEST_JSONL,  test_records),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[SAVED] {path} ({len(data)} records)")

    # Dataset card (README.md for HuggingFace)
    by_repo = {}
    for r in records:
        by_repo[r["repo"]] = by_repo.get(r["repo"], 0) + 1

    card = f"""---
language:
- en
license: mit
task_categories:
- text-retrieval
- feature-extraction
task_ids:
- document-retrieval
tags:
- code
- embedding
- code-search
- retrieval
- retrieval
{chr(10).join(f"- {r}" for r in repos)}
size_categories:
- {"1K<n<10K" if len(records) >= 1000 else "n<1K"}
---

# Code-to-Doc Embedding Dataset

AI-generated code documentation pairs for training code embedding / retrieval models.

## Dataset Description

Each record contains a **code anchor** (real production code) paired with:
- **positive**: A rich natural-language documentation of what the code does
- **queries**: {QUERIES_PER_FILE} natural-language search queries a developer might use to find this code
- **label**: A short semantic label (3-8 words)

This dataset is designed for training **bi-encoder** embedding models (e.g., with InfoNCE / contrastive loss)
where `anchor` = code, `positive` = documentation, and `queries` can serve as additional positives.

## Sources

| Repo | Language | Records |
|------|----------|---------|
{"".join(f"| {repo} | {'Rust' if 'hyperswitch' in repo else 'Python'} | {cnt} |" + chr(10) for repo, cnt in by_repo.items())}

**Total**: {len(records)} records ({len(train_records)} train / {len(test_records)} test)

## Schema

```json
{{
  "anchor":    "<code snippet, up to 3000 chars>",
  "positive":  "<150-300 word natural language documentation>",
  "queries":   ["query 1", "query 2", "query 3", "query 4"],
  "label":     "short semantic label",
  "repo":      "owner/repo",
  "language":  "Python | Rust",
  "filename":  "source_filename.py",
  "num_lines": 42,
  "split":     "train | test"
}}
```

## Generation

- **Model**: Provider-specific (`qwen/qwen3.5` on OpenRouter, `GLM-5` on Modal)
- **Method**: LLM-generated documentation + query variants per file
- **Temperature**: 0.3 (documentation), deterministic
- **Code truncation**: {MAX_FILE_CHARS} chars max input, {3000} chars max anchor

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{hf_repo or 'YOUR_HF_REPO'}")

# For contrastive training (anchor=code, positive=doc)
for example in ds["train"]:
    code = example["anchor"]
    doc  = example["positive"]
    queries = example["queries"]  # additional positives

# For retrieval evaluation
for example in ds["test"]:
    query = example["queries"][0]
    code  = example["anchor"]
```

## Training Tips

- Use `anchor` as the **code encoder** input and `positive` as the **text encoder** input
- `queries` can be used as **hard positives** or for query augmentation
- For hard negatives: sample other records from the same `language` or `repo`
"""

    with open(DATASET_CARD, "w", encoding="utf-8") as f:
        f.write(card)
    print(f"[SAVED] {DATASET_CARD}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Total records : {len(records)}")
    print(f"  Train / Test  : {len(train_records)} / {len(test_records)}")
    for repo, cnt in by_repo.items():
        print(f"  {repo}: {cnt}")
    print(f"  Output dir    : {OUTPUT_DIR}/")
    print(f"{'='*60}")

    # HuggingFace upload
    if hf_repo:
        upload_to_hf(hf_repo)


def upload_to_hf(hf_repo: str) -> None:
    """Upload the dataset to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
        from datasets import load_dataset, DatasetDict
    except ImportError:
        print("[ERROR] Install huggingface_hub and datasets: pip install huggingface-hub datasets")
        return

    token = HF_TOKEN or os.getenv("HF_TOKEN")
    if not token:
        print("[ERROR] HF_TOKEN not set. Add it to .env or export HF_TOKEN=hf_...")
        return

    print(f"\n[HF] Uploading to {hf_repo}...")
    try:
        api = HfApi(token=token)

        # Create repo if needed
        api.create_repo(repo_id=hf_repo, repo_type="dataset", exist_ok=True)

        # Upload files
        for fname in [ALL_JSONL, TRAIN_JSONL, TEST_JSONL, DATASET_CARD]:
            if Path(fname).exists():
                api.upload_file(
                    path_or_fileobj=fname,
                    path_in_repo=Path(fname).name,
                    repo_id=hf_repo,
                    repo_type="dataset",
                    token=token,
                )
                print(f"  [HF] Uploaded {fname}")

        print(f"[HF] Done! Dataset at: https://huggingface.co/datasets/{hf_repo}")
    except Exception as e:
        print(f"[ERROR] HF upload failed: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate code-to-doc embedding dataset and optionally upload to HuggingFace"
    )
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
        help="Max files per repo (default: all)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without calling the LLM",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max concurrent LLM calls per provider (default: 4)",
    )
    p.add_argument(
        "--provider",
        choices=["auto", "openrouter", "modal"],
        default="openrouter",
        help="Which API provider to use (default: openrouter)",
    )
    p.add_argument(
        "--include-sandbox-files",
        action="store_true",
        help="Include sandbox-related source files (default: skip them)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
    )
    p.add_argument(
        "--upload",
        action="store_true",
        help="Upload to HuggingFace Hub after generation",
    )
    p.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="HuggingFace repo ID to upload to (e.g. your-username/code-embedding-dataset)",
    )
    return p.parse_args()


async def main_async() -> None:
    args = parse_args()
    random.seed(args.seed)

    active_providers = load_providers(force_provider=args.provider, base_concurrency=args.concurrency)

    repos = ["hyperswitch", "verl"] if args.repo == "both" else [args.repo]

    print(f"[CONFIG] Repos       : {repos}")
    print(f"[CONFIG] Max files   : {args.max_files or 'all'}")
    print(f"[CONFIG] Dry run     : {args.dry_run}")
    print(f"[CONFIG] Sandbox code: {'included' if args.include_sandbox_files else 'skipped'}")
    if args.upload:
        print(f"[CONFIG] HF repo     : {args.hf_repo or '(not set)'}")

    t0 = time.time()
    records = await run_pipeline(
        repos=repos,
        max_files_per_repo=args.max_files,
        dry_run=args.dry_run,
        active_providers=active_providers,
        skip_sandbox_files=not args.include_sandbox_files,
    )
    elapsed = time.time() - t0
    print(f"\n[INFO] Pipeline completed in {elapsed:.1f}s — {len(records)} records")

    hf_repo = args.hf_repo if args.upload else None
    save_outputs(records, repos, args.dry_run, hf_repo=hf_repo)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
