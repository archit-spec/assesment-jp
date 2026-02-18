"""
Optimized Code-to-Doc Embedding Dataset Generator
=================================================
Improvements aimed at Recall@10 / MRR@10 / nDCG@10:
- Function-aware chunking (default) instead of whole-file indexing
- Metadata-enriched anchors (path/module/symbol)
- Robust code quality filtering
- OpenRouter-first generation with strict JSON parsing
- Optional hard-negative mining export
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
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import httpx
from openai import AsyncOpenAI

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")
RAW_HTTP_CLIENTS: Dict[str, httpx.AsyncClient] = {}


# ── Providers ────────────────────────────────────────────────────────────────

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


def load_providers(force_provider: str, base_concurrency: int) -> List[Provider]:
    providers: List[Provider] = []

    or_key = os.getenv("OPENROUTER_API_KEY")
    if or_key:
        providers.append(
            Provider(
                name="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_key=or_key,
                model="z-ai/glm-4.7-flash",
                concurrency=base_concurrency,
            )
        )

    modal_key = os.getenv("OPENAI_API_KEY")
    modal_base = os.getenv("OPENAI_BASE_URL")
    if modal_key and modal_base:
        providers.append(
            Provider(
                name="modal",
                base_url=modal_base,
                api_key=modal_key,
                model=os.getenv("OPENAI_MODEL", "zai-org/GLM-5-FP8"),
                concurrency=2,
            )
        )

    if force_provider != "auto":
        providers = [p for p in providers if p.name == force_provider]

    if not providers:
        print("[ERROR] No API providers found. Check .env")
        sys.exit(1)

    print(f"[CONFIG] Active providers: {', '.join(p.name for p in providers)}")
    return providers


# ── Corpus config ────────────────────────────────────────────────────────────

CORPUS_CONFIGS = {
    "hyperswitch": {
        "corpus_dir": "data/code_corpus_hyperswitch",
        "language": "Rust",
        "repo": "juspay/hyperswitch",
        "description": "a payment orchestration platform",
        "lang_lower": "rust",
        "repo_map": "hyperswitch/REPO_MAP.md",
    },
    "verl": {
        "corpus_dir": "data/code_corpus_verl",
        "language": "Python",
        "repo": "volcengine/verl",
        "description": "a reinforcement learning from human feedback (RLHF) training framework",
        "lang_lower": "python",
        "repo_map": "",
    },
}


# ── Constants ────────────────────────────────────────────────────────────────

MAX_FILE_CHARS = 12000
MAX_CHUNK_CHARS = 3500
MAX_REPO_MAP_CHARS = 2500
TEST_RATIO = 0.15
QUERIES_PER_CHUNK = 4
RANDOM_SEED = 42

BUCKET_PRIORITY = {
    "router_core": 1.30,
    "connectors": 1.25,
    "router_routes": 1.20,
    "router_services": 1.15,
    "storage_db": 1.10,
    "scheduler": 1.05,
    "drainer": 1.05,
    "api_domain": 1.00,
    "common_utils": 0.95,
    "other": 0.80,
    "tests": 0.30,
    "migrations": 0.20,
}


# ── Prompt ───────────────────────────────────────────────────────────────────

DOC_SYSTEM = """\
You are a senior engineer generating training data for text-to-code retrieval.
Optimize for:
- Recall@10: include precise identifiers and architecture context
- MRR@10: highlight unique behavior and distinguishing details
- nDCG@10: preserve ranking-relevant specificity

Return valid JSON only.
"""

DOC_USER_TMPL = """\
Repository: {repo} ({description})
Language: {language}
Path: {path}
Symbol: {symbol}
Unit: {unit_type}

Repo architecture hints:
{repo_map_content}

Code:
```{lang_lower}
{code}
```

Return ONLY JSON:
{{
  "documentation": "80-140 words explaining purpose, behavior, and integration context",
  "queries": [
    "4 distinct natural-language retrieval queries developers would type"
  ],
  "label": "3-8 word semantic label"
}}
"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def pair_hash(anchor: str, key: str) -> str:
    return hashlib.md5(f"{anchor[:200]}|{key}".encode()).hexdigest()


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.7)
    tail = max_chars - head
    return text[:head] + f"\n\n... [truncated {len(text) - max_chars} chars] ...\n\n" + text[-tail:]


def extract_json_block(text: str) -> Optional[Dict]:
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                txt = item.get("text") or item.get("content")
                if isinstance(txt, str):
                    parts.append(txt)
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


def response_text_from_completion(resp: Any) -> str:
    choices = getattr(resp, "choices", None)
    if isinstance(choices, (list, tuple)) and choices:
        msg = getattr(choices[0], "message", None)
        if msg is not None:
            raw = message_content_to_text(getattr(msg, "content", None))
            if raw:
                return raw
            raw = message_content_to_text(getattr(msg, "reasoning", None))
            if raw:
                return raw

    payload = None
    if hasattr(resp, "model_dump"):
        try:
            payload = resp.model_dump()
        except Exception:
            payload = None
    elif isinstance(resp, dict):
        payload = resp

    if isinstance(payload, dict):
        choices_d = payload.get("choices")
        if isinstance(choices_d, list) and choices_d:
            msg_d = choices_d[0].get("message", {}) if isinstance(choices_d[0], dict) else {}
            raw = message_content_to_text(msg_d.get("content"))
            if raw:
                return raw
            raw = message_content_to_text(msg_d.get("reasoning"))
            if raw:
                return raw
    return ""


def is_rate_limit_error(err: Exception) -> bool:
    status = getattr(err, "status_code", None)
    if status == 429:
        return True
    text = str(err).lower()
    return "429" in text or "rate-limit" in text or "rate limit" in text


async def get_raw_http_client(base_url: str) -> httpx.AsyncClient:
    client = RAW_HTTP_CLIENTS.get(base_url)
    if client is not None:
        return client
    client = httpx.AsyncClient(base_url=base_url, timeout=60.0)
    RAW_HTTP_CLIENTS[base_url] = client
    return client


async def close_raw_http_clients() -> None:
    for client in RAW_HTTP_CLIENTS.values():
        await client.aclose()
    RAW_HTTP_CLIENTS.clear()


async def create_chat_completion(
    provider: Provider,
    client: Optional[AsyncOpenAI],
    req: Dict[str, Any],
) -> Any:
    if provider.name == "openrouter":
        raw_client = await get_raw_http_client(provider.base_url)
        payload = {
            **req,
            "response_format": {"type": "json_object"},
            "reasoning": {"enabled": False},
        }
        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
        }
        resp = await raw_client.post("/chat/completions", json=payload, headers=headers)
        data: Any
        try:
            data = resp.json()
        except Exception:
            data = {"raw_text": resp.text}

        if resp.status_code >= 400:
            err_obj = data.get("error") if isinstance(data, dict) else None
            if isinstance(err_obj, dict):
                raise RuntimeError(f"Error code: {resp.status_code} - {err_obj}")
            raise RuntimeError(f"Error code: {resp.status_code} - {str(data)[:500]}")

        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            err_obj = data["error"]
            raise RuntimeError(f"Error code: {err_obj.get('code', 'unknown')} - {err_obj}")

        return data

    if client is None:
        raise RuntimeError(f"No client configured for provider={provider.name}")

    try:
        return await client.chat.completions.create(
            **req,
            response_format={"type": "json_object"},
            extra_body={"reasoning": {"enabled": False}},
        )
    except Exception as param_err:
        err_text = str(param_err).lower()
        if "response_format" in err_text or "reasoning" in err_text:
            return await client.chat.completions.create(**req)
        raise


def tokenize_for_overlap(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", text.lower())


def is_quality_code_file(code: str, language: str) -> Tuple[bool, str]:
    lines = code.splitlines()
    total = len(lines)
    if total < 30:
        return False, f"too short ({total} lines)"

    if language == "Rust":
        comment_prefixes = ("//", "/*", "*", "#!")
        code_keywords = ("fn ", "impl ", "struct ", "enum ", "trait ", "pub ", "use ", "mod ")
    else:
        comment_prefixes = ("#",)
        code_keywords = ("def ", "class ", "import ", "from ", "async ")

    code_lines = [l for l in lines if l.strip() and not l.strip().startswith(comment_prefixes)]
    comment_lines = [l for l in lines if l.strip() and l.strip().startswith(comment_prefixes)]

    if len(code_lines) < 15:
        return False, f"too few code lines ({len(code_lines)})"

    comment_ratio = len(comment_lines) / max(total, 1)
    if comment_ratio > 0.7:
        return False, f"mostly comments ({comment_ratio:.0%})"

    has_definition = any(any(k in l for k in code_keywords) for l in code_lines)
    if not has_definition:
        return False, "no clear definitions"

    return True, "ok"


def normalize_corpus_path(path: str) -> str:
    p = path.lower().replace("\\", "/")
    leaf = p.split("/")[-1]
    if "__" in leaf:
        return leaf.replace("__", "/")
    return p


def path_bucket(path: str) -> str:
    p = normalize_corpus_path(path)
    if "migration" in p:
        return "migrations"
    if "/tests/" in p or p.endswith("_test.rs") or p.endswith("_tests.rs"):
        return "tests"
    if "/connectors/" in p:
        return "connectors"
    if "/router/src/core/" in p:
        return "router_core"
    if "/router/src/routes/" in p:
        return "router_routes"
    if "/router/src/services/" in p:
        return "router_services"
    if "/scheduler/" in p:
        return "scheduler"
    if "/drainer/" in p:
        return "drainer"
    if "/storage_impl/" in p or "/router/src/db" in p or "/diesel_models/" in p:
        return "storage_db"
    if "/api_models/" in p or "/hyperswitch_domain_models/" in p:
        return "api_domain"
    if "/common_" in p or "/utils/" in p:
        return "common_utils"
    return "other"


def definition_count(code: str, language: str) -> int:
    if language == "Rust":
        return len(
            re.findall(
                r"^\s*(?:pub\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+\w+|^\s*(?:pub\s+)?(?:struct|enum|trait|impl)\s+\w+",
                code,
                flags=re.MULTILINE,
            )
        )
    return len(re.findall(r"^\s*(?:async\s+def|def|class)\s+\w+", code, flags=re.MULTILINE))


def file_signal_score(code: str, path: str, language: str) -> float:
    lines = code.splitlines()
    line_score = min(len(lines), 900) / 900.0
    defs = definition_count(code, language)
    def_score = min(defs, 35) / 35.0

    bucket = path_bucket(path)
    priority = BUCKET_PRIORITY.get(bucket, 0.8)

    # Prefer files with richer code structure while de-prioritizing noisy buckets.
    return (0.45 * line_score + 0.55 * def_score) * priority


def smart_sample_files(files: List[Dict], max_files: int, language: str) -> List[Dict]:
    if max_files <= 0 or len(files) <= max_files:
        return files

    prepared: List[Dict[str, Any]] = []
    for f in files:
        bucket = path_bucket(f["path"])
        score = file_signal_score(f["code"], f["path"], language)
        prepared.append({**f, "_bucket": bucket, "_score": score})

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in prepared:
        buckets[f["_bucket"]].append(f)
    for bucket in buckets:
        buckets[bucket].sort(key=lambda x: (x["_score"], x["path"]), reverse=True)

    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    bucket_counts: Dict[str, int] = defaultdict(int)

    # Seed the sample with one file from each strong bucket for architecture coverage.
    strong_buckets = [
        b for b, priority in sorted(BUCKET_PRIORITY.items(), key=lambda x: x[1], reverse=True) if priority >= 1.0
    ]
    for bucket in strong_buckets:
        if len(selected) >= max_files:
            break
        pool = buckets.get(bucket, [])
        if not pool:
            continue
        pick = pool[0]
        selected.append(pick)
        selected_ids.add(pick["path"])
        bucket_counts[bucket] += 1

    remaining = [f for f in prepared if f["path"] not in selected_ids]
    while len(selected) < max_files and remaining:
        best_idx = 0
        best_value = -1.0
        for i, f in enumerate(remaining):
            bucket = f["_bucket"]
            diversity_penalty = 1.0 + 0.55 * bucket_counts[bucket]
            value = f["_score"] / diversity_penalty
            if value > best_value:
                best_value = value
                best_idx = i

        pick = remaining.pop(best_idx)
        selected.append(pick)
        selected_ids.add(pick["path"])
        bucket_counts[pick["_bucket"]] += 1

    out = []
    for f in selected:
        clean = {k: v for k, v in f.items() if not k.startswith("_")}
        out.append(clean)

    mix = ", ".join(
        f"{bucket}:{count}"
        for bucket, count in sorted(bucket_counts.items(), key=lambda x: (-x[1], x[0]))
        if count > 0
    )
    print(f"[SAMPLE] Smart sample selected {len(out)}/{len(files)} files across {len(bucket_counts)} buckets")
    if mix:
        print(f"[SAMPLE] Bucket mix: {mix}")
    return out


def rust_symbol(line: str) -> str:
    m = re.search(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)", line)
    if m:
        return m.group(1)
    m = re.search(r"\b(struct|enum|trait|impl)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
    if m:
        return m.group(2)
    return "unknown"


def py_symbol(line: str) -> str:
    m = re.search(r"\b(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
    return m.group(1) if m else "unknown"


def find_chunk_starts(lines: List[str], language: str) -> List[int]:
    starts: List[int] = []
    for i, line in enumerate(lines):
        if language == "Rust":
            if re.match(r"^\s*(pub\s+)?(async\s+)?(unsafe\s+)?fn\s+[A-Za-z_]", line):
                starts.append(i)
            elif re.match(r"^\s*(pub\s+)?(struct|enum|trait|impl)\s+[A-Za-z_]", line):
                starts.append(i)
        else:
            if re.match(r"^\s*(async\s+def|def|class)\s+[A-Za-z_]", line):
                starts.append(i)
    return starts


def split_into_units(file_info: Dict, cfg: Dict, unit_type: str, max_chunks_per_file: int) -> List[Dict]:
    code = file_info["code"]
    if unit_type == "file":
        return [
            {
                "filename": file_info["filename"],
                "path": file_info["path"],
                "symbol": "file",
                "unit_type": "file",
                "code": truncate_text(code, MAX_CHUNK_CHARS),
                "num_lines": len(code.splitlines()),
            }
        ]

    lines = code.splitlines()
    starts = find_chunk_starts(lines, cfg["language"])
    if not starts:
        return [
            {
                "filename": file_info["filename"],
                "path": file_info["path"],
                "symbol": "file",
                "unit_type": "file-fallback",
                "code": truncate_text(code, MAX_CHUNK_CHARS),
                "num_lines": len(lines),
            }
        ]

    starts = sorted(starts)
    units: List[Dict] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        chunk_lines = lines[start:end]
        if len(chunk_lines) < 8:
            continue
        header_line = lines[start]
        symbol = rust_symbol(header_line) if cfg["language"] == "Rust" else py_symbol(header_line)
        chunk = "\n".join(chunk_lines)
        chunk = truncate_text(chunk, MAX_CHUNK_CHARS)
        units.append(
            {
                "filename": file_info["filename"],
                "path": file_info["path"],
                "symbol": symbol,
                "unit_type": "function",
                "code": chunk,
                "num_lines": len(chunk_lines),
            }
        )
        if len(units) >= max_chunks_per_file:
            break

    if not units:
        return [
            {
                "filename": file_info["filename"],
                "path": file_info["path"],
                "symbol": "file",
                "unit_type": "file-fallback",
                "code": truncate_text(code, MAX_CHUNK_CHARS),
                "num_lines": len(lines),
            }
        ]
    return units


def metadata_prefix(path: str, symbol: str, language: str) -> str:
    prefix = "//" if language == "Rust" else "#"
    module = path.replace("/", "::")
    return f"{prefix} PATH: {path}\n{prefix} MODULE: {module}\n{prefix} SYMBOL: {symbol}\n"


def load_repo_map(path: str) -> str:
    if path and Path(path).exists():
        return truncate_text(Path(path).read_text(encoding="utf-8", errors="ignore"), MAX_REPO_MAP_CHARS)
    return "No architecture map available."


def load_corpus_files(
    corpus_dir: str,
    language: str,
    max_files: Optional[int],
    skip_sandbox_files: bool,
    smart_sample: bool,
) -> List[Dict]:
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        print(f"[WARN] Corpus dir not found: {corpus_dir}")
        return []

    files: List[Dict] = []
    skipped_quality = 0
    skipped_sandbox = 0

    for p in sorted(corpus_path.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        if skip_sandbox_files and "sandbox" in p.name.lower():
            skipped_sandbox += 1
            continue

        content = p.read_text(encoding="utf-8", errors="ignore")
        if not content.strip():
            continue

        ok, _ = is_quality_code_file(content, language)
        if not ok:
            skipped_quality += 1
            continue

        files.append({"filename": p.name, "path": str(p), "code": content})

    if max_files:
        if smart_sample:
            files = smart_sample_files(files, max_files=max_files, language=language)
        else:
            random.shuffle(files)
            files = files[:max_files]
    else:
        random.shuffle(files)

    if skipped_quality:
        print(f"[FILTER] Skipped {skipped_quality} low-quality files")
    if skipped_sandbox:
        print(f"[FILTER] Skipped {skipped_sandbox} sandbox files")

    return files


# ── LLM generation ───────────────────────────────────────────────────────────

async def generate_doc_for_unit(
    client: Optional[AsyncOpenAI],
    provider: Provider,
    unit: Dict,
    cfg: Dict,
    repo_map_content: str,
    retries: int = 2,
) -> Optional[Dict]:
    user_msg = DOC_USER_TMPL.format(
        language=cfg["language"],
        repo=cfg["repo"],
        description=cfg["description"],
        path=unit["path"],
        symbol=unit["symbol"],
        unit_type=unit["unit_type"],
        repo_map_content=repo_map_content,
        lang_lower=cfg["lang_lower"],
        code=unit["code"],
    )

    async with provider.semaphore:
        for attempt in range(retries + 1):
            try:
                req = {
                    "model": provider.model,
                    "messages": [
                        {"role": "system", "content": DOC_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 900,
                }
                resp = await create_chat_completion(provider, client, req)

                raw = response_text_from_completion(resp)
                if not raw:
                    print(
                        f"  [WARN] [{provider.name}] Empty response for "
                        f"{unit['filename']}::{unit['symbol']} attempt {attempt + 1}"
                    )
                    if attempt < retries:
                        await asyncio.sleep(1.0 + random.uniform(0.0, 1.0))
                    continue
                parsed = extract_json_block(raw)

                if (
                    parsed
                    and isinstance(parsed.get("documentation"), str)
                    and isinstance(parsed.get("queries"), list)
                    and isinstance(parsed.get("label"), str)
                    and len(parsed["documentation"].strip()) >= 40
                ):
                    return parsed

                print(f"  [WARN] [{provider.name}] Bad JSON for {unit['filename']}::{unit['symbol']} attempt {attempt + 1}")
            except Exception as e:
                print(f"  [ERROR] [{provider.name}] {unit['filename']}::{unit['symbol']} attempt {attempt + 1}: {e}")
                if attempt < retries:
                    if is_rate_limit_error(e):
                        sleep_s = min(30.0, 6.0 * (2**attempt)) + random.uniform(0.0, 2.5)
                    else:
                        sleep_s = 2.0 * (2**attempt)
                    await asyncio.sleep(sleep_s)
    return None


async def process_unit(
    provider: Provider,
    client: Optional[AsyncOpenAI],
    unit: Dict,
    cfg: Dict,
    repo_map_content: str,
    dry_run: bool,
) -> Optional[Dict]:
    if dry_run:
        print(f"  [DRY RUN] [{provider.name}] {unit['filename']}::{unit['symbol']}")
        return None

    if client is None:
        if provider.name != "openrouter":
            return None
    result = await generate_doc_for_unit(client, provider, unit, cfg, repo_map_content)

    if not result:
        return None

    queries = [q.strip() for q in result["queries"] if isinstance(q, str) and len(q.strip()) > 5][:QUERIES_PER_CHUNK]
    if not queries:
        return None

    anchor = metadata_prefix(unit["path"], unit["symbol"], cfg["language"]) + unit["code"]
    return {
        "anchor": anchor,
        "positive": result["documentation"].strip(),
        "queries": queries,
        "label": result["label"].strip()[:100],
        "repo": cfg["repo"],
        "language": cfg["language"],
        "filename": unit["filename"],
        "path": unit["path"],
        "symbol": unit["symbol"],
        "unit_type": unit["unit_type"],
        "num_lines": unit["num_lines"],
    }


async def run_pipeline(
    repos: List[str],
    max_files_per_repo: Optional[int],
    unit: str,
    max_chunks_per_file: int,
    dry_run: bool,
    providers: List[Provider],
    skip_sandbox_files: bool,
    smart_sample: bool,
    progress_every: int,
    max_units: int,
    task_batch_size: int,
) -> List[Dict]:
    records: List[Dict] = []
    clients: Dict[str, AsyncOpenAI] = {}

    if not dry_run:
        clients = {
            p.name: AsyncOpenAI(
                api_key=p.api_key,
                base_url=p.base_url,
                timeout=60.0,
            )
            for p in providers
        }

    try:
        for repo_name in repos:
            cfg = CORPUS_CONFIGS[repo_name]
            repo_map_content = load_repo_map(cfg.get("repo_map", ""))
            files = load_corpus_files(
                cfg["corpus_dir"],
                cfg["language"],
                max_files=max_files_per_repo,
                skip_sandbox_files=skip_sandbox_files,
                smart_sample=smart_sample,
            )
            if not files:
                print(f"[WARN] No files for {repo_name}")
                continue

            units: List[Dict] = []
            for f in files:
                units.extend(split_into_units(f, cfg, unit, max_chunks_per_file=max_chunks_per_file))

            if max_units > 0 and len(units) > max_units:
                units = units[:max_units]
                print(f"[INFO] Capped units to {len(units)} via --max-units")

            print(f"\n[REPO] {repo_name}: files={len(files)}, units={len(units)}")

            if dry_run:
                tasks = []
                for i, u in enumerate(units):
                    provider = providers[i % len(providers)]
                    tasks.append(process_unit(provider, None, u, cfg, repo_map_content, dry_run=True))
                await asyncio.gather(*tasks)
                print(f"[INFO] {repo_name}: generated 0 records")
                continue

            done = 0
            good = 0
            repo_start = time.time()
            total_units = len(units)
            for batch_start in range(0, total_units, max(1, task_batch_size)):
                batch = units[batch_start : batch_start + max(1, task_batch_size)]
                tasks = []
                for i, u in enumerate(batch):
                    provider = providers[(batch_start + i) % len(providers)]
                    tasks.append(
                        asyncio.create_task(
                            process_unit(
                                provider,
                                clients[provider.name],
                                u,
                                cfg,
                                repo_map_content,
                                dry_run=False,
                            )
                        )
                    )
                for fut in asyncio.as_completed(tasks):
                    res = await fut
                    done += 1
                    if res is not None:
                        records.append(res)
                        good += 1
                    if done % max(1, progress_every) == 0 or done == total_units:
                        elapsed = max(0.001, time.time() - repo_start)
                        rate = done / elapsed
                        print(
                            f"  [PROGRESS] {repo_name}: {done}/{total_units} done, "
                            f"{good} kept, {rate:.2f} units/s"
                        )

            print(f"[INFO] {repo_name}: generated {good} records")
    finally:
        for client in clients.values():
            await client.close()
        await close_raw_http_clients()

    return records


# ── Save / split / negatives ─────────────────────────────────────────────────

def dedupe_records(records: List[Dict]) -> List[Dict]:
    seen = set()
    out: List[Dict] = []
    for r in records:
        h = pair_hash(r["anchor"], f"{r['path']}::{r['symbol']}")
        if h in seen:
            continue
        seen.add(h)
        out.append(r)
    return out


def split_records(records: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    random.shuffle(records)
    split_idx = int(len(records) * (1 - TEST_RATIO))
    train = [{"split": "train", **r} for r in records[:split_idx]]
    test = [{"split": "test", **r} for r in records[split_idx:]]
    return train + test, train, test


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def overlap_score(query: str, anchor: str) -> float:
    q = set(tokenize_for_overlap(query))
    a = set(tokenize_for_overlap(anchor))
    if not q or not a:
        return 0.0
    inter = len(q & a)
    return inter / max(len(q), 1)


def build_hard_negatives(train_records: List[Dict], hard_neg_k: int) -> List[Dict]:
    rows: List[Dict] = []
    for i, r in enumerate(train_records):
        candidates = []
        for j, other in enumerate(train_records):
            if i == j:
                continue
            # avoid negatives from same file to reduce false negatives
            if r["filename"] == other["filename"]:
                continue
            score = max(overlap_score(q, other["anchor"]) for q in r["queries"])
            candidates.append((score, other))
        candidates.sort(key=lambda x: x[0], reverse=True)
        hard = [c[1]["anchor"] for c in candidates[:hard_neg_k]]

        for q in r["queries"]:
            rows.append(
                {
                    "query": q,
                    "positive": r["anchor"],
                    "hard_negatives": hard,
                    "repo": r["repo"],
                    "filename": r["filename"],
                    "symbol": r["symbol"],
                }
            )
    return rows


def save_outputs(
    records: List[Dict],
    output_dir: str,
    dry_run: bool,
    create_hard_negatives: bool,
    hard_neg_k: int,
) -> Dict[str, str]:
    paths = {
        "all": str(Path(output_dir) / "all.jsonl"),
        "train": str(Path(output_dir) / "train.jsonl"),
        "test": str(Path(output_dir) / "test.jsonl"),
        "hard_neg": str(Path(output_dir) / "train_hard_negatives.json"),
    }

    if dry_run:
        print(f"[DRY RUN] Would save {len(records)} records to {output_dir}")
        return paths

    os.makedirs(output_dir, exist_ok=True)
    deduped = dedupe_records(records)
    all_rows, train_rows, test_rows = split_records(deduped)

    write_jsonl(Path(paths["all"]), all_rows)
    write_jsonl(Path(paths["train"]), train_rows)
    write_jsonl(Path(paths["test"]), test_rows)

    print(f"[SAVED] {paths['all']} ({len(all_rows)})")
    print(f"[SAVED] {paths['train']} ({len(train_rows)})")
    print(f"[SAVED] {paths['test']} ({len(test_rows)})")

    if create_hard_negatives and train_rows:
        hard_rows = build_hard_negatives(train_rows, hard_neg_k=hard_neg_k)
        with open(paths["hard_neg"], "w", encoding="utf-8") as f:
            json.dump(hard_rows, f, ensure_ascii=False)
        print(f"[SAVED] {paths['hard_neg']} ({len(hard_rows)})")

    return paths


def upload_to_hf(hf_repo: str, output_dir: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[ERROR] Install huggingface_hub: pip install huggingface-hub")
        return

    token = HF_TOKEN or os.getenv("HF_TOKEN")
    if not token:
        print("[ERROR] HF_TOKEN missing")
        return

    api = HfApi(token=token)
    api.create_repo(repo_id=hf_repo, repo_type="dataset", exist_ok=True)

    for name in ["all.jsonl", "train.jsonl", "test.jsonl", "train_hard_negatives.json"]:
        p = Path(output_dir) / name
        if p.exists():
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=name,
                repo_id=hf_repo,
                repo_type="dataset",
                token=token,
            )
            print(f"[HF] Uploaded {p}")

    print(f"[HF] Done! https://huggingface.co/datasets/{hf_repo}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimized embedding dataset generation")
    p.add_argument("--repo", choices=["both", "hyperswitch", "verl"], default="hyperswitch")
    p.add_argument("--max-files", type=int, default=60)
    p.add_argument("--unit", choices=["function", "file"], default="function")
    p.add_argument("--max-chunks-per-file", type=int, default=12)
    p.add_argument("--provider", choices=["auto", "openrouter", "modal"], default="openrouter")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--max-units", type=int, default=0, help="Cap total units per repo; 0 means no cap")
    p.add_argument("--task-batch-size", type=int, default=120, help="Units scheduled per async batch")
    p.add_argument("--include-sandbox-files", action="store_true")
    p.add_argument("--smart-sample", action="store_true", help="Use diversity + signal-aware file sampling")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output-dir", default="data/embedding_dataset_optimized")
    p.add_argument("--create-hard-negatives", action="store_true")
    p.add_argument("--hard-neg-k", type=int, default=5)
    p.add_argument("--upload", action="store_true")
    p.add_argument("--hf-repo", type=str, default=None)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p.parse_args()


async def main_async() -> None:
    args = parse_args()
    random.seed(args.seed)

    providers = load_providers(force_provider=args.provider, base_concurrency=args.concurrency)
    repos = ["hyperswitch", "verl"] if args.repo == "both" else [args.repo]

    print(f"[CONFIG] Repos       : {repos}")
    print(f"[CONFIG] Max files   : {args.max_files}")
    print(f"[CONFIG] Unit        : {args.unit}")
    print(f"[CONFIG] Sampling    : {'smart' if args.smart_sample else 'random'}")
    print(f"[CONFIG] Dry run     : {args.dry_run}")
    print(f"[CONFIG] Sandbox code: {'included' if args.include_sandbox_files else 'skipped'}")
    print(f"[CONFIG] Hard negs   : {'on' if args.create_hard_negatives else 'off'}")

    t0 = time.time()
    records = await run_pipeline(
        repos=repos,
        max_files_per_repo=args.max_files,
        unit=args.unit,
        max_chunks_per_file=args.max_chunks_per_file,
        dry_run=args.dry_run,
        providers=providers,
        skip_sandbox_files=not args.include_sandbox_files,
        smart_sample=args.smart_sample,
        progress_every=args.progress_every,
        max_units=args.max_units,
        task_batch_size=args.task_batch_size,
    )
    elapsed = time.time() - t0
    print(f"[INFO] Pipeline complete in {elapsed:.1f}s: records={len(records)}")

    save_outputs(
        records=records,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        create_hard_negatives=args.create_hard_negatives,
        hard_neg_k=args.hard_neg_k,
    )

    if args.upload and not args.dry_run:
        if not args.hf_repo:
            print("[ERROR] --hf-repo required with --upload")
            return
        upload_to_hf(args.hf_repo, args.output_dir)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
