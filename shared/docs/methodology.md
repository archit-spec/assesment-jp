# Track B Notes (Consolidated)

## 0) Objective
Track B goal is small-scale instruction tuning (`~300-600` pairs) for coding tasks, then evaluation with coding metrics (`pass@1`, `pass@3`) and useful training diagnostics (including eval loss).

## 0.1) Task Families Used
- Explain what a function does.
- Write a docstring for a function.
- Fix a bug in a snippet.
- Suggest code improvements.
- Generate unit tests.
- Complete a partial implementation.

## 0.2) Data Construction Methods Used
- `generate_sft_data.py`:
  Deterministic AST-based synthetic data generation with 6 categories (`explain`, `docstring`, `complete`, `bugfix`, `unit_test`, `improve`).
- `prepare_sft_python_curated.py`:
  Python-focused category curation for coding evaluation (`docstring`, `bugfix`, `improve`, `unit_test`, `complete`).
- `prepare_sft_rust_from_embeddings.py`:
  Rust explain-task SFT set built from embedding dataset fields (`query + anchor + positive`).

## 0.3) Verification + Promotion Pipeline (new)
Script:
- `filter_sft_quality.py`

What it checks:
- Category allowlist.
- Instruction/response length sanity.
- Prompt-category alignment (keyword heuristics).
- Placeholder/template detection (`assert True`, TODO-like outputs, etc.).
- Category-specific response format checks.
- Python compilability checks for code-producing categories.
- Deduplication by normalized `(instruction, response, category)`.
- Response diversity cap (`--max-same-response`) to avoid repeated canned outputs.

Promotion policy:
- Keep rows with `quality_score >= threshold` and no hard failures.
- Demote rows failing hard checks.
- Demote over-represented repeated responses via response-reuse cap.

## 0.4) Current Quality Findings (Python-curated split)
Run with:
- `--min-score 0.68`
- `--max-same-response 3`

Observed:
- Train input `350` -> kept `12`, rejected `338`.
- Test input `64` -> kept `12`, rejected `52`.
- Major rejection causes:
  - response reuse cap (heavy response collapse),
  - placeholder unit-test responses,
  - occasional non-compilable code output.

Interpretation:
- Existing synthetic set is strongly mode-collapsed (many repeated responses).
- This explains weak downstream improvement despite more rows.

## 0.5) Making Eval Loss Informative
- Use a frozen, quality-controlled eval set that is category-balanced and not trivially templated.
- Keep train and eval from the same task distribution, but prevent near-duplicate response templates in eval.
- Track both:
  - optimization metric: training/eval loss,
  - task metric: pass@1/pass@3 on coding categories.
- If eval loss falls but pass@k does not improve, treat it as overfitting/style memorization.

## 0.6) Recommended Next Data Recipe
- Generate new Python simple-medium tasks (calculator/list/string/parsing classes of problems) using the same 5 prompt families.
- Apply `filter_sft_quality.py` immediately after generation.
- Enforce diversity:
  - max response reuse cap,
  - category balancing,
  - optional per-template cap.
- Target final kept set:
  - train: `300-500`,
  - eval: `60-100` (frozen).

## 0.7) Commands (current pipeline)
```bash
uv run python prepare_sft_python_curated.py
uv run python filter_sft_quality.py \
  --in-file data/sft_python_curated_train.json \
  --out-file data/sft_python_curated_train_qc.json \
  --reject-file data/sft_python_curated_train_rejects.json \
  --report-file results/sft_python_train_qc_report.json \
  --min-score 0.68 \
  --max-same-response 3
```

# Synthetic Data Card (Rust SFT from Embedding Pairs)

## 1) Purpose
This synthetic dataset is designed for Track B SFT on Rust-centric code understanding tasks, using Hyperswitch function snippets and aligned natural-language explanations.

Primary goal:
- Improve instruction-following on Rust code explanation tasks.
- Keep supervision grounded in an existing gold explanation field (`positive`) instead of fully free-form generation.

## 2) Data Lineage
Source files:
- `data/hf_download/assesment_embeddings_new_clean/train.jsonl`
- `data/hf_download/assesment_embeddings_new_clean/test.jsonl`

Source row schema (used fields):
- `queries` (list of 4 search queries)
- `anchor` (Rust function snippet + metadata comments)
- `positive` (gold explanation paragraph)
- `path`, `symbol`, `language`, `unit_type`

Observed source distribution:
- Train rows: `434`
- Test rows: `77`
- Language: `Rust` only
- Unit type: `function` only
- Queries per source row: `4` (fixed)

## 3) Synthesis Method
Script:
- `prepare_sft_rust_from_embeddings.py`

Output files:
- `data/sft_rust_embed_train.json`
- `data/sft_rust_embed_test.json`

Transformation:
1. Read each source row.
2. Keep first `N` queries per row (`--max-queries-per-row`, default `2`) for controlled augmentation.
3. Build instruction using:
   - user query
   - snippet path + symbol
   - Rust code block (`anchor`)
   - explicit answer rubric (what it does, I/O/errors, why relevant)
4. Use source `positive` as gold response.
5. Set category to `explain`.

Template:
```text
You are a Rust maintainer for the Hyperswitch codebase.
User query: {query}

Explain the following Rust snippet in the context of this query.
Your answer must include:
1) What the function/module does
2) Important inputs/outputs/errors
3) Why this snippet matches the query

Path: {path}
Symbol: {symbol}

```rust
{anchor}
```
```

## 4) Filtering and Cleaning Rules
Applied deterministic filters:
- Drop rows with empty `queries`.
- Drop rows with empty `anchor` or empty `positive`.
- Drop blank query strings.

Length controls (to reduce truncation/cost):
- `anchor` clipped to `1400` chars (`--max-code-chars`)
- `positive` clipped to `900` chars (`--max-response-chars`)
- clipping appends `...` marker

No semantic paraphrasing or random rewriting is applied in this stage.

## 5) Seeds and Reproducibility
This synthesis step is deterministic by default:
- No random sampling in the script.
- Ordering follows source file ordering.
- Main variation is controlled by CLI arguments only (e.g., max queries per row, clip sizes).

Training seeds (downstream SFT) are set in `track_b_sft.py`:
- `RANDOM_SEED = 42`

## 6) Output Distribution
Generated dataset stats:
- Train examples: `868`
- Test examples: `154`
- Category distribution: `explain` = 100%

Instruction length (chars):
- Train: min `839`, p50 `1138`, p90 `1856`, max `1909`, mean `1261.9`
- Test: min `802`, p50 `1212`, p90 `1854`, max `1901`, mean `1319.4`

Response length (chars):
- Train: min `325`, p50 `515`, p90 `665`, max `843`, mean `523.9`
- Test: min `333`, p50 `512`, p90 `628`, max `773`, mean `517.4`

## 7) Edge Cases and Risks
Known edge cases:
- Some `anchor` snippets include metadata comments (`// PATH`, `// SYMBOL`), not purely code.
- Long functions are clipped; may omit logic relevant to the query.
- Using top-2 queries per row introduces near-duplicate instructions mapped to same gold response.

Potential quality risks:
- Gold responses may be high-level and not mention all implementation details.
- Style overfitting risk (model learns explanation tone more than deep reasoning).
- Task skew: only `explain` category; weaker coverage for bugfix/edit/test generation.

## 8) Quality Checks Used
Programmatic checks:
- Row counts after generation.
- Empty-field filtering.
- Length clipping.
- Category consistency.

Suggested next checks:
- Deduplicate near-identical instructions.
- Add hard examples with misleading but related queries.
- Add contrastive negatives for stronger reasoning boundaries.

## 9) Collaboration Notes
Why this is machine-collaboration friendly:
- Deterministic, script-based generation.
- Traceable lineage from source row to final instruction/response.
- Explicit prompt rubric and clipping constraints.
- Re-runnable with parameter changes via one command:

```bash
uv run python prepare_sft_rust_from_embeddings.py
```
