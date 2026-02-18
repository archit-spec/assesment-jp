
# ML Assignment Part 2 - Track A, B, C

This repository contains the solution for the 3 requested tracks.
**Hardware Requirement**: T4 GPU (free Colab tier).

## Notebooks

| Track | Description | Notebook |
|-------|-------------|----------|
| **A** | Extended Pre-Training | [`track_a_colab.ipynb`](./track_a_colab.ipynb) |
| **B** | Instruction Tuning (SFT) | [`track_b_colab.ipynb`](./track_b_colab.ipynb) |
| **C** | Embedding Fine-Tuning | [`track_c_colab.ipynb`](./track_c_colab.ipynb) |

---

## \ud83d\ude80 Track A: Extended Pre-Training

Continue pre-training `Qwen/Qwen2.5-Coder-3B` on a curated Rust code corpus extracted from `juspay/hyperswitch`.

### \ud83d\udcca Results
| Metric | Baseline | Post-Training | \u0394 |
|--------|----------|---------------|---|
| **Perplexity** | 2.2832 | **1.5429** | **\u221232.42%** \u2713 |

### \ud83d\udce6 Artifacts
| Type | Name | Link |
|------|------|------|
| **Model** | `archit11/qwen2.5-coder-3b-hyperswitch-track-a-merged` | [Hugging Face](https://huggingface.co/archit11/qwen2.5-coder-3b-hyperswitch-track-a-merged) |
| **Dataset** | `archit11/hyperswitch-code-corpus-track-a` | [Hugging Face](https://huggingface.co/datasets/archit11/hyperswitch-code-corpus-track-a) |

### Methodology
1.  **Data Selection**: Top 300 Rust files from `crates/` selected by structural richness (func/type count) and line count (25-4000).
2.  **Curriculum Training**: Trained on chunks of increasing size: 768 \u2192 1024 \u2192 1536 tokens.
3.  **Method**: LoRA (r=16, alpha=32), LR=1e-3, Cosine Schedule.

---

## \ud83e\udde0 Track B: Instruction Tuning (SFT)

Fine-tune `Qwen/Qwen2.5-Coder-1.5B` on synthetic Python instructions (Docstring, Explain, Bugfix, Complete, Unit Test).

### \ud83d\udcca Results
| Metric | Baseline | Post-SFT | \u0394 |
|--------|----------|----------|---|
| **pass@1** | 0.565 | **0.804** | **+23.9%** \u2713 |
| **pass@3** | 0.783 | 0.848 | +6.5% |

### \ud83d\udce6 Artifacts
| Type | Name | Link |
|------|------|------|
| **Model** | `archit11/track_b_sft_merged` | [Hugging Face](https://huggingface.co/archit11/track_b_sft_merged) |
| **Dataset** | `archit11/track_b_sft` | [Hugging Face](https://huggingface.co/datasets/archit11/track_b_sft) |

### Methodology
1.  **Data Generation**: Extracted real functions from `verl` repo. Used `Qwen3-Coder-30B-Instruct` to generate 5 types of tasks (Docstring, Bugfix, etc.).
2.  **Training**: LoRA (r=16), 3 epochs, LR=2e-4.
3.  **Eval**: Execution-based `pass@k` for code tasks; structural checks for text tasks.

---

## \ud83d\udd0d Track C: Embedding Fine-Tuning

Fine-tune `Qwen/Qwen3-Embedding-0.6B` on text\u2192code retrieval pairs.

### \ud83d\udcca Results (Training)
| Metric | Value |
|--------|-------|
| **Final Loss** | 0.0380 |
| **Eval Metrics** | MRR@10, nDCG@10 (Computed in notebook) |

### \ud83d\udce6 Artifacts
| Type | Name | Link |
|------|------|------|
| **Model** | `archit11/assesment_qwen3_embedding_06b_e3` | [Hugging Face](https://huggingface.co/archit11/assesment_qwen3_embedding_06b_e3) |
| **Dataset** | `archit11/code-embedding-dataset` | [Hugging Face](https://huggingface.co/datasets/archit11/code-embedding-dataset) |

### Methodology
1.  **Data Generation**: Synthesized natural language queries for Rust files using `Qwen-3.5-Instruct`.
2.  **Training**: `MultipleNegativesRankingLoss` with `sentence-transformers`.
3.  **Eval**: Retrieval on held-out query-code pairs.

---

## Directory Structure
- `track_a_colab.ipynb`: Standalone notebook for Track A
- `track_b_colab.ipynb`: Standalone notebook for Track B
- `track_c_colab.ipynb`: Standalone notebook for Track C
- `data/`: Local data artifacts
- `results/`: Local evaluation logs
