
# ML Assignment Part 2 - Track A, B, C

> **To verify results, run the eval notebooks on Colab:**
> - 📊 [Track A – Perplexity Eval](https://colab.research.google.com/github/archit-spec/assesment-jp/blob/main/eval_track_a.ipynb) — baseline vs fine-tuned perplexity

> - 🔍 [Track C – Embedding Eval](https://colab.research.google.com/github/archit-spec/assesment-jp/blob/main/eval_track_c.ipynb) — MRR@10, nDCG@10, Recall@10



> - 💬 [Track B – SFT Output Comparison](https://colab.research.google.com/github/archit-spec/assesment-jp/blob/main/eval_track_b.ipynb) 

This repository contains the solution for the 3 requested tracks.
**Hardware Requirement**: T4 GPU (free Colab tier).

# note

there different note books to show training (trac_x_unsloth.ipynb) since gpu memory cannot be freed from the repl/notebook itself unless the kernel is restarted. since  weights are allocated directly in a C++ CUDA context outside PyTorch's allocator, making empty_cache() ineffective.

also pls read the model cards and dataset cards for more on huggingface for more details on the approach 


each track has link to model with model card and dataset with dataset card.
## Notebooks

| Track | Description | Notebook |
|-------|-------------|----------|
| **A** | Extended Pre-Training | [`track_a_unsloth.ipynb`](./track_a_unsloth.ipynb) (Unsloth) |
| **B** | Instruction Tuning (SFT) | [`track_b_unsloth.ipynb`](./track_b_unsloth.ipynb) (Unsloth) |
| **C** | Embedding Fine-Tuning | [`track_c_unsloth.ipynb`](./track_c_unsloth.ipynb) (Unsloth) |

---

## Track A: Extended Pre-Training

Continue pre-training `Qwen/Qwen2.5-Coder-3B` on a curated Rust code corpus extracted from `juspay/hyperswitch`.

### Results
| Metric | Baseline | Post-Training | Δ |
|--------|----------|---------------|---|
| **Perplexity** | 2.0576 | **1.3954** | **-32.19% ↓** |

### Artifacts
| Type | Name | Link |
|------|------|------|
| **Model** | `archit11/qwen2.5-coder-3b-hyperswitch-track-a-merged` | [Hugging Face](https://huggingface.co/archit11/qwen2.5-coder-3b
-hyperswitch-track-a-merged) |
| **Dataset** | `archit11/hyperswitch-code-corpus-track-a` | [Hugging Face](https://huggingface.co/datasets/archit11/hyperswitch-code-corpus-track-a) |

### Methodology
1.  **Data Selection**: Top 300 Rust files from `crates/` selected by structural richness (func/type count) and line count (25-4000).
2.  **Curriculum Training**: Trained on chunks of increasing size: 768 → 1024 → 1536 tokens.
3.  **Method**: LoRA (r=16, alpha=32), LR=1e-3, Cosine Schedule.

### Reproducing Results

Install dependencies:
```bash
pip install transformers datasets torch tqdm
```

Run the evaluation (compares base `Qwen2.5-Coder-3B` vs fine-tuned, outputs JSON + HTML report):
```bash
python eval_perplexity.py
```

Expected output:
```
Model             Perplexity
----------------------------
Base                  2.0576
Fine-Tuned            1.3954
Δ                    -0.6623  (-32.19%)
```

Reports saved to `results/track_a/`:
- `perplexity_metrics.json` — machine-readable summary
- `perplexity_report.html` — human-readable report with data card

---

## Track B: Instruction Tuning (SFT)

Fine-tune `Qwen/Qwen2.5-Coder-1.5B` on synthetic Python instructions (Docstring, Explain, Bugfix, Complete, Unit Test).

### Results
| Metric | Baseline | Post-SFT | ^ |
|--------|----------|----------|---|
| **pass@1** | 0.565 | **0.804** | **+23.9%**  |
| **pass@3** | 0.783 | 0.848 | +6.5% |

### Artifacts
| Type | Name | Link |
|------|------|------|
| **Model** | `archit11/track_b_sft_merged` | [Hugging Face](https://huggingface.co/archit11/track_b_sft_merged) |
| **Dataset** | `archit11/track_b_sft` | [Hugging Face](https://huggingface.co/datasets/archit11/track_b_sft) |

### Methodology
1.  **Data Generation**: Extracted real functions from `verl` repo. Used `Qwen3-Coder-30B-Instruct` to generate 5 types of tasks (Docstring, Bugfix, etc.).
2.  **Training**: LoRA (r=16), 3 epochs, LR=2e-4.
3.  **Eval**: Execution-based `pass@k` for code tasks; structural checks for text tasks.

---

## Track C: Embedding Fine-Tuning

Fine-tune `Qwen/Qwen3-Embedding-0.6B` on text\u2192code retrieval pairs.
 code↔query retrieval as a form of intermediate pre-training:           
                                                                         
  Stage 1: (anchor, positive)  → teaches domain vocabulary, Rust         
  concepts, Hyperswitch-specific terminology                             
  Stage 2: (query, anchor)     → fine-tunes specifically for the         
  retrieval direction                                                  

### \ud83d\udcca Results (Best Run - 3 Epochs)
| Metric | Baseline | Fine-Tuned | \u0394 |
|--------|----------|------------|---|
| **MRR@10** | 0.8840 | **0.9617** | **+0.0777 \u2191** |
| **nDCG@10** | 0.9093 | **0.9710** | **+0.0617 \u2191** |
| **Recall@10** | 0.9870 | **1.0000** | **+0.0130 \u2191** |

### Artifacts
| Type | Name | Link |
|------|------|------|
| **Model** | `archit11/assesment_qwen3_embedding_06b_e3` | [Hugging Face](https://huggingface.co/archit11/assesment_qwen3_embedding_06b_e3) |
| **Dataset** | `archit11/assesment_embeddings_new` | [Hugging Face](https://huggingface.co/datasets/archit11/assesment_embeddings_new) |

### Methodology
1.  **Data Generation**: Synthesized natural language queries for Rust files using `Qwen-3.5-Instruct`.
2.  **Training**: `MultipleNegativesRankingLoss` with `sentence-transformers`.
3.  **Eval**: Retrieval on held-out query-code pairs.

### Reproducing Results

Install dependencies:
```bash
pip install sentence-transformers datasets torch
```

Run the evaluation (compares base Qwen3-Embedding-0.6B vs fine-tuned):
```bash
python eval_embeddings.py
```

Expected output:
```
Metric            Baseline   Fine-Tuned        Δ
-----------------------------------------------
MRR@10              0.8875       0.9617  +0.0742
nDCG@10             0.9126       0.9710  +0.0584
Recall@10           0.9903       1.0000  +0.0097
```

---

## Directory Structure
- `track_a_unsloth.ipynb`: Unsloth-optimized notebook for Track A
- `track_b_unsloth.ipynb`: Unsloth-optimized notebook for Track B
- `track_c_unsloth.ipynb`: Unsloth-optimized notebook for Track C
- `data/`: Local data artifacts
- `results/`: Local evaluation logs
