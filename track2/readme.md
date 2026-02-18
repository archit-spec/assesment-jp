# Track B: SFT Data & Evaluation Methodology

## 1. Dataset Construction (`generate_sft_trackb.py`)
We constructed a high-quality synthetic dataset by combining **deterministic AST analysis** with **LLM-generated gold responses**.

### Process
1.  **Source Extraction**: We iterated through the `verl` corpus, parsing every Python file.
2.  **Function Filtering**: Extracted functions with 5-60 lines of code, excluding async functions for unit tests.
3.  **Task Formulation**: Created prompts for 5 specific categories using AST:
    *   **Docstring**: Stripped existing docstrings, asked model to regenerate them.
    *   **Completion**: Masked function bodies (keeping signature + docstring), asked model to complete.
    *   **Bugfix**: Deterministically injected bugs (e.g., off-by-one, flipped logic), asked model to identify and fix.
    *   **Explain**: Asked model to explain the function's purpose and logic.
    *   **Unit Test**: Asked model to generate `pytest` cases.
4.  **Gold Response Generation**: Used `Qwen/Qwen3-Coder-30B-A3B-Instruct` (via vLLM) to generate high-quality target responses.
5.  **Quality Filtering**:
    *   Compilability checks for code outputs.
    *   Length and structure checks (e.g., Google-style docstrings).
    *   De-duplication based on instruction/response hashes.
    *   Placeholder detection (rejecting "TODO", "pass", etc.).

### Dataset Stats (`archit11/track_b_sft`)
*   **Total Pairs**: ~300
*   **Train/Test Split**: 85% / 15%
*   **Categories**:
    *   `bugfix`: ~76 pairs
    *   `complete`: ~65 pairs
    *   `docstring`: ~80 pairs
    *   `explain`: ~80 pairs
    *   `unit_test`: ~2 pairs (low due to strict filtering, capable of being expanded)

## 2. Evaluation Methodology (`eval_sft_trackb.py`)
We evaluate models on the held-out test set using a multi-faceted approach.

### Metrics
1.  **pass@1**: Probability that the first generated response is correct (passes syntax check, unit tests, or style/structure constraints).
2.  **pass@k (k=3)**: Probability that at least one of top-k samples is correct.
3.  **Style Score (0-1)**: Heuristic score checking for required elements (e.g., "Args:" section in docstrings, `def test_` in unit tests).
4.  **Hallucination Rate**: Percentage of responses containing placeholders or refusal tokens.
5.  **Execution Rate**: For code tasks (complete, bugfix, unit_test), percentage of outputs that are valid Python syntax.

### Baseline Results (`Qwen/Qwen2.5-Coder-1.5B`)
*   **pass@1**: 56.5%
*   **pass@3**: 78.3%
*   **Weakest Categories**:
    *   `bugfix` (16.7% pass@1) — struggle to identify subtle injected bugs.
    *   `docstring` (40.0% pass@1) — often misses specific style requirements.
*   **Strongest Categories**:
    *   `explain` (100%) — the base model is very good at explaining code.

## 3. Next Steps
1.  **Fine-tune** the 1.5B model on the training split (~250 samples) using LoRA.
2.  **Run Post-SFT Evaluation** to measure improvement in `bugfix` and `docstring` capabilities.
3.  **Compare** deltas to demonstrate the efficacy of the targeted SFT.
