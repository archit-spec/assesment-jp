#!/bin/bash
# Track B – Full Evaluation Script
# Runs baseline eval, post-SFT eval, and comparison.
# Usage: bash run_eval.sh [baseline_model] [sft_model]

set -e

BASELINE_MODEL="${1:-Qwen/Qwen2.5-Coder-1.5B}"
SFT_MODEL="${2:-results/track_b_sft}"
TEST_FILE="data/sft_trackb_test.json"
BASELINE_OUT="results/eval_baseline.json"
POST_SFT_OUT="results/eval_post_sft.json"

mkdir -p results

echo "=============================================="
echo " Track B Evaluation"
echo " Baseline : $BASELINE_MODEL"
echo " Post-SFT : $SFT_MODEL"
echo " Test set : $TEST_FILE"
echo "=============================================="

echo ""
echo "[1/3] Evaluating baseline..."
python eval_sft_trackb.py eval \
    --model "$BASELINE_MODEL" \
    --test-file "$TEST_FILE" \
    --tag baseline \
    --output "$BASELINE_OUT"

echo ""
echo "[2/3] Evaluating post-SFT model..."
python eval_sft_trackb.py eval \
    --model "$SFT_MODEL" \
    --test-file "$TEST_FILE" \
    --tag post_sft \
    --output "$POST_SFT_OUT"

echo ""
echo "[3/3] Comparing results..."
python eval_sft_trackb.py compare "$BASELINE_OUT" "$POST_SFT_OUT"

echo ""
echo "Done! Results saved to $BASELINE_OUT and $POST_SFT_OUT"
