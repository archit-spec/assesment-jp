# assesment-jp

Track-wise project layout.

## Structure

- `track1/`:
  - `scripts/`: Extended pretraining (Track A) scripts
  - `notebooks/`: Track 1 notebook(s)
  - `docs/`: Track 1 data/process notes
- `track2/`:
  - `scripts/`: SFT (Track B) generation/training/eval scripts
  - `docs/`: Track 2 notes
- `track3/`:
  - `scripts/`: Embedding/retrieval (Track C) scripts
  - `notebooks/`: Track 3 notebook(s)
- `shared/`:
  - `scripts/`: cross-track helpers
  - `docs/`: shared docs
- `data/`, `results/`, `checkpoints/`, `wandb/`: artifacts and outputs

## Main entrypoints

- Track 1 pretraining: `track1/scripts/track_a_pretraining.py`
- Track 1 notebook: `track1/notebooks/track1_dataset_training_perplexity.ipynb`
- Track 2 training: `track2/scripts/track_b_sft.py`
- Track 2 eval: `track2/scripts/eval_sft_trackb.py`
- Track 3 embeddings: `track3/scripts/track_c_embeddings.py`

## Notes

- Large artifacts in `data/` and `results/` were intentionally left in place.
- Folder moves are organizational only; no training logic was changed.
