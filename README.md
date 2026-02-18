# assesment-jp

Track-wise project layout.

# please check

./track_a_colab.ipynb
./track_b_colab.ipynb
./track_c_colab.ipynb

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


## Notes

- Large artifacts in `data/` and `results/` were intentionally left in place.
- Folder moves are organizational only; no training logic was changed.
