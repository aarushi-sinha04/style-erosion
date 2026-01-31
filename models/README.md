
# Models Directory

This directory contains the PyTorch model definitions.

## Files
- `dann.py`: **(DANN V4 Architecture)**
    - Contains `DANNSiameseV3`: The improved Siamese network with Gradient Reversal Layer, Self-Attention, and Spectral Normalization.
    - Used by `experiments/train_dann.py`.

## Note
- The original Baseline Siamese Network is defined inline in `experiments/train_siamese.py` to preserve the exact configuration used in Phase 2 results.
