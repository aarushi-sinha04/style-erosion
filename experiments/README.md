
# Experiments Directory

This directory contains the main scripts for training and evaluating the models.

## Training Scripts
- `train_dann.py`: **(RECOMMENDED)** Trains the Domain-Adversarial Neural Network (DANN V4).
    - Features: Curriculum learning, WARMUP phase, Optimal Thresholding.
    - Usage: `python experiments/train_dann.py`
    - Output: `results/final_dann/`

- `train_siamese.py`: Trains the Siamese Network baseline on PAN22 (Single Domain).
    - Usage: `python experiments/train_siamese.py`
    - Output: `results/siamese_baseline/`

## Evaluation Scripts
- `eval_dann.py`: Evaluates the trained DANN model on all domains.
    - Usage: `python experiments/eval_dann.py`
    - Generates t-SNE plots and accuracy tables.

- `attack_siamese.py`: Runs the T5 Adversarial Attack experiment on the Siamese model.
    - Usage: `python experiments/attack_siamese.py`
