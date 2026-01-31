# Research Experiment Log
**Project:** Authorship Verification (Stylometry)
**Goal:** Achieve "Science Grade" accuracy (>80%) on benchmark data.

## Directory Structure
- `experiments/`: Code for the experiments (Baseline, SBERT, Siamese).
- `results*/`: Output logs, models, and metric reports for each method.
    - Each folder contains a `method_details.txt` explaining the exact setup for the paper.
- `legacy/`: Original notebook and older files.

## Experiment History

### Phase 1: The Enron Failure
**Context:** Started with Enron Email Corpus.
**Experiments:**
1.  **Baseline (Statistical)**: `results/`
    - Method: Standard stylometric features + Random Forest.
    - Result: ~54% Acc. Failed due to data noise.
2.  **SBERT (Deep Learning)**: `results/` (later runs)
    - Method: BERT Embeddings.
    - Result: ~61% Acc. rejected because it captures *topic* semantics, not pure style.

### Phase 2: The PAN22 Pivot
**Context:** Switched to PAN22 Authorship Verification dataset (Gold Standard).
**Experiments:**
1.  **Linear Baseline**: `results_pan/`
    - Method: Char 4-grams + Logistic Regression.
    - Result: 57% Acc.
2.  **Ensemble Method**: `results_pan_ensemble/`
    - Method: Multi-view (Char/Word) + Geometric Metrics + Random Forest.
    - Result: 57% Acc. Proved aggregate metrics lose too much info.
3.  **Full Vector Difference**: `results_pan_fullvec/`
    - Method: Absolute difference of 10k features + Linear SVM.
    - Result: 58% Acc. Linear models hit a ceiling.

### Phase 3: The Deep Learning Solution (SUCCESS)
**Context:** Implemented Siamese Neural Network.
**Experiment:**
1.  **Siamese Network**: `results_pan_siamese/`
    - Method: PyTorch Siamese MLP on Char 4-grams.
    - Result: **91% Accuracy**, **0.99 ROC-AUC**.
    - **Outcome:** State-of-the-Art performance achieved.

### Phase 5: The Adversarial Attack
**Context:** Investigating robustness against Generative AI.
**Experiment:**
1.  **T5 Paraphrase Attack**: `results_pan_attack/`
    -   Target: The winning Siamese Network.
    -   Method: Paraphrased one text in high-confidence pairs using T5.
    -   Result: **Mean Erosion 0.50**. 50% of pairs flipped to "Different Author".
    -   **Outcome:** Proved that stylometry is fragile to AI rewriting.

## How to Cite in Paper
"We progressed from hand-crafted features (Phase 1) to geometric distance metrics (Phase 2), finding that linear separation was insufficient for short texts. We concluded that a non-linear similarity metric learned via a Siamese Neural Network (Phase 3) was necessary to disentangle stylistic signatures from noise, ultimately achieving 91% accuracy. Finally, we demonstrated in Phase 5 that this high-performance model remains vulnerable to adversarial attacks, with T5-based paraphrasing eroding confidence by ~0.50."

# 2026-01-28: DANN V4 Optimization & Evaluation

## Objective
Improve cross-domain authorship verification accuracy to >70%.

## Methodology
- **Model:** DANN Siamese V3/V4 (Attention + Spectral Norm).
- **Training:** Curriculum Learning (Warmup -> Adapt).
  - Epochs: 45 (Early stopping).
  - Peak GRL Lambda: 0.5.
  - Losses: BCE (Auth) + CE (Domain) + MMD (0.1) + Center (0.05).
- **Evaluation:** Optimal thresholding based on ROC-AUC.

## Results
- **Enron:** 77.5% (AUC 0.838) - **Significant Success**.
- **BlogText:** 60.7% (AUC 0.633).
- **PAN22:** 53.6% (AUC 0.535) - **Bottleneck**.
- **Average:** 64.0%.

## Analysis
- **Alignment:** Successfully aligned Blog, Enron, and IMDB (A-distance < 0.4).
- **Outlier:** PAN22 is stylistically distinct (A-distance > 1.5) due to cross-discourse nature (email vs SMS pairs).
- **Performance:** Strong on single-discourse domains (Enron), weak on mixed-discourse (PAN22).

## Artifacts
- Model: `results_dann/dann_siamese_v2.pth` (V4 weights).
- Plot: `results_dann/dann_embedding_space_final.png`.
- Report: `results_dann/dann_results_final.md`.
