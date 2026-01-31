
# Cross-Domain Authorship Verification Project

**Goal:** Robust authorship verification across diverse domains (Email, Blog, Essays) and defense against AI attacks.

## Directory Structure
- `experiments/`: Main scripts for training and evaluation. (**Start Here**)
- `results/`: Output logs, plots, model weights, and performance reports.
- `models/`: PyTorch model definitions.
- `utils/`: Data loaders and feature extractors.
- `paper_resources/`: Reports, summaries, and logs for the research paper.
- `archive/`: Old/Legacy experiments.

## Quick Start

### 1. Train the Best Model (DANN V4)
```bash
python experiments/train_dann.py
```
This trains the Domain-Adversarial Network on PAN22, BlogText, Enron, and IMDB using Curriculum Learning.
**Output:** `results/final_dann/`

### 2. Evaluate the Model
```bash
python experiments/eval_dann.py
```
Generates t-SNE plots and accuracy tables showing domain alignment.

### 3. Run Inference (Demo)
```bash
python inference.py file1.txt file2.txt
```
Uses the Phase 2 Siamese Baseline (SOTA on PAN22) to verify authorship.

## Key Reports
- **Executive Summary:** `paper_resources/PROJECT_SUMMARY.md`
- **Full Report:** `paper_resources/DETAILED_RESEARCH_REPORT.md`
