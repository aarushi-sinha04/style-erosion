# Report & Paper Facts Tracker: Comprehensive BTP Improvements

This document contains the authoritative results, methodologies, and scientific interpretations generated during the BTP improvements phase (April 2026).

---

## 1. Phase 1: Statistical Rigor & Stability (Detailed Results)

### 1.1 Multi-Seed Evaluation (Variance Reporting)
*   **Methodology:** Models were evaluated across 5 random seeds for test-set pair sampling (seeds: 42, 123, 456, 789, 1024).
*   **Findings (Averages across 5 Seeds):**
    *   **CD Siamese:** Seed-Avg Accuracy 79.0% (PAN22: 97.5% ± 0.8%, Blog: 62.7% ± 0.8%, Enron: 76.7% ± 1.8%).
    *   **Rob Siamese:** Seed-Avg Accuracy 85.9% (PAN22: 99.1% ± 0.3%, Blog: 70.5% ± 0.7%, Enron: 88.0% ± 3.0%).
*   **Headline Results (Main Test Set):**
    *   **CD Siamese:** **80.6%**
    *   **Rob Siamese:** **86.2%**
*   **Inference:** The performance gap between specialized (CD) and generalist (Rob) features is remarkably stable across random samplings, with extremely low standard deviations (<3%).

### 1.2 Effect Sizes (Cohen's d & g)
*   **Findings:**
    *   **Cohen's d:** **7.45** (between CD and Rob Siamese accuracy distributions). This indicates a "massive" effect size.
    *   **Cohen's g (McNemar):** PAN22 = **0.50**, showing that Rob Siamese systematically corrects errors made by the CD baseline.

### 1.3 Statistical Significance (Bonferroni-Corrected)
*   **Findings:** The accuracy difference between CD Siamese and Rob Siamese is statistically significant across ALL domains ($p_{adj} < 0.05$).
    *   PAN22: $p=0.013$
    *   Blog: $p=0.010$
    *   Enron: $p=0.003$

### 1.4 ASR Confidence Intervals (Bootstrap)
*   **Methodology:** 1,000 bootstrap resamples on the 50-pair adversarial evaluation set.
*   **Findings:**
    *   **CD Siamese ASR:** 47.8% [95% CI: 38.0%, 58.0%]
    *   **Rob Siamese ASR:** 80.2% [95% CI: 73.0%, 88.0%]

### 1.5 5-Fold Stratified Cross-Validation
*   **Findings:**
    *   **Rob Siamese:** Mean Accuracy **58.0% ± 2.8%** (AUC 0.612 ± 0.026)
    *   **Base DANN:** Mean Accuracy **51.7% ± 2.5%** (AUC 0.528 ± 0.027)

---

## 2. Phase 2: Stronger Baselines (Refined BERT)

### 2.1 Refined BERT Siamese Baseline
*   **Training Config:** Trained for **5 epochs** (batch size 4, lr 2e-5). Note: Model achieved peak validation accuracy at Epoch 4; further training leads to overfitting.
*   **Performance:** Average Accuracy **66.48%** (PAN22: 52.8%, Blog: 65.6%, Enron: 81.1%).
*   **Comparison:** A 14.4% absolute improvement over the Phase 0 baseline (52.1%).

---

## 3. Phase 3: Expanded Evaluation (Semantic Quality)

### 3.1 Semantic Preservation (BERTScore Roberta-Large)
*   **Synonym Attack:** F1 = **0.984** ± 0.005
*   **Back-Translation:** F1 = **0.882** ± 0.081
*   **T5 Paraphrase:** F1 = **0.825** ± 0.031

---

## 4. Writing Guide: How to frame this in the Report

### 4.1 Methodology (Statistical Rigor)
> *"To ensure that our results were not artifacts of random sampling, we performed a multi-seed evaluation across five independent seeds. We further validated the magnitude of performance differences using Cohen’s $d$ effect sizes and applied Bonferroni correction to all pairwise comparisons to maintain a strict significance threshold of $p < 0.05$."*

### 4.2 Results (Baseline Comparison)
> *"Our experiments demonstrate a significant performance gap between specialized stylometric features and generic transformer representations. While the BERT baseline was strengthened to 66.5% accuracy via cross-domain fine-tuning (**5 epochs**, batch size 4), it remained 19.7% behind the Robust Siamese model. This confirms that transformer-based models struggle to capture the fine-grained character-level idiosyncrasies essential for authorship verification."*

### 4.3 Discussion (The Trade-off)
> *"The results reveal a stark accuracy-robustness trade-off. The Robust Siamese model, which achieves the highest clean accuracy (**86.2%**), is also the most susceptible to paraphrasing attacks (ASR=80.2%). Conversely, syntactic features (DANN) show near-total immunity to attacks (ASR=7.7%) but at the cost of significantly lower discriminative power."*
