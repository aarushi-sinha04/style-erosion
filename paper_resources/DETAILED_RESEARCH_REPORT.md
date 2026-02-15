# Research Report: Cross-Domain Stylometry & Adversarial Robustness

**Authors:** [Your Name/Team]
**Date:** February 15, 2026
**Status:** Complete — Ready for Paper Submission

---

## 1. Executive Summary

This research investigates the robustness of authorship verification against **Domain Shift** and **Generative AI Attacks** (T5 paraphrasing). We progressed through 7 experimental phases, ultimately discovering that **feature granularity determines the accuracy–robustness trade-off**.

**Key Result:** Our Rob Siamese model achieves **86.2% cross-domain accuracy** (SOTA) but character 4-gram features are inherently vulnerable to paraphrase attacks (74% ASR). Meanwhile, syntactic features (POS + readability) achieve only 60.4% accuracy but near-immunity to attacks (7.7% ASR).

---

## 2. Experimental Progression

### Phase 1: Classical Baselines
- **Method:** Random Forest + hand-crafted features, then LogReg + char 3-grams
- **Result:** ~54% average cross-domain accuracy
- **Finding:** Classical features are insufficient for short, noisy texts

### Phase 2: Siamese Network (Single Domain)
- **Method:** Character 4-gram Siamese on PAN22
- **Result:** **97.0% PAN22** accuracy, 0.998 ROC-AUC
- **Finding:** Character n-grams are excellent within a single domain

### Phase 3: Adversarial Vulnerability
- **Method:** T5 paraphrasing attack on high-confidence same-author pairs
- **Result:** **50% Attack Success Rate**, BERTScore F1 = 0.885
- **Finding:** AI paraphrasers destroy character-level authorial fingerprints

### Phase 4: DANN Domain Adaptation
- **Method:** Domain-Adversarial Neural Network with curriculum learning
- **Result:** 78.8% Enron, 55.8% Blog, 53.2% PAN22 (avg 62.6%)
- **Finding:** Multi-view features work for structured domains (email) but fail on diverse domains (fanfiction)

### Phase 5: Cross-Domain Siamese
- **Method:** Train Siamese on PAN22 + Blog + Enron combined
- **Result:** **83.5% accuracy**, 0.98 ROC-AUC
- **Finding:** Cross-domain data exposure is more effective than domain adaptation

### Phase 6: Adversarial Fine-Tuning
- **Method:** Fine-tune CD Siamese with adversarial consistency loss
- **Result:** **88.6% validation accuracy** (+5.6pp), 0.97 ROC-AUC
- **Finding:** Adversarial training acts as data augmentation, improving clean accuracy

### Phase 7: Comprehensive Evaluation
- **All 7 models** evaluated on 3 domains + ASR

---

## 3. Final Results

### 3.1 Clean Accuracy

| Model | Features | PAN22 | Blog | Enron | **Avg** |
|:---|:---|---:|---:|---:|---:|
| LogReg | Char 3-grams | 62.8% | 50.0% | 50.0% | 54.3% |
| Base DANN | Multi-view | 53.2% | 55.8% | 78.8% | 62.6% |
| Robust DANN | Multi-view | 54.4% | 52.8% | 74.0% | 60.4% |
| PAN22 Siamese | Char 4-grams | 97.0% | 52.1% | 56.8% | 68.6% |
| CD Siamese | Char 4-grams | 98.2% | 66.5% | 77.2% | 80.6% |
| **Rob Siamese** | **Char 4-grams** | **99.4%** | **71.9%** | **87.2%** | **86.2%** |
| Ensemble | Hybrid | 98.0% | 64.4% | 76.8% | 79.7% |

### 3.2 Adversarial Robustness

| Model | Features | ASR ↓ |
|:---|:---|---:|
| **Robust DANN** | **POS + readability** | **7.7%** |
| LogReg | Char 3-grams | 10.8% |
| Base DANN | Multi-view | 14.3% |
| CD Siamese | Char 4-grams | 44.0% |
| Ensemble | Hybrid | 48.0% |
| PAN22 Siamese | Char 4-grams | 50.0% |
| Rob Siamese | Char 4-grams | 74.0% |

**BERTScore F1 = 0.885** confirms semantic preservation of attacks.

### 3.3 Error Analysis

| Domain | Errors | Error Rate | Concentration |
|:---|---:|---:|---:|
| PAN22 | 6 | 1.2% | 3.5% |
| Blog | 131 | 27.9% | **76.6%** |
| Enron | 34 | 15.2% | 19.9% |

- **FPs have high confidence** (0.886): model is "very sure" about wrong same-author predictions
- **FNs have low confidence** (0.118): correctly flagging uncertainty
- **Blog dominates errors** (77%): high intra-author stylistic variance

---

## 4. Core Finding: The Accuracy–Robustness Trade-off

**Feature granularity determines the trade-off:**
- Fine-grained (char n-grams): High accuracy (86.2%), high vulnerability (74% ASR)
- Coarse-grained (syntactic): Low accuracy (60.4%), near-immunity (7.7% ASR)
- Hybrid (ensemble): Middle ground (79.7%, 48% ASR)

**This trade-off is fundamental—not architectural.** Adversarial training cannot overcome it because the vulnerability is at the feature level.

---

## 5. Artifacts & Code

### Models
| File | Description |
|:---|:---|
| `results/robust_siamese/best_model.pth` | **Best model** — 86.2% cross-domain |
| `results/siamese_crossdomain/best_model.pth` | CD Siamese — 80.6% cross-domain |
| `results/siamese_baseline/best_model.pth` | PAN22 Siamese — 97% single-domain |
| `results/final_dann/dann_model_v4.pth` | Base DANN — most robust (14.3% ASR) |
| `results/robust_dann/robust_dann_model.pth` | Robust DANN — 7.7% ASR |

### Scripts
| File | Purpose |
|:---|:---|
| `experiments/train_dann.py` | DANN V4 curriculum learning |
| `experiments/train_siamese_crossdomain.py` | Cross-domain Siamese |
| `experiments/train_robust_siamese.py` | Adversarial fine-tuning |
| `experiments/eval_robust_all.py` | 6-model comprehensive evaluation |
| `experiments/eval_baselines.py` | LogReg baseline |
| `experiments/error_analysis.py` | FP/FN pattern analysis |
| `figures/generate_paper_figures.py` | 5 publication figures |

### Paper
| File | Description |
|:---|:---|
| `paper/01_abstract.txt` | 250-word abstract |
| `paper/02_introduction.txt` | Introduction with RQs |
| `paper/03_methodology.md` | Full methodology |
| `paper/04_results.md` | All results with tables |
| `paper/05_discussion.md` | Trade-off analysis |
| `paper/06_conclusion.md` | Conclusion |
| `paper/tables.tex` | 5 LaTeX tables |

### Figures
| File | Description |
|:---|:---|
| `figures/fig1_tradeoff.png` | Accuracy vs ASR scatter |
| `figures/fig2_accuracy_bars.png` | Cross-domain accuracy |
| `figures/fig3_asr_comparison.png` | ASR horizontal bars |
| `figures/fig4_ablation.png` | Siamese progression |
| `figures/fig5_error_analysis.png` | Error pattern panels |
