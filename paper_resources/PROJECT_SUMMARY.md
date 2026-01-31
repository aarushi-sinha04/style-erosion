# Project Status: Cross-Domain Authorship Verification

**Date:** 2026-01-31
**Goal:** Achieve robust authorship verification (>70%) across varying domains (Emails, Blogs, Research/Stories).

---

## 1. What We Have Done

### Phase 1-3: Single-Domain Mastery (Completed)
- **Baseline Models:** Statistical methods failed on noisy data (Enron).
- **Siamese Network (PAN22):** Achieved **91% Accuracy** and **0.99 ROC-AUC** on the PAN22 benchmark (state-of-the-art for this specific domain).
- **Adversarial Vulnerability:** Demonstrated that AI-paraphrased attacks (T5) reduce model confidence by ~50%.

### Phase 4: Cross-Domain Generalization (Current Focus)
- **Problem:** The high-performing Siamese model failed to generalize to other domains (emails, blogs).
- **Solution:** Implemented **Domain-Adversarial Neural Network (DANN)** to learn domain-invariant features.
- **Iterations:**
  - **V2 (Baseline DANN):** 59% avg accuracy. GRL was too unstable.
  - **V3 (Strong GRL):** 51% avg accuracy. Strong alignment collapsed authorship features.
  - **V4 (Curriculum DANN + Optimal Thresholds):** **64.0% avg accuracy**. Best result so far.

---

## 2. Current Status (DANN V4)

| Domain | Accuracy | Status | Insight |
|--------|----------|--------|---------|
| **Enron** | **77.5%** | ✅ Success | Model successfully learned email style signatures. High AUC (0.84). |
| **BlogText** | 60.7% | ⚠️ Mixed | Decent generalization, but blog styles are diverse/noisy. |
| **PAN22** | 53.6% | ❌ Fail | **Negative Transfer.** The features that gave 91% accuracy in single-domain training were domain-specific. DANN forced the model to discard them to satisfy domain invariance, causing accuracy to drop to random guessing. |
| **Average** | **64.0%** | ⚠️ Gap | Gap of 6% to reach the 70% target. |

---

## 3. What is Left

### Immediate Actions (To reach 70%)
1.  **Fix PAN22 Bottleneck:**
    - The DANN approach is too aggressive for PAN22.
    - **Strategy:** Relax domain alignment constraint for PAN22 specifically (Partial DANN) or use a "Cross-Discourse Adapter" that handles the email-to-SMS shift within PAN22.
2.  **Ensemble Approach:**
    - Combine the **Specialized Siamese Network** (91% on PAN22) with the **DANN** (77% on Enron).
    - If Domain == PAN22 -> Use Siamese.
    - If Domain == Other -> Use DANN.
    - **Expected Result:** (91% + 77% + 60%) / 3 = **~76% Average**. This comfortably beats the 70% target.

### Future/Optional
- **Advanced Features:** Move from char 4-grams to Transformer embeddings (BERT/RoBERTa) for better semantic robustness.
- **Defense:** Retrain with adversarial examples to harden against T5 attacks.

---

## 4. Target
- **Primary:** Achieve **70%+ Average Cross-Domain Accuracy**. (Achievable via Ensemble).
- **Secondary:** Submit paper demonstrating trade-off between *Specialization* (91% PAN22) and *Generalization* (DANN).
