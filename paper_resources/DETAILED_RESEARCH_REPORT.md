# Research Report: Cross-Domain Stylometry & Adversarial Robustness

**Authors:** [Your Name/Team]
**Date:** January 31, 2026
**Status:** Phase 4 Complete (Cross-Domain Optimization)

---

## 1. Executive Summary
This research investigates the robustness of **Stylometry (Authorship Verification)** against two critical threats: **Domain Shift** (generalizing across emails, blogs, and essays) and **Generative AI Attacks** (robustness against T5 paraphrasing).

We progressed from statistical baselines to a **State-of-the-Art Siamese Neural Network** that achieved **91% accuracy** on the PAN22 benchmark. However, we engaged in "Red Teaming" and discovered extensive vulnerabilities:
1.  **Adversarial Fragility:** AI-paraphrasing erodes model confidence by **~50%**.
2.  **Domain Brittleness:** The model failed to generalize to other domains (Enron Emails, BlogText), dropping to near-random performance.

Our current focus is **Domain Adaptation**. Using a **Domain-Adversarial Neural Network (DANN)** with curriculum learning, we successfully aligned the Email, Blog, and Review domains, achieving **77.5% accuracy on Enron** (up from ~66%). We identified a critical "Negative Transfer" phenomenon on PAN22, which informs our final ensemble strategy.

---

## 2. Experimental Roadmap & Results

### Phase 1: Baselines & The Enron Failure
*   **Objective:** Establish a baseline using classical methods.
*   **Method:** Random Forest + Hand-crafted stylometric features (avg sentence length, punctuation frequency).
*   **Data:** Enron Email Corpus (Noisy, real-world data).
*   **Result:** **~54% Accuracy**.
*   **Conclusion:** Classical features are insufficient for short, noisy texts. Linear separation is impossible in low-dimensional space.

### Phase 2: The Siamese Solution (SOTA Performance)
*   **Objective:** Learn a non-linear similarity metric for authorship.
*   **Method:**
    *   **Architecture:** Siamese Neural Network (MLP) with shared weights.
    *   **Features:** Character 4-grams (TF-IDF, Top 3000).
    *   **Data:** PAN22 (Gold Standard Verification Dataset).
*   **Result:**
    *   **Accuracy:** **90.99%**
    *   **ROC-AUC:** **0.9941** (Near perfect separation)
*   **Contribution:** Demonstrated that deep learning can disentangle style from topic on a single domain.

### Phase 3: Adversarial Vulnerability (Red Teaming)
*   **Objective:** Test robustness against Generative AI.
*   **Method:**
    *   **Attack:** T5-base Paraphraser (`humarin/chatgpt_paraphraser_on_T5_base`).
    *   **Target:** High-confidence "Same Author" pairs (>85% conf).
*   **Result:**
    *   **Mean Confidence Erosion:** **0.497** (Confidence dropped by half).
    *   **Attack Success Rate:** **50.0%** (flipped to "Different").
*   **Conclusion:** Stylometry detects "unconscious habits" (n-grams). AI paraphrasers rewrite these habits, effectively "anonymizing" the style. SOTA models are brittle.

### Phase 4: Cross-Domain Generalization (DANN)
*   **Objective:** Build a single model that works across Email (Enron), Blogs (BlogText), and Essays (PAN22).
*   **Method:** **Domain-Adversarial Neural Network (DANN)**.
    *   **Core Idea:** Force the feature extractor to be *blind* to the domain (e.g., confuse "Email" vs "Blog") while predicting authorship.
    *   **Evolution:**
        *   *V2 (Standard GRL):* Unstable training.
        *   *V3 (Strong GRL):* Model unlearned authorship features (Accuracy ~51%).
        *   *V4 (Curriculum Learning + Optimal Thresholds):* **Success**.
*   **V4 Results:**

| Domain | Accuracy | Precision | Recall | ROC-AUC | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Enron Emails** | **77.5%** | 0.945 | 0.585 | **0.838** | ✅ **Solved** |
| **BlogText** | 60.7% | 0.591 | 0.695 | 0.633 | ⚠️ Viable |
| **PAN22** | 53.6% | 0.625 | 0.269 | 0.535 | ❌ Negative Transfer |
| **AVERAGE** | **64.0%** | **0.720** | **0.516** | **0.669** | |

---

## 3. Analysis of Current Challenges

### The "PAN22 Bottleneck"
While DANN successfully adapted Enron and BlogText (A-distance < 0.4), it failed on PAN22 (A-distance > 1.5).
*   **Reason:** PAN22 contains **Cross-Discourse** pairs (e.g., matching a formal Email to an informal SMS).
*   **Impact:** The style *shift* within a PAN22 pair is larger than the style shift between authors. Standard DANN aligns "PAN22" as a monolith, but internally PAN22 is fractured.
*   **Negative Transfer:** The alignment constraint forced the model to discard the specific n-gram features that made the Siamese network (Phase 2) successful, dropping accuracy from 91% to 53%.

---

## 4. Final Plan: The Ensemble Strategy
To achieve the **>70% Target**, we will leverage the strengths of both models rather than forcing a single model to do everything.

**The "Style-Expert" Ensemble:**
1.  **Router:** A simple classifier determines the domain (Email/Blog vs. PAN22).
2.  **Expert 1 (The Specialist):** If Domain is PAN22, use the **Siamese Network** (91% Acc).
3.  **Expert 2 (The Generalist):** If Domain is Other, use the **DANN V4** (77% Acc on Enron).

**Projected Performance:**
$$ \text{Avg} = \frac{91\% + 77.5\% + 60.7\%}{3} \approx \mathbf{76.4\%} $$

This strategy:
1.  Solves the generalization problem (via DANN).
2.  Preserves SOTA performance on the gold standard (via Siamese).
3.  Scientifically valuable: Demonstrates the trade-off between specialization and generalization.

---

## 5. Artifacts & Resources
*   **Models:**
    *   Siamese (SOTA): `results_pan_siamese/best_model.pth`
    *   DANN (Generalist): `results_dann/dann_siamese_v2.pth`
*   **Code:**
    *   Training: `experiments/dann_training_v4.py`
    *   Evaluation: `experiments/dann_evaluation_v3.py`
*   **Visuals:** `results_dann/dann_embedding_space_final.png` (t-SNE).
