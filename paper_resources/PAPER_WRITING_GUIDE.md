# Research Paper Writing Guide
**Project Title Suggestion**: *From Linear Baselines to Deep Siamese Networks: Disentangling Stylistic Signatures in Short-Text Authorship Verification*

## 1. Abstract
State that traditional stylometry fails on modern short texts (emails/tweets) due to noise. We leveraged the PAN22 benchmark. Linear models (Logistic Regression, SVM) plateaued at 58% accuracy. We proposed a **Siamese Neural Network** using Character N-gram embeddings, which learns a non-linear metric space for style. This approach achieved **91% Accuracy** and **0.99 ROC-AUC**, effectively solving the verification task where statistical baselines failed.

## 2. Introduction
**Key Phrasing:**
- *"Authorship Verification is the fundamental task of determining if two texts were written by the same individual."*
- *"Traditional Stylometry relies on hand-crafted features (Winnow, Burrows’ Delta) which assume long texts. In digital forensics (emails), these signals are drowned out by noise."*
- *"We hypothesize that 'style' in short text is not a linear summation of feature counts, but a complex interaction of character patterns. To capture this, we employ Deep Metric Learning."*

## 3. Methodology
**A. Dataset (PAN22)** -> See `results_pan/method_details.txt`
- Benchmark dataset standardizes the problem (unlike Enron which we found to be noisy).
- Structure: Text Pairs (A, B) -> Label (Same/Different).

**B. Feature Engineering**
- **Why Character N-grams?**
    - *"We utilized Character 4-grams (e.g., 'tion', ' the') because they capture sub-lexical morphological and punctuation patterns, which are harder for an author to consciously manipulate than word choice."*
- **Vectorization**: TF-IDF with sublinear scaling (to reduce impact of outliers). Dimension: 3000.

**C. The Model (Siamese Network)** -> See `results_pan_siamese/method_details.txt`
- **Architecture**:
    - Two identical sub-networks (weights shared) process Text A and Text B.
    - They map texts into a "Stylometric Latent Space".
    - A "Comparator Head" takes $|u - v|$ (Distance) and $u \times v$ (interaction) to classify.
- **Why Siamese?**: It allows the model to learn "Similarity" generically, rather than learning specific author classes (which is impossible in Verification where unseen authors appear in test time).

## 4. Experiments & Results
**A. Baseline Failure**
- *"We first implemented standard statistical approaches (Cosine Distance, Logistic Regression) on the same feature set."*
- *"Result: 57-58% Accuracy. This 'linear ceiling' indicated that simple feature overlap is insufficient."*

**B. The Winning Model (Siamese)**
- **Accuracy**: 90.99%
- **ROC-AUC**: 0.9941
- **Interpretation**: The massive jump (58% -> 91%) confirms that the relationship between stylistic features is highly non-linear. The neural network successfully learned to ignore topic keywords and focus on structural style.

**C. Resilience to Attack (Erosion)**
- *"To test robustness, we subjected the model to an adversarial attack using a T5-based paraphraser."*
- **Result**: "The mean probability of 'Same Author' dropped by **0.497** (Erosion). The attack successfully flipped **50%** of high-confidence predictions."
- **Implication**: "While our model is highly accurate on human-written text, it is vulnerable to AI-obfuscation. The 'Stylometric Fingerprint' is fragile."

**D. Visualizations (Include these in your report!)**
- **Erosion Plot (`erosion_plot.png`)**: Shows the dramatic probability drop after paraphrasing (Green dots -> Red dots).
- **ROC Curve (`roc_curve.png`)**: Shows near-perfect separation (AUC=0.99).
- **Confusion Matrix (`confusion_matrix.png`)**: Shows balanced performance on both "Same" and "Different" pairs.
- **Learning Curves (`training_curves.png`)**: Shows steady convergence without massive overfitting (Validation accuracy tracks Training accuracy well).

## 5. Conclusion
*"We demonstrated that Deep Metric Learning (Siamese Networks) is superior to traditional statistical stylometry for short-text authorship verification. By moving from engineered distance metrics (Cosine) to learned metrics, we improved accuracy from 58% to 91% on the PAN22 benchmark."*
