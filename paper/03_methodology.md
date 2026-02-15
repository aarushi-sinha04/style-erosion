# 3. Methodology

## 3.1 Datasets

We evaluate on three publicly available datasets spanning distinct text domains:

**PAN22 Authorship Verification Corpus** (Bevendorff et al., 2022). The gold-standard benchmark for authorship verification, containing 24,000 fanfiction texts across diverse genres. Pairs are pre-constructed with same-author and different-author labels. Texts range from 200 to 10,000 words, with significant stylistic variation even within same-author pairs due to cross-discourse sampling.

**BlogText Corpus** (Schler et al., 2006). Contains blog posts from 19,320 authors. We use a subset of 10,847 posts from 277 authors. Blog texts are informal, short (median ~300 words), and exhibit high intra-author stylistic variance due to topic diversity. This dataset is the most challenging for verification.

**Enron Email Corpus** (Klimt & Yang, 2004). Contains 8,962 emails from corporate employees. Email style is domain-specific: short, formulaic, with organizational jargon. Same-author pairs are constructed from emails by the same sender; different-author pairs from different senders.

| Dataset | Texts | Authors | Avg Length | Domain |
|:---|---:|---:|---:|:---|
| PAN22 | 24,000 | ~6,000 | ~2,000 words | Fanfiction |
| BlogText | 10,847 | 277 | ~300 words | Blogs |
| Enron | 8,962 | ~150 | ~100 words | Corporate email |

## 3.2 Feature Representations

We investigate two fundamentally different feature types:

### 3.2.1 Character N-grams (Fine-Grained)

Character 4-grams capture subword writing patterns: character sequences, punctuation habits, spacing preferences, and typographical idiosyncrasies. We extract TF-IDF-weighted character 4-gram vectors with sublinear term frequency (top 5,000 features, min_df=3). These features are highly discriminative for authorship because they encode unconscious writing habits that authors cannot easily control.

**Hypothesis:** Character n-grams are vulnerable to paraphrase attacks because paraphrasing inherently changes the character-level surface form of text, even when preserving meaning.

### 3.2.2 Syntactic Features (Coarse-Grained)

Multi-view syntactic features include:
- **POS-tag trigrams:** Part-of-speech tag sequences capturing syntactic structure (1,000 features, extracted via spaCy).
- **Readability metrics:** Flesch-Kincaid Grade Level, Gunning Fog Index, Coleman-Liau Index, ARI (4 features via textstat).
- **Lexical features:** Function word frequencies (200 features from a closed-class word list).
- **Character n-grams:** Supplementary character 3-gram features (3,000 features).

Total: 4,308 features per text after concatenation.

**Hypothesis:** Syntactic features are robust to paraphrase attacks because paraphrasing preserves grammatical structure, reading level, and function word usage—even while transforming surface forms.

## 3.3 Model Architectures

### 3.3.1 Logistic Regression Baseline
Classical baseline using absolute feature differences |f(A) − f(B)| between text pairs with character 3-gram TF-IDF features (5,000 dimensions). L2-regularized logistic regression.

### 3.3.2 Siamese Neural Network
Twin-branch MLP with shared weights processing TF-IDF character 4-gram representations. Branch architecture: 5000→512→256 with ReLU, BatchNorm, and 30% dropout. Interaction head computes |u−v| (absolute difference) and u⊙v (element-wise product), concatenating into a 512-dimensional vector processed by a 512→256→1 classifier.

**Variants:**
- **PAN22 Siamese:** Trained on PAN22 only. Represents single-domain specialist.
- **CD Siamese:** Trained on PAN22 + BlogText + Enron combined (cross-domain generalization).
- **Rob Siamese:** CD Siamese adversarially fine-tuned with T5 paraphrase examples.

### 3.3.3 Domain-Adversarial Neural Network (DANN)
Multi-view feature extractor with gradient reversal for domain-invariant learning. Architecture: 4308→1024→512 shared encoder, authorship classifier with [u, v, |u−v|, u⊙v] interaction (2048→512→256→1), domain classifier with gradient reversal layer (512→256→4). Training uses curriculum learning: 5-epoch authorship-only warmup, then gradual GRL lambda increase to 0.3.

**Variants:**
- **Base DANN:** Trained with curriculum learning on all 4 domains.
- **Robust DANN:** Base DANN fine-tuned with adversarial consistency loss.

### 3.3.4 Ensemble
Confidence-weighted voting with domain-specific priors. Combines Siamese (character n-gram) and DANN (syntactic) predictions, weighting models based on prediction confidence and domain affinity.

## 3.4 Adversarial Attack Protocol

We generate adversarial examples using the T5-based paraphrase model (`humarin/chatgpt_paraphraser_on_T5_base`). For each positive pair (A, P) where texts share the same author:
1. Paraphrase P to produce P' while preserving meaning
2. Measure whether the model's prediction flips from "same author" to "different author"

**Attack Success Rate (ASR):** Among pairs correctly classified as same-author on original text, the fraction that flip to different-author after paraphrasing:

ASR = |{(A,P): f(A,P)=1 ∧ f(A,P')=0}| / |{(A,P): f(A,P)=1}|

**Semantic Preservation:** We measure BERTScore (Zhang et al., 2020) between P and P' to verify that attacks preserve meaning. Only semantically valid attacks (BERTScore F1 > 0.80) are considered.

## 3.5 Adversarial Training

The Rob Siamese is fine-tuned using mixed clean-adversarial training:
- **Clean loss:** Standard BCEWithLogitsLoss on original pairs from all 3 domains
- **Adversarial loss:** BCEWithLogitsLoss on (A, P') pairs where label remains "same author"
- **Consistency loss:** MSE between predictions on (A, P) and (A, P')

Total loss: L = L_clean + L_pos + λ_adv · L_adv + λ_cons · L_cons

with λ_adv = 0.3, λ_cons = 0.3, learning rate 2×10⁻⁵ (low for fine-tuning), gradient clipping at 1.0.

## 3.6 Evaluation Metrics

- **Accuracy:** Fraction of correctly classified pairs per domain
- **ROC-AUC:** Area under the receiver operating characteristic curve
- **F1 Score:** Harmonic mean of precision and recall
- **Attack Success Rate (ASR):** Fraction of successful adversarial attacks (lower = more robust)
- **BERTScore F1:** Semantic preservation of adversarial text (Zhang et al., 2020)
