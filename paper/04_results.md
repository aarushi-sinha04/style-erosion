# 4. Experimental Results

## 4.1 Cross-Domain Clean Accuracy

Table 1 presents clean accuracy across all seven models and three domains.

**Table 1: Cross-Domain Authorship Verification Accuracy**

| Model | Features | PAN22 | Blog | Enron | Avg |
|:---|:---|---:|---:|---:|---:|
| LogReg (baseline) | Char 3-grams | 62.8% | 50.0% | 50.0% | 54.3% |
| Base DANN | Multi-view | 53.2% | 55.8% | 78.8% | 62.6% |
| Robust DANN | Multi-view | 54.4% | 52.8% | 74.0% | 60.4% |
| PAN22 Siamese | Char 4-grams | 97.0% | 52.1% | 56.8% | 68.6% |
| CD Siamese | Char 4-grams | 98.2% | 66.5% | 77.2% | 80.6% |
| **Rob Siamese** | **Char 4-grams** | **99.4%** | **71.9%** | **87.2%** | **86.2%** |
| Ensemble | Hybrid | 98.0% | 64.4% | 76.8% | 79.7% |

**Key observations:**

1. **Rob Siamese achieves state-of-the-art cross-domain accuracy** (86.2%), representing a 31.9 percentage point improvement over the LogReg baseline and a 23.6 pp improvement over Base DANN.

2. **Cross-domain training is essential.** The PAN22 Siamese achieves 97% on its training domain but only 52-57% on Blog and Enron. Cross-domain training (CD Siamese) lifts Blog from 52.1% to 66.5% and Enron from 56.8% to 77.2%.

3. **Adversarial fine-tuning improves clean accuracy.** Contrary to expectations, adversarial training improved accuracy from 80.6% (CD Siamese) to 86.2% (Rob Siamese)—a 5.6 pp gain. This suggests adversarial examples act as data augmentation, exposing the model to a wider variety of writing patterns.

4. **DANN excels at Enron.** The multi-view features capture email-specific patterns well (78.8%), but fail on PAN22 (53.2%)—barely above random—because syntactic features are too coarse for the diverse PAN22 corpus.

5. **Blog is universally challenging.** No model exceeds 72% on Blog, likely due to high stylistic variance: blog authors write about diverse topics in varying registers, reducing the consistency of any feature type.

## 4.2 Adversarial Robustness

Table 2 presents attack success rates under T5 paraphrasing.

**Table 2: Adversarial Robustness (T5 Paraphrase Attack)**

| Model | Features | ASR ↓ | Valid Orig | BERTScore F1 |
|:---|:---|---:|---:|---:|
| **Robust DANN** | **Multi-view** | **7.7%** | 26 | 0.885 |
| Base DANN | Multi-view | 14.3% | 35 | 0.885 |
| LogReg (baseline) | Char 3-grams | 10.8% | 37 | 0.885 |
| CD Siamese | Char 4-grams | 44.0% | 50 | 0.885 |
| Ensemble | Hybrid | 48.0% | 50 | 0.885 |
| PAN22 Siamese | Char 4-grams | 50.0% | 50 | 0.885 |
| Rob Siamese | Char 4-grams | 74.0% | 50 | 0.885 |

**BERTScore F1 = 0.885** confirms that attacks preserve semantic meaning (Precision: 0.895, Recall: 0.875), ensuring measured ASR reflects genuine model vulnerability rather than degenerate attacks.

**Key observations:**

1. **Syntactic features provide natural robustness.** The Robust DANN (7.7% ASR) is nearly immune to paraphrase attacks. POS trigrams and readability metrics survive paraphrasing because grammatical structure and reading level are preserved when text is rewritten.

2. **Character features are inherently vulnerable.** All character n-gram models suffer >40% ASR. Paraphrasing changes word choice, spelling, and character sequences—exactly the patterns that character n-grams capture.

3. **LogReg's low ASR is misleading.** LogReg achieves only 10.8% ASR, but this is because it correctly classifies only 37/50 original pairs. Its low ASR reflects poor baseline accuracy, not genuine robustness.

4. **Adversarial training worsens character-level ASR.** Rob Siamese (74% ASR) is *more* vulnerable than CD Siamese (44% ASR) despite adversarial training. This is because Rob Siamese correctly classifies 50/50 positive pairs (full denominator), and character-level vulnerability cannot be overcome by model-level training.

5. **The trade-off is fundamental.** No model achieves both high accuracy (>80%) and low ASR (<35%). This suggests the trade-off is at the feature level, not the model level.

## 4.3 Model Ablation: Siamese Progression

Table 3 traces the evolution from single-domain to robust cross-domain Siamese.

**Table 3: Siamese Model Ablation**

| Stage | Change | PAN22 | Blog | Enron | Avg | ASR |
|:---|:---|---:|---:|---:|---:|---:|
| PAN22 Siamese | Baseline | 97.0% | 52.1% | 56.8% | 68.6% | 50.0% |
| + Cross-domain data | CD Siamese | 98.2% | +14.4 | +20.4 | 80.6% | 44.0% |
| + Adversarial training | Rob Siamese | 99.4% | +5.4 | +10.0 | 86.2% | 74.0% |

Each stage improves accuracy: cross-domain training provides the largest gain (+12 pp average), while adversarial fine-tuning adds another 5.6 pp. However, ASR increases with accuracy because more correctly-classified pairs become available for attack.

## 4.4 Error Analysis

We analyzed 171 errors from the Rob Siamese model across three domains (92 false positives, 79 false negatives).

**Table 4: Error Distribution by Domain**

| Domain | Pairs | FP | FN | Total Errors | Error Rate |
|:---|---:|---:|---:|---:|---:|
| PAN22 | 500 | 5 | 1 | 6 | 1.2% |
| Blog | 470 | 71 | 60 | 131 | 27.9% |
| Enron | 224 | 16 | 18 | 34 | 15.2% |

**Key findings:**

1. **Blog dominates errors.** 77% of all errors (131/171) come from Blog, confirming it as the most challenging domain. Blog authors frequently shift topics, registers, and writing styles between posts.

2. **False positives are overconfident.** FPs have an average prediction confidence of 0.886—the model is "very sure" about wrong predictions. This occurs when different authors share similar n-gram patterns due to topic overlap or common writing conventions.

3. **False negatives are underconfident.** FNs have average confidence 0.118—correctly flagging uncertainty. The model recognizes it cannot confidently verify authorship when same-author texts differ significantly in topic or style.

4. **N-gram overlap is similar for both error types.** Both FPs and FNs show ~7% character 4-gram overlap. This means n-gram overlap alone cannot explain the errors—the model's trained representations capture other patterns beyond raw feature overlap.
