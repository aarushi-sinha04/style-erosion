# 6. Conclusion

We presented the first systematic study of the accuracy–robustness trade-off in cross-domain authorship verification. By evaluating seven models spanning three feature representations across three text domains, we demonstrated a fundamental finding: **feature granularity determines a system's position on the accuracy–robustness frontier.**

Character 4-gram models achieve state-of-the-art cross-domain accuracy (86.2%) by capturing fine-grained authorial fingerprints, but are inherently vulnerable to paraphrase attacks (74% ASR) because text transformation destroys character-level patterns. Syntactic models using POS trigrams and readability metrics achieve vastly superior robustness (7.7% ASR) because these features are paraphrase-invariant, but at the cost of accuracy (60.4%) due to their coarser granularity.

Our Rob Siamese model—a cross-domain adversarially fine-tuned Siamese network—achieves the highest cross-domain accuracy reported on this multi-domain benchmark (99.4% PAN22, 87.2% Enron, 71.9% Blog). We further show that adversarial training improves clean accuracy by 5.6 percentage points through data augmentation effects, but cannot overcome feature-level vulnerability to text transformation.

These findings have direct practical implications. Forensic analysts requiring high accuracy should use character-based models while acknowledging their adversarial limitations. Systems deployed in adversarial environments (spam/sockpuppet detection) should prioritize syntactic features. Ensemble approaches offer a middle ground when both accuracy and robustness matter.

Our work establishes a clear research direction: developing hybrid feature representations that combine the discriminative power of character-level features with the robustness of syntactic features at the representation level rather than the model level.
