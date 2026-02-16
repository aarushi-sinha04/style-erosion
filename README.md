# From Characters to Syntax: Characterizing the Accuracy–Robustness Trade-off in Cross-Domain Authorship Verification


## Overview

This repository accompanies the paper investigating the **fundamental trade-off between accuracy and adversarial robustness** in authorship verification (AV) systems. We evaluate eight models spanning three feature families — fine-grained character n-grams, coarse-grained syntactic features, and contextual embeddings — across three text domains under semantic-preserving paraphrase attacks.

### Key Findings

| | Best Accuracy | Best Robustness |
|---|---|---|
| **Model** | Robust Siamese | Robust DANN |
| **Features** | Char 4-grams | Multi-view syntactic |
| **Avg Accuracy** | 86.2% | 60.4% |
| **Attack Success Rate** | 74.0% | 7.7% |

- **Feature granularity** — not model architecture — determines a system's position on the accuracy–robustness frontier.
- **Adversarial training paradox:** Improves accuracy (+5.6 pp) but *increases* vulnerability (+30 pp ASR) by deepening reliance on fragile character-level features.
- **Attack validity confirmed:** BERTScore F1 = 0.885 across T5 paraphrase, synonym replacement, and back-translation attacks.

## Repository Structure

```
stylometry/
├── attacks/              # Adversarial attack implementations
│   ├── back_translation.py
│   ├── gradient_attack.py
│   ├── llm_impersonator.py
│   ├── synonym_attack.py
│   └── t5_paraphraser.py
│
├── data/                 # Adversarial caches & training data
│   └── raw/              # Raw datasets (gitignored, see below)
│
├── evaluation/           # Statistical significance tests
├── experiments/          # All training & evaluation scripts
├── figures/              # Publication figures & generation scripts
├── models/               # Model architecture definitions (PyTorch)
├── results/              # Experiment result JSONs
│   └── checkpoints/      # Trained model weights (gitignored)
├── utils/                # Shared utilities (features, data loading, metrics)
│
├── inference.py                    # Run inference with trained models
├── SCIE_PAPER_STRUCTURE_GUIDE.md   # Paper writing guide with all results
├── EXPERIMENT_LOG.md               # Comprehensive experiment log
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended for training; CPU works for inference)

### Installation

```bash
git clone https://github.com/aarushi-sinha04/style-erosion.git
cd style-erosion

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Datasets

Download the raw datasets and place them in `data/raw/`:

| Dataset | Source | File |
|---|---|---|
| PAN22 | [PAN 2022 AV Task](https://pan.webis.de/clef22/pan22-web/authorship-verification.html) | `pan22_texts.jsonl`, `pan22_labels.jsonl` |
| BlogText | [Schler et al., 2006](https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm) | `blogtext.csv` |
| Enron | [Klimt & Yang, 2004](https://www.cs.cmu.edu/~enron/) | `emails.csv` |
| IMDB | [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) | `imdb.csv` |

## Models

| Model | Architecture | Features | Training Script |
|---|---|---|---|
| Logistic Regression | Linear baseline | Char 3-grams | `experiments/eval_baselines.py` |
| PAN22 Siamese | Siamese network | Char 4-grams (3K) | `experiments/train_siamese.py` |
| CD Siamese | Siamese network | Char 4-grams (5K) | `experiments/train_siamese_crossdomain.py` |
| Robust Siamese | Siamese + adversarial training | Char 4-grams (5K) | `experiments/train_robust_siamese.py` |
| Base DANN | Domain-adversarial network | Multi-view (4,308-d) | `experiments/train_dann.py` |
| Robust DANN | DANN + adversarial training | Multi-view (4,308-d) | `experiments/train_robust.py` |
| BERT Siamese | BERT fine-tuned | Contextual embeddings | `experiments/train_bert_baseline.py` |
| Ensemble | Confidence-weighted hybrid | Char + syntactic | — |

## Attacks

Three semantic-preserving adversarial attacks implemented:

1. **T5 Paraphrase** (`attacks/t5_paraphraser.py`) — Sequence-to-sequence rewriting using `humarin/chatgpt_paraphraser_on_T5_base`
2. **Synonym Replacement** (`attacks/synonym_attack.py`) — WordNet-based word substitution with POS-aware filtering
3. **Back-Translation** (`attacks/back_translation.py`) — English → German → English via MarianMT

## Evaluation

```bash
# Evaluate robustness across all attacks
python experiments/eval_robust_all.py

# Run syntactic feature ablations
python experiments/eval_ablations.py

# Generate publication figures
python figures/generate_paper_figures.py
```

## Results

All experiment results are stored as JSON in `results/`:

| File | Contents |
|---|---|
| `final_robustness_metrics.json` | Main results: accuracy, AUC, F1, ASR for all models |
| `syntactic_ablations.json` | Feature decomposition ablation (POS / function words / readability) |
| `synonym_attack_results.json` | Synonym replacement ASR across all models |
| `backtranslation_attack.json` | Back-translation ASR across all models |
| `bert_attack_results.json` | BERT Siamese robustness under all 3 attacks |
| `statistical_tests.json` | McNemar & bootstrap significance tests |
| `error_analysis.json` | Per-domain error breakdown (FP/FN analysis) |
| `qualitative_error_analysis.json` | Representative error examples with text snippets |
| `bertscore.json` | Semantic preservation validation |
| `baseline_results.json` | Logistic Regression baseline accuracy & ASR |

## Figures

All publication figures are in `figures/` and can be regenerated:

```bash
python figures/generate_paper_figures.py
```

| Figure | File | Description |
|---|---|---|
| Fig 1 | `fig1_tradeoff.png` | Accuracy–Robustness frontier scatter (8 models) |
| Fig 2 | `fig2_accuracy_bars.png` | Cross-domain accuracy grouped bars |
| Fig 3 | `fig3_asr_comparison.png` | Multi-attack ASR heatmap (T5 / Synonym / BackTrans) |
| Fig 4 | `fig4_ablation.png` | Siamese ablation: accuracy progression + ASR paradox |
| Fig 5 | `fig5_error_analysis.png` | Error analysis: domain errors, accuracy, confidence |
| Fig 6 | `fig6_syntactic_ablation.png` | Syntactic feature decomposition (POS / function words / readability) |
| Fig 7 | `fig7_attack_granularity.png` | Attack granularity hierarchy (word → sentence) |


## License

This project is for academic research purposes. Please contact the authors for commercial use.
