---
description: How to run the SCIE-grade robustness training pipeline
---

# SCIE Robustness Sprint - Run Order

All commands should be run from the project root: `/Users/aarushisinha/Desktop/college/sem8/btp/stylometry`

## Step 1: Train Cross-Domain Siamese (~3 min)
// turbo
```bash
./venv/bin/python experiments/train_siamese_crossdomain.py
```
Target: Val Acc > 70%, ROC > 0.75

## Step 2: Retrain Base DANN (~30 min)
This is the longest step. It needs 80 epochs with 4000 samples/domain.
```bash
./venv/bin/python experiments/train_dann.py
```
Target: Avg validation accuracy > 65% across PAN22, Blog, Enron

## Step 3: Train Robust DANN (~10 min)
Must be run AFTER Step 2 completes (depends on `results/final_dann/dann_model_v4.pth`).
```bash
./venv/bin/python experiments/train_robust.py
```
Target: Val accuracy drop < 5% from base DANN

## Step 4: Run Full Evaluation (~15 min, includes T5 attack generation)
Must be run AFTER all models are trained.
```bash
./venv/bin/python experiments/eval_robust_all.py
```
Results saved to `results/final_robustness_metrics.json`

## Step 5: Review Results
// turbo
```bash
cat results/final_robustness_metrics.json | python3 -m json.tool
```

## Notes
- If DANN training (~30 min) is too slow, you can reduce `EPOCHS` in `experiments/train_dann.py` to 40
- The extractor only needs to be re-fitted if you delete `results/final_dann/extractor.pkl`
- Adversarial eval results are cached in `data/eval_adversarial_cache.jsonl` - delete to regenerate
