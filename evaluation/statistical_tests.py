"""
Statistical Significance Tests for SCIE Publication
=====================================================
Computes:
1. Bootstrap Confidence Intervals (95%, 1000 resamples) for accuracy, F1, AUC
2. McNemar's test for pairwise model comparisons
3. Per-domain and aggregate results

Usage:
    python -m evaluation.statistical_tests
"""
import sys
import os
import json
import numpy as np
import pickle
from itertools import combinations
from scipy import stats

import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.dann import DANNSiameseV3
from utils.data_loader_scie import PAN22Loader, BlogTextLoader, EnronLoader
from experiments.train_siamese import SiameseNetwork, preprocess

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

# Model paths (same as eval_robust_all.py)
MODEL_CONFIGS = {
    'Base DANN': {
        'type': 'dann',
        'model_path': 'results/final_dann/dann_model_v4.pth',
    },
    'Robust DANN': {
        'type': 'dann',
        'model_path': 'results/robust_dann/robust_dann_model.pth',
    },
    'PAN22 Siamese': {
        'type': 'siamese',
        'model_path': 'results/siamese_baseline/best_model.pth',
        'vec_path': 'results/siamese_baseline/vectorizer.pkl',
        'scaler_path': 'results/siamese_baseline/scaler.pkl',
        'input_dim': 3000,
    },
    'CD Siamese': {
        'type': 'siamese',
        'model_path': 'results/siamese_crossdomain/best_model.pth',
        'vec_path': 'results/siamese_crossdomain/vectorizer.pkl',
        'scaler_path': 'results/siamese_crossdomain/scaler.pkl',
        'input_dim': 5000,
    },
    'Rob Siamese': {
        'type': 'siamese',
        'model_path': 'results/robust_siamese/best_model.pth',
        'vec_path': 'results/robust_siamese/vectorizer.pkl',
        'scaler_path': 'results/robust_siamese/scaler.pkl',
        'input_dim': 5000,
    },
}

EXTRACTOR_PATH = 'results/final_dann/extractor.pkl'
N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95
SEED = 42


# ==============================================================================
# Helper Functions
# ==============================================================================

def get_feats(extractor, texts):
    f_dict = extractor.transform(texts)
    return np.hstack([f_dict['char'], f_dict['pos'], f_dict['lex'], f_dict['readability']])


def load_models():
    """Load all models and return configs dict with loaded objects."""
    print("[1/4] Loading models...")
    extractor = pickle.load(open(EXTRACTOR_PATH, 'rb'))
    loaded = {}

    for name, cfg in MODEL_CONFIGS.items():
        if not os.path.exists(cfg['model_path']):
            print(f"  ✗ {name} not found at {cfg['model_path']}")
            continue

        try:
            if cfg['type'] == 'dann':
                model = DANNSiameseV3(input_dim=4308, num_domains=4).to(DEVICE)
                model.load_state_dict(torch.load(cfg['model_path'], map_location=DEVICE))
                model.eval()
                loaded[name] = {'model': model, 'type': 'dann'}

            elif cfg['type'] == 'siamese':
                vec = pickle.load(open(cfg['vec_path'], 'rb'))
                scaler = pickle.load(open(cfg['scaler_path'], 'rb'))
                model = SiameseNetwork(input_dim=cfg['input_dim']).to(DEVICE)
                model.load_state_dict(torch.load(cfg['model_path'], map_location=DEVICE))
                model.eval()
                loaded[name] = {'model': model, 'type': 'siamese', 'vec': vec, 'scaler': scaler}

            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    # Try ensemble
    try:
        from models.robust_ensemble import RobustMultiExpertEnsemble
        ensemble = RobustMultiExpertEnsemble()
        loaded['Ensemble'] = {'model': ensemble, 'type': 'ensemble'}
        print(f"  ✓ Ensemble")
    except Exception as e:
        print(f"  ✗ Ensemble: {e}")

    return loaded, extractor


def predict_batch(cfg, extractor, t1, t2, domain='pan22'):
    """Get predictions from a model config."""
    BATCH_SIZE = 64
    all_probs = []

    for i in range(0, len(t1), BATCH_SIZE):
        bt1 = t1[i:i+BATCH_SIZE]
        bt2 = t2[i:i+BATCH_SIZE]

        if cfg['type'] == 'dann':
            f1 = get_feats(extractor, bt1)
            f2 = get_feats(extractor, bt2)
            x1 = torch.tensor(f1, dtype=torch.float32).to(DEVICE)
            x2 = torch.tensor(f2, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                p, _, _ = cfg['model'](x1, x2, alpha=0.0)
            probs = p.cpu().numpy().reshape(-1)

        elif cfg['type'] == 'siamese':
            p_t1 = [preprocess(t) for t in bt1]
            p_t2 = [preprocess(t) for t in bt2]
            v1 = cfg['scaler'].transform(cfg['vec'].transform(p_t1).toarray())
            v2 = cfg['scaler'].transform(cfg['vec'].transform(p_t2).toarray())
            x1 = torch.tensor(v1, dtype=torch.float32).to(DEVICE)
            x2 = torch.tensor(v2, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                logits = cfg['model'](x1, x2)
                probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

        elif cfg['type'] == 'ensemble':
            probs = []
            for k in range(len(bt1)):
                p = cfg['model'].predict(bt1[k], bt2[k], domain=domain.lower())
                probs.append(p)
            probs = np.array(probs)

        all_probs.extend(probs.flatten().tolist())

    return np.array(all_probs)


# ==============================================================================
# Statistical Tests
# ==============================================================================

def bootstrap_ci(y_true, y_prob, metric_fn, n_bootstrap=N_BOOTSTRAP,
                 confidence=CONFIDENCE_LEVEL, seed=SEED):
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        y_true: true labels (0/1)
        y_prob: predicted probabilities
        metric_fn: function(y_true, y_pred_or_prob) -> float
        n_bootstrap: number of resamples
        confidence: confidence level (e.g. 0.95)
        seed: random seed

    Returns:
        dict with 'mean', 'lower', 'upper', 'std'
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        y_t = y_true[idx]
        y_p = y_prob[idx]

        # Skip if only one class in bootstrap sample
        if len(np.unique(y_t)) < 2:
            continue

        try:
            s = metric_fn(y_t, y_p)
            scores.append(s)
        except Exception:
            continue

    if not scores:
        return {'mean': float('nan'), 'lower': float('nan'),
                'upper': float('nan'), 'std': float('nan')}

    scores = np.array(scores)
    alpha = 1 - confidence
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))

    return {
        'mean': round(float(np.mean(scores)), 4),
        'lower': round(float(lower), 4),
        'upper': round(float(upper), 4),
        'std': round(float(np.std(scores)), 4),
    }


def mcnemar_test(y_true, preds_a, preds_b):
    """
    McNemar's test comparing two models' predictions.

    Tests whether the two models make significantly different errors.

    Returns:
        dict with 'b' (A right, B wrong), 'c' (A wrong, B right),
        'chi2', 'p_value', 'significant' (at α=0.05)
    """
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)

    # b: A correct, B incorrect
    b = int(np.sum(correct_a & ~correct_b))
    # c: A incorrect, B correct
    c = int(np.sum(~correct_a & correct_b))

    # McNemar's chi-squared with continuity correction
    if b + c == 0:
        return {
            'b': b, 'c': c,
            'chi2': 0.0, 'p_value': 1.0,
            'significant': False
        }

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {
        'b': b, 'c': c,
        'chi2': round(float(chi2), 4),
        'p_value': round(float(p_value), 6),
        'significant': bool(p_value < 0.05)
    }


# ==============================================================================
# Main
# ==============================================================================

def run_statistical_tests():
    """Run all statistical tests and save results."""
    np.random.seed(SEED)

    # Load models
    models, extractor = load_models()
    if not models:
        print("No models loaded! Exiting.")
        return

    # Domain loaders
    domain_loaders = {
        'PAN22': PAN22Loader("pan22-authorship-verification-training.jsonl",
                             "pan22-authorship-verification-training-truth.jsonl"),
        'Blog': BlogTextLoader("blogtext.csv"),
        'Enron': EnronLoader("emails.csv"),
    }

    results = {
        'bootstrap_ci': {},
        'mcnemar': {},
        'config': {
            'n_bootstrap': N_BOOTSTRAP,
            'confidence_level': CONFIDENCE_LEVEL,
            'seed': SEED,
            'models': list(models.keys()),
            'domains': list(domain_loaders.keys()),
        }
    }

    # ==========================
    # Phase 1: Get all predictions
    # ==========================
    print("\n[2/4] Generating predictions for all models × domains...")
    all_predictions = {}  # {(model, domain): {'probs': np.array, 'labels': np.array}}

    for domain, loader in domain_loaders.items():
        print(f"\n  Loading {domain}...")
        loader.load(limit=6000)
        t1, t2, labels = loader.create_pairs(num_pairs=500)

        if not t1:
            print(f"    No pairs for {domain}")
            continue

        labels = np.array(labels)
        valid_mask = labels != -1
        labels = labels[valid_mask].astype(int)
        t1 = [t1[i] for i in range(len(t1)) if valid_mask[i]]
        t2 = [t2[i] for i in range(len(t2)) if valid_mask[i]]

        print(f"    {len(labels)} valid pairs")

        for model_name, cfg in models.items():
            print(f"    Predicting: {model_name}...", end=" ")
            probs = predict_batch(cfg, extractor, t1, t2, domain=domain)
            probs = probs[valid_mask[:len(probs)]] if len(probs) > len(labels) else probs[:len(labels)]
            all_predictions[(model_name, domain)] = {
                'probs': probs,
                'labels': labels,
            }
            acc = accuracy_score(labels, (probs > 0.5).astype(int))
            print(f"Acc={acc:.3f}")

    # ==========================
    # Phase 2: Bootstrap CIs
    # ==========================
    print("\n[3/4] Computing bootstrap confidence intervals...")

    for (model_name, domain), data in all_predictions.items():
        y_true = data['labels']
        y_prob = data['probs']
        y_pred = (y_prob > 0.5).astype(int)

        key = f"{model_name}|{domain}"

        # Accuracy CI
        acc_ci = bootstrap_ci(y_true, y_pred,
                              lambda yt, yp: accuracy_score(yt, yp))

        # F1 CI
        f1_ci = bootstrap_ci(y_true, y_pred,
                             lambda yt, yp: f1_score(yt, yp, zero_division=0))

        # AUC CI (uses probabilities, not binary predictions)
        auc_ci = bootstrap_ci(y_true, y_prob,
                              lambda yt, yp: roc_auc_score(yt, yp))

        results['bootstrap_ci'][key] = {
            'model': model_name,
            'domain': domain,
            'n_samples': len(y_true),
            'accuracy': acc_ci,
            'f1': f1_ci,
            'auc': auc_ci,
        }
        print(f"  {model_name} on {domain}: "
              f"Acc={acc_ci['mean']:.3f} [{acc_ci['lower']:.3f}, {acc_ci['upper']:.3f}]")

    # ==========================
    # Phase 3: McNemar's Tests
    # ==========================
    print("\n[4/4] Running McNemar's tests (pairwise model comparisons)...")

    model_names = list(models.keys())
    for domain in domain_loaders.keys():
        domain_results = {}

        for model_a, model_b in combinations(model_names, 2):
            key_a = (model_a, domain)
            key_b = (model_b, domain)

            if key_a not in all_predictions or key_b not in all_predictions:
                continue

            y_true = all_predictions[key_a]['labels']
            preds_a = (all_predictions[key_a]['probs'] > 0.5).astype(int)
            preds_b = (all_predictions[key_b]['probs'] > 0.5).astype(int)

            # Ensure same length
            min_len = min(len(preds_a), len(preds_b), len(y_true))
            result = mcnemar_test(y_true[:min_len], preds_a[:min_len], preds_b[:min_len])

            pair_key = f"{model_a} vs {model_b}"
            domain_results[pair_key] = result

            sig = "***" if result['significant'] else "   "
            print(f"  {domain} | {pair_key:40s} | "
                  f"χ²={result['chi2']:7.2f} | p={result['p_value']:.4f} {sig}")

        results['mcnemar'][domain] = domain_results

    # ==========================
    # Save Results
    # ==========================
    output_path = "results/statistical_tests.json"
    os.makedirs("results", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    print(f"{'='*60}")

    # Print summary table for paper
    print("\n" + "="*80)
    print("PAPER TABLE: Model Performance with 95% CI")
    print("="*80)
    print(f"{'Model':<18} {'Domain':<8} {'Accuracy':<24} {'F1':<24} {'AUC':<24}")
    print("-"*80)
    for key, data in sorted(results['bootstrap_ci'].items()):
        m = data['model']
        d = data['domain']
        a = data['accuracy']
        f = data['f1']
        u = data['auc']
        print(f"{m:<18} {d:<8} "
              f"{a['mean']:.3f} [{a['lower']:.3f},{a['upper']:.3f}]  "
              f"{f['mean']:.3f} [{f['lower']:.3f},{f['upper']:.3f}]  "
              f"{u['mean']:.3f} [{u['lower']:.3f},{u['upper']:.3f}]")

    # Print significant McNemar findings
    print("\n" + "="*80)
    print("SIGNIFICANT PAIRWISE DIFFERENCES (McNemar's test, p < 0.05)")
    print("="*80)
    for domain, pairs in results['mcnemar'].items():
        sig_pairs = {k: v for k, v in pairs.items() if v['significant']}
        if sig_pairs:
            print(f"\n  {domain}:")
            for pair, res in sig_pairs.items():
                print(f"    {pair}: χ²={res['chi2']:.2f}, p={res['p_value']:.6f}")

    return results


if __name__ == "__main__":
    run_statistical_tests()
