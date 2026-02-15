"""
Evaluate All Models Against Synonym Replacement Attacks
=========================================================
Loads synonym-attacked adversarial samples and evaluates ASR
for all models, comparing with T5 paraphrase attacks.

Usage:
    python experiments/eval_synonym_attacks.py
"""
import sys
import os
import json
import numpy as np
import pickle
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.dann import DANNSiameseV3
from experiments.train_siamese import SiameseNetwork, preprocess
from utils.feature_extraction import EnhancedFeatureExtractor

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

# Paths
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
SYNONYM_CACHE = 'data/synonym_adversarial_cache.jsonl'
T5_CACHE = 'data/eval_adversarial_cache.jsonl'


def get_feats(extractor, texts):
    f_dict = extractor.transform(texts)
    return np.hstack([f_dict['char'], f_dict['pos'], f_dict['lex'], f_dict['readability']])


def load_adversarial_cache(cache_path):
    """Load adversarial cache (anchor, positive, attacked)."""
    samples = []
    with open(cache_path) as f:
        for line in f:
            entry = json.loads(line)
            samples.append(entry)
    return samples


def compute_asr(model_cfg, extractor, anchors, positives, attacked_texts, model_type='siamese'):
    """
    Compute Attack Success Rate.
    ASR = fraction of positive pairs where model flips from "same author" to "different author"
    after the attack.
    """
    BATCH = 32
    clean_correct = 0
    attacked_fooled = 0
    total = 0

    for i in range(0, len(anchors), BATCH):
        a = anchors[i:i+BATCH]
        p = positives[i:i+BATCH]
        atk = attacked_texts[i:i+BATCH]

        if model_type == 'dann':
            f_a = get_feats(extractor, a)
            f_p = get_feats(extractor, p)
            f_atk = get_feats(extractor, atk)

            x_a = torch.tensor(f_a, dtype=torch.float32).to(DEVICE)
            x_p = torch.tensor(f_p, dtype=torch.float32).to(DEVICE)
            x_atk = torch.tensor(f_atk, dtype=torch.float32).to(DEVICE)

            with torch.no_grad():
                clean_probs, _, _ = model_cfg['model'](x_a, x_p, alpha=0.0)
                atk_probs, _, _ = model_cfg['model'](x_a, x_atk, alpha=0.0)
            clean_probs = clean_probs.cpu().numpy().reshape(-1)
            atk_probs = atk_probs.cpu().numpy().reshape(-1)

        elif model_type == 'siamese':
            vec = model_cfg['vec']
            scaler = model_cfg['scaler']

            pa = [preprocess(t) for t in a]
            pp = [preprocess(t) for t in p]
            patk = [preprocess(t) for t in atk]

            va = scaler.transform(vec.transform(pa).toarray())
            vp = scaler.transform(vec.transform(pp).toarray())
            vatk = scaler.transform(vec.transform(patk).toarray())

            xa = torch.tensor(va, dtype=torch.float32).to(DEVICE)
            xp = torch.tensor(vp, dtype=torch.float32).to(DEVICE)
            xatk = torch.tensor(vatk, dtype=torch.float32).to(DEVICE)

            with torch.no_grad():
                clean_logits = model_cfg['model'](xa, xp)
                atk_logits = model_cfg['model'](xa, xatk)
                clean_probs = torch.sigmoid(clean_logits).cpu().numpy().reshape(-1)
                atk_probs = torch.sigmoid(atk_logits).cpu().numpy().reshape(-1)

        elif model_type == 'ensemble':
            clean_probs_list = []
            atk_probs_list = []
            for k in range(len(a)):
                cp = model_cfg['model'].predict(a[k], p[k])
                ap = model_cfg['model'].predict(a[k], atk[k])
                clean_probs_list.append(cp)
                atk_probs_list.append(ap)
            clean_probs = np.array(clean_probs_list)
            atk_probs = np.array(atk_probs_list)

        # Count: clean prediction "same author" (>0.5) AND attack flips to "different" (<=0.5)
        for j in range(len(clean_probs)):
            if clean_probs[j] > 0.5:  # Model correctly says "same author"
                clean_correct += 1
                if atk_probs[j] <= 0.5:  # Attack successfully fools the model
                    attacked_fooled += 1
            total += 1

    asr = attacked_fooled / max(clean_correct, 1)
    return {
        'asr': round(asr, 4),
        'total': total,
        'clean_correct': clean_correct,
        'attacked_fooled': attacked_fooled,
    }


def main():
    print("=" * 60)
    print("SYNONYM ATTACK EVALUATION")
    print("=" * 60)

    # Load models
    print("\n[Loading Models]")
    extractor = pickle.load(open(EXTRACTOR_PATH, 'rb'))
    models = {}

    for name, cfg in MODEL_CONFIGS.items():
        if not os.path.exists(cfg['model_path']):
            print(f"  ✗ {name} not found")
            continue

        try:
            if cfg['type'] == 'dann':
                model = DANNSiameseV3(input_dim=4308, num_domains=4).to(DEVICE)
                model.load_state_dict(torch.load(cfg['model_path'], map_location=DEVICE))
                model.eval()
                models[name] = {'model': model, 'type': 'dann'}

            elif cfg['type'] == 'siamese':
                vec = pickle.load(open(cfg['vec_path'], 'rb'))
                scaler = pickle.load(open(cfg['scaler_path'], 'rb'))
                model = SiameseNetwork(input_dim=cfg['input_dim']).to(DEVICE)
                model.load_state_dict(torch.load(cfg['model_path'], map_location=DEVICE))
                model.eval()
                models[name] = {'model': model, 'type': 'siamese', 'vec': vec, 'scaler': scaler}

            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    # Try ensemble
    try:
        from models.robust_ensemble import RobustMultiExpertEnsemble
        ensemble = RobustMultiExpertEnsemble()
        models['Ensemble'] = {'model': ensemble, 'type': 'ensemble'}
        print(f"  ✓ Ensemble")
    except Exception as e:
        print(f"  ✗ Ensemble: {e}")

    # Load adversarial caches
    results = {}

    for attack_name, cache_path in [('Synonym', SYNONYM_CACHE), ('T5 Paraphrase', T5_CACHE)]:
        if not os.path.exists(cache_path):
            print(f"\n  ⚠ {attack_name} cache not found at {cache_path}")
            continue

        samples = load_adversarial_cache(cache_path)
        anchors = [s['anchor'] for s in samples]
        positives = [s['positive'] for s in samples]
        attacked = [s['attacked'] for s in samples]

        print(f"\n{'='*60}")
        print(f"  {attack_name} Attack ({len(samples)} samples)")
        print(f"{'='*60}")

        attack_results = {}
        for name, cfg in models.items():
            print(f"\n  Evaluating {name}...", end=" ")
            asr_data = compute_asr(cfg, extractor, anchors, positives, attacked, cfg['type'])
            attack_results[name] = asr_data
            print(f"ASR={asr_data['asr']:.1%} "
                  f"(fooled {asr_data['attacked_fooled']}/{asr_data['clean_correct']})")

        results[attack_name] = attack_results

    # Save results
    output_path = "results/synonym_attack_results.json"
    os.makedirs("results", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print comparison table
    print(f"\n{'='*70}")
    print("COMPARISON TABLE: T5 Paraphrase vs Synonym Replacement ASR")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'T5 ASR':>10} {'Synonym ASR':>12} {'Δ ASR':>10}")
    print("-" * 52)

    for name in models.keys():
        t5_asr = results.get('T5 Paraphrase', {}).get(name, {}).get('asr', None)
        syn_asr = results.get('Synonym', {}).get(name, {}).get('asr', None)

        t5_str = f"{t5_asr:.1%}" if t5_asr is not None else "N/A"
        syn_str = f"{syn_asr:.1%}" if syn_asr is not None else "N/A"

        if t5_asr is not None and syn_asr is not None:
            delta = syn_asr - t5_asr
            delta_str = f"{delta:+.1%}"
        else:
            delta_str = "N/A"

        print(f"{name:<20} {t5_str:>10} {syn_str:>12} {delta_str:>10}")

    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    main()
