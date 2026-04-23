"""
Multi-Seed Evaluation
=====================
Runs the evaluation across 5 different random seeds to compute
mean ± std and effect sizes for statistical rigor.
"""
import sys
import os
import torch
import json
import numpy as np
import pickle
import random
from collections import defaultdict
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader_scie import PAN22Loader, BlogTextLoader, EnronLoader
from experiments.eval_robust_all import (
    load_dann_model, load_siamese_model, predict_dann, predict_siamese,
    BASE_DANN, ROBUST_DANN, EXTRACTOR, ADV_CACHE
)

PAN_SIAMESE = "results/checkpoints/siamese_baseline/best_model.pth"
PAN_VEC = "results/checkpoints/siamese_baseline/vectorizer.pkl"
PAN_SCALER = "results/checkpoints/siamese_baseline/scaler.pkl"
CD_SIAMESE = "results/checkpoints/cd_siamese/best_model.pth"
CD_VEC = "results/checkpoints/cd_siamese/vectorizer.pkl"
CD_SCALER = "results/checkpoints/cd_siamese/scaler.pkl"
ROB_SIAMESE = "results/checkpoints/robust_siamese/best_model.pth"
ROB_VEC = "results/checkpoints/robust_siamese/vectorizer.pkl"
ROB_SCALER = "results/checkpoints/robust_siamese/scaler.pkl"

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
SEEDS = [42, 123, 456, 789, 1024]

def cohens_d(x, y):
    """Compute Cohen's d effect size between two arrays of measurements."""
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2: return 0.0
    dof = nx + ny - 2
    pool_sd = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    if pool_sd == 0: return 0.0
    return (np.mean(x) - np.mean(y)) / pool_sd

def main():
    print("=" * 60)
    print("MULTI-SEED EVALUATION (5 Seeds)")
    print("=" * 60)

    try:
        extractor = pickle.load(open(EXTRACTOR, 'rb'))
    except FileNotFoundError:
        print(f"Warning: {EXTRACTOR} not found. DANN models will be skipped.")
        extractor = None
    models = {}
    
    # Load models
    print("\n[Loading Models]")
    if extractor is not None:
        models['Base DANN'] = {'model': load_dann_model(BASE_DANN, "Base DANN"), 'type': 'dann'}
        models['Robust DANN'] = {'model': load_dann_model(ROBUST_DANN, "Robust DANN"), 'type': 'dann'}
    else:
        print("  ✗ Skipping DANN models because extractor is missing")
    siam, sv, ss = load_siamese_model(PAN_SIAMESE, PAN_VEC, PAN_SCALER, 3000, "PAN22 Siamese")
    models['PAN22 Siamese'] = {'model': siam, 'type': 'siamese', 'vec': sv, 'scaler': ss}
    cd_siam, cv, cs = load_siamese_model(CD_SIAMESE, CD_VEC, CD_SCALER, 5000, "CD Siamese")
    models['CD Siamese'] = {'model': cd_siam, 'type': 'siamese', 'vec': cv, 'scaler': cs}
    rob_siam, rv, rs = load_siamese_model(ROB_SIAMESE, ROB_VEC, ROB_SCALER, 5000, "Rob Siamese")
    models['Rob Siamese'] = {'model': rob_siam, 'type': 'siamese', 'vec': rv, 'scaler': rs}

    # Verify all models loaded
    models = {k: v for k, v in models.items() if v['model'] is not None}

    # Metrics storage: results[model][domain][metric] = list of values
    raw_results = {name: defaultdict(lambda: defaultdict(list)) for name in models}
    asr_results = {name: [] for name in models}
    
    # Load loaders once
    domain_loaders = {
        'PAN22': PAN22Loader("data/raw/pan22_texts.jsonl",
                             "data/raw/pan22_labels.jsonl"),
        'Blog': BlogTextLoader("data/raw/blogtext.csv"),
        'Enron': EnronLoader("data/raw/emails.csv"),
    }
    
    for domain, loader in domain_loaders.items():
        loader.load(limit=6000)

    # ASR Evaluation needs cached attacks (independent of seed sampling)
    print("\n[Loading Attack Cache]")
    cached = []
    with open(ADV_CACHE, 'r') as f:
        for line in f:
            cached.append(json.loads(line))
    anchors_adv = [c['anchor'] for c in cached]
    originals_adv = [c['positive'] for c in cached]
    attacked_adv = [c['attacked'] for c in cached]

    for seed in SEEDS:
        print(f"\n" + "=" * 40)
        print(f"EVALUATING SEED {seed}")
        print("=" * 40)
        
        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        BATCH_SIZE = 64
        
        for domain, loader in domain_loaders.items():
            t1, t2, labels = loader.create_pairs(num_pairs=500)
            if not t1: continue
            
            labels = np.array(labels)
            valid_mask = labels != -1
            v_labels = labels[valid_mask].astype(int)
            
            for name, cfg in models.items():
                all_probs = []
                for i in range(0, len(t1), BATCH_SIZE):
                    bt1 = t1[i:i+BATCH_SIZE]
                    bt2 = t2[i:i+BATCH_SIZE]
                    
                    if cfg['type'] == 'dann':
                        probs = predict_dann(cfg['model'], extractor, bt1, bt2)
                    elif cfg['type'] == 'siamese':
                        probs = predict_siamese(cfg['model'], cfg['vec'], cfg['scaler'], bt1, bt2)
                    
                    if isinstance(probs, float): probs = np.array([probs])
                    all_probs.extend(probs.flatten().tolist())
                
                all_probs = np.array(all_probs)[valid_mask]
                preds = (all_probs > 0.5).astype(int)
                
                acc = accuracy_score(v_labels, preds)
                try: roc = roc_auc_score(v_labels, all_probs)
                except: roc = 0.0
                f1 = f1_score(v_labels, preds, zero_division=0)
                
                raw_results[name][domain]['acc'].append(acc)
                raw_results[name][domain]['roc'].append(roc)
                raw_results[name][domain]['f1'].append(f1)
                
        # ASR Evaluation (same cache, but compute again in case model dropout/state changes? 
        # Models are in eval mode, so ASR should be identical per seed if deterministic.
        # But we do it once per seed for structural consistency if needed, or just once.)
        # Since it's deterministic in eval mode, we'll just run it once for the first seed.
        if seed == SEEDS[0]:
            print("\n  Evaluating ASR...")
            for name, cfg in models.items():
                success = 0; valid_orig = 0
                for i in range(len(anchors_adv)):
                    if cfg['type'] == 'dann':
                        p_o = predict_dann(cfg['model'], extractor, [anchors_adv[i]], [originals_adv[i]])[0]
                        p_a = predict_dann(cfg['model'], extractor, [anchors_adv[i]], [attacked_adv[i]])[0]
                    elif cfg['type'] == 'siamese':
                        p_o = predict_siamese(cfg['model'], cfg['vec'], cfg['scaler'], [anchors_adv[i]], [originals_adv[i]])
                        p_a = predict_siamese(cfg['model'], cfg['vec'], cfg['scaler'], [anchors_adv[i]], [attacked_adv[i]])
                        if isinstance(p_o, np.ndarray): p_o = p_o[0]
                        if isinstance(p_a, np.ndarray): p_a = p_a[0]
                    
                    if p_o > 0.5:
                        valid_orig += 1
                        if p_a < 0.5: success += 1
                asr = success / valid_orig if valid_orig > 0 else 0.0
                asr_results[name] = asr

    print("\n" + "=" * 60)
    print("FINAL MULTI-SEED AGGREGATES")
    print("=" * 60)
    
    summary = {}
    for name in models:
        summary[name] = {}
        print(f"\n{name}:")
        avg_accs = []
        for domain in domain_loaders:
            if not raw_results[name][domain]['acc']: continue
            accs = raw_results[name][domain]['acc']
            f1s = raw_results[name][domain]['f1']
            
            m_acc = np.mean(accs); s_acc = np.std(accs)
            m_f1 = np.mean(f1s); s_f1 = np.std(f1s)
            avg_accs.append(m_acc)
            
            summary[name][domain] = {
                'acc_mean': round(m_acc, 4), 'acc_std': round(s_acc, 4),
                'f1_mean': round(m_f1, 4), 'f1_std': round(s_f1, 4)
            }
            print(f"  {domain:6s} | Acc: {m_acc*100:.1f}±{s_acc*100:.1f}% | F1: {m_f1:.3f}±{s_f1:.3f}")
            
        summary[name]['Average'] = {'acc_mean': round(np.mean(avg_accs), 4)}
        summary[name]['ASR'] = round(asr_results[name], 4)
        print(f"  Avg Acc: {np.mean(avg_accs)*100:.1f}% | ASR: {asr_results[name]*100:.1f}%")

    # Effect Sizes
    print("\n[Effect Sizes (Cohen's d)]")
    effect_sizes = {}
    
    if 'Rob Siamese' in raw_results and 'CD Siamese' in raw_results:
        # 1. Rob Siamese vs CD Siamese Accuracy (Avg over seeds)
        rob_acc_seeds = [np.mean([raw_results['Rob Siamese'][d]['acc'][i] for d in domain_loaders]) for i in range(len(SEEDS))]
        cd_acc_seeds = [np.mean([raw_results['CD Siamese'][d]['acc'][i] for d in domain_loaders]) for i in range(len(SEEDS))]
        d_acc = cohens_d(rob_acc_seeds, cd_acc_seeds)
        effect_sizes['Rob_vs_CD_Acc'] = round(d_acc, 4)
        print(f"  Rob Siamese vs CD Siamese (Avg Acc): d = {d_acc:.3f}")
    else:
        print("  ✗ Could not compute Cohen's d: missing Rob Siamese or CD Siamese")
    
    summary['effect_sizes'] = effect_sizes
    summary['raw_results'] = raw_results
    
    os.makedirs('results', exist_ok=True)
    with open('results/multi_seed_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nSaved to results/multi_seed_results.json")

if __name__ == "__main__":
    main()
