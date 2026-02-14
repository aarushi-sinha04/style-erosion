"""
Comprehensive Robustness Evaluation V2
========================================
Evaluates all models side-by-side:
1. Base DANN (Domain Adversarial)
2. Robust DANN (Adversarial-Trained)
3. PAN22 Siamese (Specialist)
4. Cross-Domain Siamese (Generalist)
5. Ensemble (Weighted Voting)

Metrics: Accuracy, ROC-AUC, F1 per domain + ASR
"""
import sys
import os
import torch
import json
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.dann import DANNSiameseV3
from utils.data_loader_scie import PAN22Loader, BlogTextLoader, EnronLoader, IMDBLoader
from utils.paraphraser import Paraphraser
from experiments.train_siamese import SiameseNetwork, preprocess

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

# Paths
BASE_DANN = "results/final_dann/dann_model_v4.pth"
ROBUST_DANN = "results/robust_dann/robust_dann_model.pth"
PAN_SIAMESE = "results/siamese_baseline/best_model.pth"
PAN_VEC = "results/siamese_baseline/vectorizer.pkl"
PAN_SCALER = "results/siamese_baseline/scaler.pkl"
CD_SIAMESE = "results/siamese_crossdomain/best_model.pth"
CD_VEC = "results/siamese_crossdomain/vectorizer.pkl"
CD_SCALER = "results/siamese_crossdomain/scaler.pkl"
ROB_SIAMESE = "results/robust_siamese/best_model.pth"
ROB_VEC = "results/robust_siamese/vectorizer.pkl"
ROB_SCALER = "results/robust_siamese/scaler.pkl"
EXTRACTOR = "results/final_dann/extractor.pkl"
ADV_CACHE = "data/eval_adversarial_cache.jsonl"

def get_feats(extractor, texts):
    f_dict = extractor.transform(texts)
    return np.hstack([f_dict['char'], f_dict['pos'], f_dict['lex'], f_dict['readability']])

def load_dann_model(path, name):
    if not os.path.exists(path):
        print(f"  ✗ {name} not found at {path}")
        return None
    model = DANNSiameseV3(input_dim=4308, num_domains=4).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    print(f"  ✓ Loaded {name}")
    return model

def load_siamese_model(model_path, vec_path, scaler_path, input_dim, name):
    if not os.path.exists(model_path):
        print(f"  ✗ {name} not found")
        return None, None, None
    try:
        vec = pickle.load(open(vec_path, 'rb'))
        scaler = pickle.load(open(scaler_path, 'rb'))
        model = SiameseNetwork(input_dim=input_dim).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print(f"  ✓ Loaded {name}")
        return model, vec, scaler
    except Exception as e:
        print(f"  ✗ Failed to load {name}: {e}")
        return None, None, None

def predict_dann(model, extractor, t1_batch, t2_batch):
    """Batch prediction for DANN models."""
    f1 = get_feats(extractor, t1_batch)
    f2 = get_feats(extractor, t2_batch)
    x1 = torch.tensor(f1, dtype=torch.float32).to(DEVICE)
    x2 = torch.tensor(f2, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        p, _, _ = model(x1, x2, alpha=0.0)
    return p.squeeze().cpu().numpy()

def predict_siamese(model, vec, scaler, t1_batch, t2_batch):
    """Batch prediction for Siamese models."""
    p_t1 = [preprocess(t) for t in t1_batch]
    p_t2 = [preprocess(t) for t in t2_batch]
    v1 = scaler.transform(vec.transform(p_t1).toarray())
    v2 = scaler.transform(vec.transform(p_t2).toarray())
    x1 = torch.tensor(v1, dtype=torch.float32).to(DEVICE)
    x2 = torch.tensor(v2, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(x1, x2)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    return probs

def eval_robust_all():
    print("=" * 60)
    print("COMPREHENSIVE EVALUATION V2")
    print("=" * 60)

    # 1. Load Extractor & Models
    print("\n[Loading Models]")
    extractor = pickle.load(open(EXTRACTOR, 'rb'))
    
    # Store all model configs
    models = {}
    
    base_dann = load_dann_model(BASE_DANN, "Base DANN")
    if base_dann: models['Base DANN'] = {'model': base_dann, 'type': 'dann'}
    
    robust_dann = load_dann_model(ROBUST_DANN, "Robust DANN")
    if robust_dann: models['Robust DANN'] = {'model': robust_dann, 'type': 'dann'}
    
    siam, siam_v, siam_s = load_siamese_model(PAN_SIAMESE, PAN_VEC, PAN_SCALER, 3000, "PAN22 Siamese")
    if siam: models['PAN22 Siamese'] = {'model': siam, 'type': 'siamese', 'vec': siam_v, 'scaler': siam_s}
    
    cd_siam, cd_v, cd_s = load_siamese_model(CD_SIAMESE, CD_VEC, CD_SCALER, 5000, "CD Siamese")
    if cd_siam: models['CD Siamese'] = {'model': cd_siam, 'type': 'siamese', 'vec': cd_v, 'scaler': cd_s}
    
    rob_siam, rob_v, rob_s = load_siamese_model(ROB_SIAMESE, ROB_VEC, ROB_SCALER, 5000, "Robust Siamese")
    if rob_siam: models['Rob Siamese'] = {'model': rob_siam, 'type': 'siamese', 'vec': rob_v, 'scaler': rob_s}
    
    # Try ensemble
    try:
        from models.robust_ensemble import RobustMultiExpertEnsemble
        ensemble = RobustMultiExpertEnsemble()
        models['Ensemble'] = {'model': ensemble, 'type': 'ensemble'}
    except Exception as e:
        print(f"  ✗ Ensemble failed: {e}")

    if not models:
        print("No models loaded!")
        return

    # 2. Clean Accuracy Evaluation
    print("\n" + "=" * 60)
    print("[Phase 1] Clean Accuracy Across Domains")
    print("=" * 60)
    
    domain_loaders = {
        'PAN22': PAN22Loader("pan22-authorship-verification-training.jsonl",
                             "pan22-authorship-verification-training-truth.jsonl"),
        'Blog': BlogTextLoader("blogtext.csv"),
        'Enron': EnronLoader("emails.csv"),
    }
    
    results = {name: {} for name in models}
    BATCH_SIZE = 64
    
    for domain, loader in domain_loaders.items():
        print(f"\n  Evaluating {domain}...")
        loader.load(limit=6000)
        t1, t2, labels = loader.create_pairs(num_pairs=500)
        
        if not t1:
            print(f"    No pairs for {domain}")
            continue
        
        labels = np.array(labels)
        valid_mask = labels != -1
        
        if valid_mask.sum() == 0:
            print(f"    No labeled pairs for {domain}")
            continue
        
        for name, cfg in models.items():
            all_probs = []
            
            for i in range(0, len(t1), BATCH_SIZE):
                bt1 = t1[i:i+BATCH_SIZE]
                bt2 = t2[i:i+BATCH_SIZE]
                
                if cfg['type'] == 'dann':
                    probs = predict_dann(cfg['model'], extractor, bt1, bt2)
                elif cfg['type'] == 'siamese':
                    probs = predict_siamese(cfg['model'], cfg['vec'], cfg['scaler'], bt1, bt2)
                elif cfg['type'] == 'ensemble':
                    probs = []
                    for k in range(len(bt1)):
                        p = cfg['model'].predict(bt1[k], bt2[k], domain=domain.lower())
                        probs.append(p)
                    probs = np.array(probs)
                
                if isinstance(probs, float):
                    probs = np.array([probs])
                all_probs.extend(probs.flatten().tolist())
            
            all_probs = np.array(all_probs)
            preds = (all_probs > 0.5).astype(int)
            
            # Compute metrics on valid (labeled) samples
            v_labels = labels[valid_mask].astype(int)
            v_preds = preds[valid_mask]
            v_probs = all_probs[valid_mask]
            
            acc = accuracy_score(v_labels, v_preds)
            try:
                roc = roc_auc_score(v_labels, v_probs)
            except:
                roc = 0.0
            f1 = f1_score(v_labels, v_preds, zero_division=0)
            
            results[name][domain] = {'acc': round(acc, 4), 'roc': round(roc, 4), 'f1': round(f1, 4)}
            print(f"    {name:20s}: Acc={acc*100:.1f}% ROC={roc:.3f} F1={f1:.3f}")

    # 3. Adversarial Evaluation
    print("\n" + "=" * 60)
    print("[Phase 2] Adversarial Robustness (T5 Attack)")
    print("=" * 60)
    
    # Load/generate adversarial cache
    if os.path.exists(ADV_CACHE):
        print("Loading cached adversarial examples...")
        cached = []
        with open(ADV_CACHE, 'r') as f:
            for line in f:
                cached.append(json.loads(line))
        anchors_adv = [c['anchor'] for c in cached]
        originals_adv = [c['positive'] for c in cached]
        attacked_adv = [c['attacked'] for c in cached]
    else:
        print("Generating adversarial examples (will cache for reuse)...")
        pan_loader = domain_loaders['PAN22']
        t1_pan, t2_pan, y_pan = pan_loader.create_pairs(num_pairs=500)
        pos_idx = [i for i, y in enumerate(y_pan) if y == 1][:50]
        
        anchors_adv = [t1_pan[i] for i in pos_idx]
        originals_adv = [t2_pan[i] for i in pos_idx]
        
        paraphraser = Paraphraser(device=DEVICE)
        attacked_adv = []
        for text in tqdm(originals_adv, desc="Attacking"):
            try:
                res = paraphraser.attack([text])
                attacked_adv.append(res[0])
            except:
                attacked_adv.append(text)
        
        # Cache
        os.makedirs("data", exist_ok=True)
        with open(ADV_CACHE, 'w') as f:
            for a, o, atk in zip(anchors_adv, originals_adv, attacked_adv):
                f.write(json.dumps({'anchor': a, 'positive': o, 'attacked': atk}) + "\n")
        print(f"  Cached {len(attacked_adv)} adversarial examples")
    
    # Evaluate ASR
    asr_results = {}
    for name, cfg in models.items():
        success = 0
        valid_orig = 0
        
        for i in range(len(anchors_adv)):
            # Predict original
            if cfg['type'] == 'dann':
                p_orig = predict_dann(cfg['model'], extractor, [anchors_adv[i]], [originals_adv[i]])[0]
                p_atk = predict_dann(cfg['model'], extractor, [anchors_adv[i]], [attacked_adv[i]])[0]
            elif cfg['type'] == 'siamese':
                p_orig = predict_siamese(cfg['model'], cfg['vec'], cfg['scaler'], 
                                         [anchors_adv[i]], [originals_adv[i]])
                p_atk = predict_siamese(cfg['model'], cfg['vec'], cfg['scaler'],
                                        [anchors_adv[i]], [attacked_adv[i]])
                if isinstance(p_orig, np.ndarray): p_orig = p_orig[0]
                if isinstance(p_atk, np.ndarray): p_atk = p_atk[0]
            elif cfg['type'] == 'ensemble':
                p_orig = cfg['model'].predict(anchors_adv[i], originals_adv[i], domain='pan22')
                p_atk = cfg['model'].predict(anchors_adv[i], attacked_adv[i], domain='pan22')
            
            if p_orig > 0.5:
                valid_orig += 1
                if p_atk < 0.5:
                    success += 1
        
        asr = success / valid_orig if valid_orig > 0 else 0.0
        asr_results[name] = round(asr, 4)
        print(f"  {name:20s}: ASR={asr*100:.1f}% ({success}/{valid_orig})")

    # 4. Save Complete Report
    final = {'clean_accuracy': results, 'asr': asr_results}
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    # Print formatted table
    domains = list(domain_loaders.keys())
    header = f"{'Model':20s} | " + " | ".join([f"{d:>10s}" for d in domains]) + " |    ASR"
    print(header)
    print("-" * len(header))
    for name in models:
        row = f"{name:20s} | "
        for d in domains:
            if d in results[name]:
                row += f"{results[name][d]['acc']*100:>9.1f}%"
            else:
                row += f"{'N/A':>10s}"
            row += " | "
        row += f"{asr_results.get(name, 0)*100:5.1f}%"
        print(row)
    
    with open("results/final_robustness_metrics.json", "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved to results/final_robustness_metrics.json")

if __name__ == "__main__":
    eval_robust_all()
