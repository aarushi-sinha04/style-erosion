"""
Evaluate syntactic ablations on all attack types.
Tests 4 DANN variants (3 ablations + full baseline) against T5, synonym, and back-translation attacks.

Expected time: ~30-60 min (mostly feature extraction)
"""
import sys
import os
import json
import torch
import numpy as np
import pickle
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.dann import DANNSiameseV3
from experiments.train_syntactic_ablations import AblationDANN, ABLATION_CONFIGS

DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')


def load_jsonl(path):
    """Load JSONL cache file."""
    items = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def predict_ablation(model, extractor, text_a, text_b, feature_key):
    """Predict with ablation model using only the specified feature view."""
    f1_dict = extractor.transform([text_a])
    f2_dict = extractor.transform([text_b])
    
    X1 = f1_dict[feature_key]
    X2 = f2_dict[feature_key]
    
    if hasattr(X1, 'toarray'):
        X1 = X1.toarray()
    if hasattr(X2, 'toarray'):
        X2 = X2.toarray()
    
    x1 = torch.tensor(np.array(X1), dtype=torch.float32).to(DEVICE)
    x2 = torch.tensor(np.array(X2), dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        p, _, _ = model(x1, x2, alpha=0.0)
    return p.cpu().item()


def predict_full(model, extractor, text_a, text_b):
    """Predict with full DANN using all features."""
    f1_dict = extractor.transform([text_a])
    f2_dict = extractor.transform([text_b])
    
    X1 = np.hstack([f1_dict['char'], f1_dict['pos'], f1_dict['lex'], f1_dict['readability']])
    X2 = np.hstack([f2_dict['char'], f2_dict['pos'], f2_dict['lex'], f2_dict['readability']])
    
    x1 = torch.tensor(X1, dtype=torch.float32).to(DEVICE)
    x2 = torch.tensor(X2, dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        p, _, _ = model(x1, x2, alpha=0.0)
    return p.cpu().item()


def evaluate_ablations():
    """
    Test 4 DANN variants on 3 attack types.
    """
    print(f"\n{'='*60}")
    print("SYNTACTIC FEATURE ABLATION EVALUATION")
    print(f"{'='*60}")
    
    # Load extractor
    extractor_path = "results/final_dann/extractor.pkl"
    print(f"\nLoading extractor from {extractor_path}...")
    with open(extractor_path, 'rb') as f:
        extractor = pickle.load(f)
    
    # Load ablation models
    models = {}
    
    for ablation_type, config in ABLATION_CONFIGS.items():
        model_path = f"models/dann_{ablation_type}.pth"
        if not os.path.exists(model_path):
            print(f"  ✗ {ablation_type} not found at {model_path}")
            continue
        model = AblationDANN(input_dim=config['input_dim'], num_domains=4).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        models[ablation_type] = {
            'model': model,
            'feature_key': config['feature_key'],
            'type': 'ablation',
            'description': config['description']
        }
        print(f"  ✓ Loaded {ablation_type} ({config['description']})")
    
    # Load full baseline (Robust DANN)
    full_model_path = "results/robust_dann/robust_dann_model.pth"
    if os.path.exists(full_model_path):
        full_model = DANNSiameseV3(input_dim=4308, num_domains=4).to(DEVICE)
        full_model.load_state_dict(torch.load(full_model_path, map_location=DEVICE))
        full_model.eval()
        models['full_baseline'] = {
            'model': full_model,
            'type': 'full',
            'description': 'Full multi-view (robust DANN)'
        }
        print(f"  ✓ Loaded full baseline (Robust DANN)")
    
    # Also try base DANN
    base_model_path = "results/final_dann/dann_model_v4.pth"
    if os.path.exists(base_model_path):
        base_model = DANNSiameseV3(input_dim=4308, num_domains=4).to(DEVICE)
        base_model.load_state_dict(torch.load(base_model_path, map_location=DEVICE))
        base_model.eval()
        models['full_base_dann'] = {
            'model': base_model,
            'type': 'full',
            'description': 'Full multi-view (base DANN)'
        }
        print(f"  ✓ Loaded full baseline (Base DANN)")
    
    if not models:
        print("No models loaded! Exiting.")
        return
    
    # Load attack caches
    attack_caches = {}
    cache_files = {
        't5': 'data/eval_adversarial_cache.jsonl',
        'synonym': 'data/synonym_adversarial_cache.jsonl',
        'backtrans': 'data/backtranslation_adversarial_cache.jsonl'
    }
    
    # Map field names (different caches use different key names)
    field_maps = {
        't5': {'anchor': 'anchor', 'original': 'positive', 'attacked': 'attacked'},
        'synonym': {'anchor': 'anchor', 'original': 'positive', 'attacked': 'attacked'},
        'backtrans': {'anchor': 'anchor', 'original': 'positive', 'attacked': 'attacked'}
    }
    
    for attack_name, cache_path in cache_files.items():
        if os.path.exists(cache_path):
            data = load_jsonl(cache_path)
            attack_caches[attack_name] = data
            print(f"\n  Loaded {attack_name}: {len(data)} samples")
        else:
            print(f"\n  ✗ {attack_name} cache not found at {cache_path}")
    
    if not attack_caches:
        print("No attack caches found! Exiting.")
        return
    
    # Detect field names from first entry
    for attack_name, data in attack_caches.items():
        sample = data[0]
        keys = list(sample.keys())
        print(f"  {attack_name} fields: {keys}")
        
        # Try to auto-detect field mapping
        fmap = field_maps[attack_name]
        for field in ['text_a', 'anchor']:
            if field in sample:
                fmap['anchor'] = field
                break
        for field in ['text_b_original', 'positive', 'original']:
            if field in sample:
                fmap['original'] = field
                break
        for field in ['text_b_attacked', 'attacked']:
            if field in sample:
                fmap['attacked'] = field
                break
        field_maps[attack_name] = fmap
    
    # Evaluate each model on each attack
    results = {}
    
    for model_name, model_cfg in models.items():
        model = model_cfg['model']
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name} ({model_cfg['description']})")
        print(f"{'='*60}")
        
        model_results = {}
        
        for attack_name, data in attack_caches.items():
            fmap = field_maps[attack_name]
            successful = 0
            total_valid = 0
            errors = 0
            
            for item in tqdm(data, desc=f"  {attack_name}", leave=False):
                try:
                    text_a = item[fmap['anchor']]
                    text_b_orig = item[fmap['original']]
                    text_b_atk = item[fmap['attacked']]
                    
                    # Predict
                    if model_cfg['type'] == 'ablation':
                        p_orig = predict_ablation(model, extractor, text_a, text_b_orig, 
                                                  model_cfg['feature_key'])
                        p_atk = predict_ablation(model, extractor, text_a, text_b_atk,
                                                 model_cfg['feature_key'])
                    else:
                        p_orig = predict_full(model, extractor, text_a, text_b_orig)
                        p_atk = predict_full(model, extractor, text_a, text_b_atk)
                    
                    # ASR: did model correctly identify original as same-author, 
                    # and did attack flip the prediction?
                    if p_orig > 0.5:
                        total_valid += 1
                        if p_atk < 0.5:
                            successful += 1
                except Exception as e:
                    errors += 1
                    if errors <= 3:
                        print(f"    Error: {e}")
            
            asr = successful / total_valid if total_valid > 0 else 0.0
            model_results[attack_name] = {
                'asr': round(float(asr), 4),
                'successful_attacks': successful,
                'total_valid': total_valid,
                'total_samples': len(data),
                'errors': errors
            }
            print(f"  {attack_name:12s}: ASR={asr*100:.1f}% ({successful}/{total_valid})")
        
        results[model_name] = model_results
    
    # Save results
    os.makedirs("results", exist_ok=True)
    output_path = "results/syntactic_ablations.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY: Which syntactic features drive robustness?")
    print(f"{'='*60}")
    
    header = f"{'Model':24s} | {'T5 ASR':>8s} | {'Synonym':>8s} | {'BackTr':>8s} | {'Avg ASR':>8s}"
    print(header)
    print("-" * len(header))
    
    for model_name, model_results in results.items():
        desc = models[model_name]['description'] if model_name in models else model_name
        asrs = []
        row = f"{desc:24s} | "
        
        for attack_name in ['t5', 'synonym', 'backtrans']:
            if attack_name in model_results:
                asr = model_results[attack_name]['asr']
                asrs.append(asr)
                row += f"{asr*100:>7.1f}% | "
            else:
                row += f"{'N/A':>8s} | "
        
        avg_asr = np.mean(asrs) if asrs else 0.0
        row += f"{avg_asr*100:>7.1f}%"
        print(row)
    
    print(f"\n✅ Results saved to {output_path}")
    return results


if __name__ == "__main__":
    evaluate_ablations()
