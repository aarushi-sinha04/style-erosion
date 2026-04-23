
import sys
import os
import torch
import torch.nn as nn
import json
import numpy as np
from tqdm import tqdm
from bert_score import score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_loader_scie import PAN22Loader
from utils.paraphraser import Paraphraser
import pickle
from experiments.train_siamese import SiameseNetwork

# Config
DEVICE = 'cpu'
OUTPUT_FILE = 'results/bertscore.json'
MODEL_PATH = 'results/siamese_baseline/best_model.pth'
VEC_PATH = 'results/siamese_baseline/vectorizer.pkl'
SCALER_PATH = 'results/siamese_baseline/scaler.pkl'

def load_cache(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]

def measure_attack_quality():
    print("="*60)
    print("Measuring Attack Quality (BERTScore)")
    print("="*60)
    
    caches = {
        'Synonym': 'data/synonym_adversarial_cache.jsonl',
        'T5 Paraphrase': 'data/eval_adversarial_cache.jsonl',
        'Back-Translation': 'data/backtranslation_adversarial_cache.jsonl'
    }
    
    overall_results = {}
    
    for name, path in caches.items():
        samples = load_cache(path)
        if not samples:
            print(f"  ⚠ {name} cache not found at {path}")
            continue
            
        print(f"\nEvaluating {name} ({len(samples)} samples)...")
        originals = [s['positive'] for s in samples]
        attacked = [s['attacked'] for s in samples]
        
        P, R, F1 = score(attacked, originals, lang='en', verbose=False, device=DEVICE)
        
        res = {
            'precision': float(P.mean().item()),
            'recall': float(R.mean().item()),
            'f1': float(F1.mean().item()),
            'f1_std': float(F1.std().item()),
            'count': len(samples)
        }
        overall_results[name] = res
        
        print(f"    F1: {res['f1']:.4f} ± {res['f1_std']:.4f}")

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(overall_results, f, indent=2)
        
    print(f"\n✅ Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    measure_attack_quality()
