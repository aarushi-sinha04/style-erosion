
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
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
OUTPUT_FILE = 'results/bertscore.json'
MODEL_PATH = 'results/siamese_baseline/best_model.pth'
VEC_PATH = 'results/siamese_baseline/vectorizer.pkl'
SCALER_PATH = 'results/siamese_baseline/scaler.pkl'

def measure_attack_quality():
    print("="*60)
    print("Measuring Attack Quality (BERTScore)")
    print("="*60)
    
    # 1. Load Data (Texts to attack)
    loader = PAN22Loader("pan22-authorship-verification-training.jsonl", 
                         "pan22-authorship-verification-training-truth.jsonl")
    
    # Get pairs
    # Attack 20 positives
    t1_list, t2_list, labels = loader.create_pairs(num_pairs=100)
    positive_indices = [i for i, l in enumerate(labels) if l == 1][:20]
    
    # Load Siamese for ASR
    print("Loading Siamese Model for ASR...")
    with open(VEC_PATH, 'rb') as f: vec = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
    siamese = SiameseNetwork(input_dim=3000).to(DEVICE)
    siamese.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    siamese.eval()

    from experiments.train_siamese import preprocess 

    originals = [t2_list[i] for i in positive_indices]
    anchors = [t1_list[i] for i in positive_indices] # We need anchors for prediction
    
    # 2. Attack
    print(f"Generating attacks for {len(originals)} texts...")
    attacker = Paraphraser(device=DEVICE)
    
    attacked = attacker.attack(originals)
    
    if not attacked or len(attacked) != len(originals):
        print("Attack failed or returned mismatched length.")
        return

    # 3. Calculate ASR (Attack Success Rate)
    print("Calculating Attack Success Rate...")
    success_count = 0
    total_valid = 0
    
    for i in range(len(originals)):
        # Check original prediction
        t1_p = preprocess(anchors[i])
        t2_p = preprocess(originals[i])
        
        x1 = scaler.transform(vec.transform([t1_p]).toarray())
        x2 = scaler.transform(vec.transform([t2_p]).toarray())
        X1 = torch.tensor(x1, dtype=torch.float32).to(DEVICE)
        X2 = torch.tensor(x2, dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            orig_prob = torch.sigmoid(siamese(X1, X2)).item()
            
        if orig_prob > 0.5: # Only count if model was originally correct
            total_valid += 1
            
            # Check attacked prediction
            t2_adv_p = preprocess(attacked[i])
            x2_adv = scaler.transform(vec.transform([t2_adv_p]).toarray())
            X2_adv = torch.tensor(x2_adv, dtype=torch.float32).to(DEVICE)
            
            with torch.no_grad():
                adv_prob = torch.sigmoid(siamese(X1, X2_adv)).item()
                
            if adv_prob < 0.5:
                success_count += 1
                
    asr = success_count / total_valid if total_valid > 0 else 0
    print(f"Attack Success Rate: {asr*100:.1f}% ({success_count}/{total_valid})")

    # 4. Compute BERTScore
    print("Computing BERTScore...")
    # lang='en', verbose=True
    # Returns (P, R, F1)
    P, R, F1 = score(attacked, originals, lang='en', verbose=True, device=DEVICE)
    
    avg_p = P.mean().item()
    avg_r = R.mean().item()
    avg_f1 = F1.mean().item()
    
    print(f"\nResults:")
    print(f"Precision: {avg_p:.4f}")
    print(f"Recall:    {avg_r:.4f}")
    print(f"F1:        {avg_f1:.4f}")
    
    # 5. Save
    results = {
        'precision': avg_p,
        'recall': avg_r,
        'f1': avg_f1,
        'asr': asr
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nSaved to {OUTPUT_FILE}")

if __name__ == "__main__":
    measure_attack_quality()
