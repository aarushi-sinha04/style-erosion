import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import os
import re
from tqdm import tqdm
import pandas as pd
from collections import Counter

# Config
MODEL_DIR = "results_pan_siamese"
DATA_DIR = "."
OUTPUT_DIR = "results_pan_attack_v3"
TRAIN_FILE = "pan22-authorship-verification-training.jsonl"
TRUTH_FILE = "pan22-authorship-verification-training-truth.jsonl"
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
NUM_ATTACK_SAMPLES = 50

# Model Class
HIDDEN_DIM = 512
DROPOUT = 0.3
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SiameseNetwork, self).__init__()
        self.branch = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(1024, HIDDEN_DIM), nn.BatchNorm1d(HIDDEN_DIM), nn.ReLU(), nn.Dropout(DROPOUT)
        )
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 4, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1)
        )
    def forward_one(self, x): return self.branch(x)
    def forward(self, x1, x2):
        u, v = self.forward_one(x1), self.forward_one(x2)
        return self.head(torch.cat([u, v, torch.abs(u - v), u * v], dim=1))

def preprocess(text):
    text = text.replace("<nl>", " ")
    text = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def get_prob(model, vec, scaler, t1, t2):
    x1 = scaler.transform(vec.transform([preprocess(t1)]).toarray())
    x2 = scaler.transform(vec.transform([preprocess(t2)]).toarray())
    with torch.no_grad():
        logits = model(torch.tensor(x1, dtype=torch.float32).to(DEVICE),
                       torch.tensor(x2, dtype=torch.float32).to(DEVICE))
        return torch.sigmoid(logits).item()

class CopyPasteAttacker:
    def __init__(self, vectorizer):
        self.vec = vectorizer
        self.feature_names = vectorizer.get_feature_names_out()
        
    def extract_signature(self, text, top_k=5):
        # Find 4-grams present in text A
        # Simple heuristic: Just take random 4-grams or first few?
        # Better: use the vectorizer to find non-zero entries
        x = self.vec.transform([text]).toarray()[0]
        indices = np.where(x > 0)[0]
        if len(indices) == 0: return []
        
        # Pick random n-grams that exist in A
        # In a real attack, we'd pick high-weight ones.
        # Here we just pick random "present" features to verify 'Identity Theft'
        selected_idx = np.random.choice(indices, min(len(indices), top_k), replace=False)
        return [self.feature_names[i] for i in selected_idx]
        
    def attack(self, target_text, source_signature):
        # Inject signature into target text
        # Simply append them at the end or insert randomly
        injection = " ".join(source_signature)
        # return f"{target_text} {injection}"
        
        # Interleave for stealth? No, append is strongest "Frame"
        return target_text + " " + injection

def run_impersonation_attack():
    print("Loading Original Model...")
    with open(f"{MODEL_DIR}/vectorizer.pkl", "rb") as f: vec = pickle.load(f)
    with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f: scaler = pickle.load(f)
    model = SiameseNetwork(input_dim=3000).to(DEVICE)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/best_model.pth", map_location=DEVICE))
    model.eval()
    
    attacker = CopyPasteAttacker(vec)
    
    # Load Data
    print("Loading Data...")
    pairs_dict = {}
    with open(os.path.join(DATA_DIR, TRAIN_FILE), 'r') as f:
        for line in f: 
            obj = json.loads(line)
            pairs_dict[obj['id']] = obj['pair']
    labels_dict = {}
    with open(os.path.join(DATA_DIR, TRUTH_FILE), 'r') as f:
        for line in f:
            obj = json.loads(line)
            labels_dict[obj['id']] = 1.0 if obj['same'] else 0.0
            
    # Find True Negatives (Correctly identified as Diff)
    targets = []
    ids = sorted(list(pairs_dict.keys()))[:2000]
    for pid in ids:
        if labels_dict[pid] == 0.0:
            t1, t2 = pairs_dict[pid]
            prob = get_prob(model, vec, scaler, t1, t2)
            if prob < 0.10: # High confidence they are diff
                targets.append({'t1': t1, 't2': t2, 'prob_orig': prob})
                if len(targets) >= NUM_ATTACK_SAMPLES: break
                
    print(f"Selected {len(targets)} pairs for Impersonation Attack.")
    
    results = []
    print("Running Copy-Paste Attack...")
    
    for item in tqdm(targets):
        # Extract style from t1 (The Victim)
        signature = attacker.extract_signature(item['t1'], top_k=10) # extract 10 markers
        
        # Inject into t2 (The Forgery)
        t2_aug = attacker.attack(item['t2'], signature)
        
        prob_new = get_prob(model, vec, scaler, item['t1'], t2_aug)
        
        results.append({
            'prob_orig': item['prob_orig'],
            'prob_new': prob_new,
            'delta': prob_new - item['prob_orig'],
            'success': 1 if prob_new > 0.5 else 0
        })
        
    df = pd.DataFrame(results)
    mean_increase = df['delta'].mean()
    success_rate = df['success'].mean()
    
    print(f"\nRESULTS (Impersonation):")
    print(f"Mean Probability Increase: +{mean_increase:.4f}")
    print(f"Success Rate (Framing):    {success_rate*100:.1f}%")
    
    with open(f"{OUTPUT_DIR}/impersonation_report.txt", "w") as f:
        f.write(f"Impersonation Attack (Copy-Paste of 10 n-grams)\n")
        f.write(f"Mean Prob Increase: {mean_increase:.4f}\n")
        f.write(f"Success Rate: {success_rate*100:.1f}%\n")

if __name__ == "__main__":
    run_impersonation_attack()
