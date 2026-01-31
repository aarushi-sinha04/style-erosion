import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import os
import re
from tqdm import tqdm
import pandas as pd
import sys

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.attack_models import MultiPassParaphraser

# Config
MODEL_PATH = "results_pan_adversarial_training/best_model_robust.pth"
VEC_PATH = "results_pan_siamese/vectorizer.pkl"
SCALER_PATH = "results_pan_siamese/scaler.pkl"
DATA_DIR = "."
OUTPUT_DIR = "results_pan_defense_verification"
TRAIN_FILE = "pan22-authorship-verification-training.jsonl"
TRUTH_FILE = "pan22-authorship-verification-training-truth.jsonl"

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
NUM_ATTACK_SAMPLES = 25

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Model Class (Must match)
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

# Heuristic Attacker (Same as Training)
import random
class HeuristicAttacker:
    def attack(self, text):
        words = text.split()
        new_words = []
        for i, w in enumerate(words):
            if random.random() < 0.10: continue # Delete
            if i < len(words)-1 and random.random() < 0.10: # Swap
                new_words.append(words[i+1])
                new_words.append(words[i])
                words[i+1] = "" 
            else:
                if w != "": new_words.append(w)
        return " ".join(new_words)

def run_verification():
    print("Loading ROBUST Model...")
    with open(VEC_PATH, "rb") as f: vec = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
    model = SiameseNetwork(input_dim=3000).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    print("Initialize Heuristic Attacker...")
    attacker = HeuristicAttacker()
    
    # Load High Confidence Pairs
    print("Loading Validation Data...")
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
            
    print("Finding vulnerable pairs (Prob > 0.90)...")
    targets = []
    candidate_ids = sorted(list(pairs_dict.keys()))[:2000] 
    
    for pid in tqdm(candidate_ids):
        if labels_dict[pid] == 1.0:
            t1, t2 = pairs_dict[pid]
            prob = get_prob(model, vec, scaler, t1, t2)
            if prob > 0.90:
                targets.append({'t1': t1, 't2': t2, 'prob_orig': prob})
                if len(targets) >= NUM_ATTACK_SAMPLES:
                    break
                    
    print(f"Selected {len(targets)} pairs for verification.")
    
    results = []
    print("Running Heuristic Attack on Robust Model...")
    
    for item in tqdm(targets):
        # Heuristic Attack
        t2_aug = attacker.attack(item['t2'])
        prob_new = get_prob(model, vec, scaler, item['t1'], t2_aug)
        
        results.append({
            'prob_orig': item['prob_orig'],
            'prob_new': prob_new,
            'erosion': item['prob_orig'] - prob_new,
            'flipped': 1 if prob_new < 0.5 else 0
        })
            
    # Stats
    df = pd.DataFrame(results)
    mean_erosion = df['erosion'].mean()
    flip_rate = df['flipped'].mean()
    
    print(f"\nRESULTS (Defense Verification - Heuristic):")
    print(f"Mean Erosion: {mean_erosion:.4f}")
    print(f"Flip Rate:    {flip_rate*100:.1f}%")
    
    with open(f"{OUTPUT_DIR}/defense_report.txt", "w") as f:
        f.write(f"Defense Verification (Trained on Heuristic, Tested on Heuristic)\n")
        f.write(f"Mean Erosion: {mean_erosion:.4f}\n")
        f.write(f"Flip Rate: {flip_rate*100:.1f}%\n")
        
    df.to_csv(f"{OUTPUT_DIR}/defense_results.csv", index=False)

if __name__ == "__main__":
    run_verification()
