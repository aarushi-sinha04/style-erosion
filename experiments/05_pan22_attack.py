import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import os
import re
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# CONFIG
# ==============================================================================
MODEL_DIR = "results_pan_siamese"
DATA_DIR = "."
OUTPUT_DIR = "results_pan_attack"
TRAIN_FILE = "pan22-authorship-verification-training.jsonl"
TRUTH_FILE = "pan22-authorship-verification-training-truth.jsonl"

# We use a T5 model fine-tuned for paraphrasing
PARAPHRASE_MODEL_NAME = "humarin/chatgpt_paraphraser_on_T5_base"
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Using Device: {DEVICE}")

# ==============================================================================
# 1. LOAD SIAMESE MODEL (Target)
# ==============================================================================
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SiameseNetwork, self).__init__()
        HIDDEN_DIM = 512
        DROPOUT = 0.3 # Value doesn't matter for eval
        
        self.branch = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(1024, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )
        
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward_one(self, x):
        return self.branch(x)

    def forward(self, x1, x2):
        u = self.forward_one(x1)
        v = self.forward_one(x2)
        diff = torch.abs(u - v)
        prod = u * v
        combined = torch.cat([u, v, diff, prod], dim=1)
        return self.head(combined)

def load_target_model():
    print("Loading Target Siamese Model...")
    with open(f"{MODEL_DIR}/vectorizer.pkl", "rb") as f: vec = pickle.load(f)
    with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f: scaler = pickle.load(f)
    
    model = SiameseNetwork(input_dim=3000).to(DEVICE)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/best_model.pth", map_location=DEVICE))
    model.eval()
    return model, vec, scaler

# ==============================================================================
# 2. LOAD DATA
# ==============================================================================
def preprocess(text):
    text = text.replace("<nl>", " ")
    text = re.sub(r'<addr\d+_[A-Z]+>', ' <TAG> ', text)
    text = re.sub(r'<[^>]+>', ' <TAG> ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_validation_pairs():
    print("Loading PAN22 Data...")
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
            
    ids = sorted(list(pairs_dict.keys()))
    pairs = []
    labels = []
    
    for i in ids:
        if i in labels_dict:
            pairs.append(pairs_dict[i])
            labels.append(labels_dict[i])
            
    # Use same split as training to get validation set
    _, pairs_val, _, y_val = train_test_split(pairs, labels, test_size=0.20, random_state=42, stratify=labels)
    
    return pairs_val, y_val

# ==============================================================================
# 3. PARAPHRASER (Attacker)
# ==============================================================================
class Paraphraser:
    def __init__(self):
        print(f"Loading Paraphraser: {PARAPHRASE_MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(PARAPHRASE_MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASE_MODEL_NAME).to(DEVICE)
        self.model.eval()
        
    def generate(self, text, num_beams=5, num_return_sequences=1):
        """Paraphrase text using T5."""
        input_ids = self.tokenizer(
            f'paraphrase: {text}',
            return_tensors="pt", padding="longest",
            max_length=512, truncation=True
        ).input_ids.to(DEVICE)
        
        outputs = self.model.generate(
            input_ids,
            max_length=512,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            temperature=1.5,
            do_sample=True,  # Add randomness to ensure change
            early_stopping=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==============================================================================
# 4. ATTACK LOOP
# ==============================================================================
def get_prediction(model, vec, scaler, t1, t2):
    t1_p = preprocess(t1)
    t2_p = preprocess(t2)
    
    x1 = scaler.transform(vec.transform([t1_p]).toarray())
    x2 = scaler.transform(vec.transform([t2_p]).toarray())
    
    x1_t = torch.tensor(x1, dtype=torch.float32).to(DEVICE)
    x2_t = torch.tensor(x2, dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        logits = model(x1_t, x2_t)
        prob = torch.sigmoid(logits).item()
        
    return prob

def run_attack():
    # 1. Load Components
    siamese, vec, scaler = load_target_model()
    pairs_val, y_val = load_validation_pairs()
    
    # Delayed loading to save VRAM if needed, but here we load it
    paraphraser = Paraphraser()
    
    # 2. Select High-Confidence "Same Author" Pairs
    # We only want to attack pairs the model ALREADY gets right with high confidence.
    targets = []
    print("Scanning for vulnerable high-confidence pairs...")
    
    for i, (t1, t2) in enumerate(tqdm(pairs_val[:1000])): # Scan first 1000 to find enough targets
        if y_val[i] == 1.0: # Same Author
            prob = get_prediction(siamese, vec, scaler, t1, t2)
            if prob > 0.85: # High confidence
                targets.append((t1, t2, prob))
                
    print(f"Found {len(targets)} high-confidence 'Same Author' pairs.")
    
    # 3. Perform Attack
    # Limit to 50 pairs to save time (paraphrasing is slow)
    attack_subset = targets[:50]
    
    results = []
    erosions = []
    
    print(f"Attacking {len(attack_subset)} pairs...")
    
    with open(f"{OUTPUT_DIR}/attack_log.txt", "w") as f:
        f.write("ADVERSARIAL ATTACK LOG\n")
        f.write("======================\n\n")
        
        for t1, t2_orig, prob_orig in tqdm(attack_subset):
            # Paraphrase Text B
            try:
                t2_new = paraphraser.generate(t2_orig)
            except Exception as e:
                print(f"Paraphrase failed: {e}")
                continue
            
            # Re-evaluate
            prob_new = get_prediction(siamese, vec, scaler, t1, t2_new)
            
            erosion = prob_orig - prob_new
            erosions.append(erosion)
            
            # Log
            f.write(f"Original Prob: {prob_orig:.4f}\n")
            f.write(f"Attacked Prob: {prob_new:.4f}\n")
            f.write(f"Erosion:       {erosion:.4f}\n")
            f.write(f"--- Text B Original ---\n{t2_orig[:200]}...\n")
            f.write(f"--- Text B Paraphrased ---\n{t2_new[:200]}...\n")
            f.write("\n" + "="*40 + "\n\n")
            
            results.append({
                'prob_orig': prob_orig,
                'prob_new': prob_new,
                'erosion': erosion
            })
            
    # 4. Analysis
    mean_erosion = np.mean(erosions)
    success_rate = np.mean([1 if r['prob_new'] < 0.5 else 0 for r in results])
    
    print(f"\nRESULTS")
    print(f"Mean Erosion: {mean_erosion:.4f}")
    print(f"Attack Success Rate (flipped to different): {success_rate*100:.1f}%")
    
    # 5. Visualization
    probs_orig = [r['prob_orig'] for r in results]
    probs_new = [r['prob_new'] for r in results]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(results)), probs_orig, color='green', label='Original (Same Author)', alpha=0.7)
    plt.scatter(range(len(results)), probs_new, color='red', label='Attacked (Paraphrased)', alpha=0.7)
    
    # Draw lines connecting them
    for i in range(len(results)):
        plt.plot([i, i], [probs_orig[i], probs_new[i]], color='gray', linestyle='--', alpha=0.3)
        
    plt.axhline(y=0.5, color='black', linestyle='-', label='Decision Boundary')
    plt.title(f"Adversarial Attack: Probability Drop (Mean Erosion: {mean_erosion:.2f})")
    plt.xlabel("Sample Index")
    plt.ylabel("Probability of Same Author")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/erosion_plot.png")
    print("Saved erosion_plot.png")
    
    # Save Report
    with open(f"{OUTPUT_DIR}/attack_summary.txt", "w") as f:
        f.write(f"Mean Erosion: {mean_erosion:.4f}\n")
        f.write(f"Attack Success Rate: {success_rate*100:.1f}%\n")
        f.write(f"Total Samples Attacked: {len(results)}\n")

if __name__ == "__main__":
    run_attack()
