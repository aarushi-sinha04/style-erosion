import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.metrics import accuracy_score, roc_auc_score
from utils.data_loader_scie import BlogTextLoader, EnronLoader
from tqdm import tqdm

# Configuration
MODEL_PATH = "results_pan_siamese/best_model.pth"
VEC_PATH = "results_pan_siamese/vectorizer.pkl"
SCALER_PATH = "results_pan_siamese/scaler.pkl"
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

# Hyperparameters (Must match training)
HIDDEN_DIM = 512
DROPOUT = 0.3
MAX_FEATURES = 3000

# ==============================================================================
# 1. MODEL DEFINITION (Must match exactly)
# ==============================================================================
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
    text = re.sub(r'<addr\d+_[A-Z]+>', ' <TAG> ', text)
    text = re.sub(r'<[^>]+>', ' <TAG> ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==============================================================================
# 2. EVALUATION LOGIC
# ==============================================================================
def load_resources():
    print(f"Loading resources from results_pan_siamese/...")
    with open(VEC_PATH, 'rb') as f: vec = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
    
    model = SiameseNetwork(input_dim=MAX_FEATURES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    return model, vec, scaler

def evaluate_domain(name, pairs_t1, pairs_t2, labels, model, vec, scaler):
    print(f"\nEvaluating Domain: {name} ({len(labels)} pairs)")
    
    # Preprocess
    print("Preprocessing...")
    t1_clean = [preprocess(str(t)) for t in pairs_t1]
    t2_clean = [preprocess(str(t)) for t in pairs_t2]
    
    # Vectorize
    print("Vectorizing...")
    try:
        x1 = vec.transform(t1_clean).toarray()
        x2 = vec.transform(t2_clean).toarray()
    except Exception as e:
        print(f"Vectorization Error: {e}")
        return
    
    # Scale
    print("Scaling...")
    all_vecs = np.vstack([x1, x2])
    # Note: We use the SCALER fitted on PAN22. This is correct for Zero-Shot.
    # If the domain is too different, this might be why it fails (feature shift).
    x1 = scaler.transform(x1)
    x2 = scaler.transform(x2)
    
    # To Tensor
    x1_t = torch.tensor(x1, dtype=torch.float32).to(DEVICE)
    x2_t = torch.tensor(x2, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(labels, dtype=torch.float32).to(DEVICE)
    
    # Predict (Batching logic simplified for 1000 samples - can pass all at once on most GPUs)
    BATCH_SIZE = 100
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(labels), BATCH_SIZE):
            bx1 = x1_t[i:i+BATCH_SIZE]
            bx2 = x2_t[i:i+BATCH_SIZE]
            logits = model(bx1, bx2)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            
    # Metrics
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    acc = accuracy_score(labels, preds)
    roc = roc_auc_score(labels, all_probs)
    
    print(f"RESULTS for {name}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {roc:.4f}")
    
    return acc, roc

# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================
def run_cross_domain_eval():
    model, vec, scaler = load_resources()
    
    # 1. BlogText
    if os.path.exists("blogtext.csv"):
        loader = BlogTextLoader("blogtext.csv")
        loader.load(min_posts_per_author=50) # Strict filter
        t1, t2, y = loader.create_pairs(num_pairs=1000)
        evaluate_domain("BlogText (Zero-Shot)", t1, t2, y, model, vec, scaler)
    else:
        print("Skipping BlogText (file not found)")
        
    # 2. Enron
    if os.path.exists("emails.csv"):
        loader = EnronLoader("emails.csv")
        loader.load(top_n_authors=50) # Top 50 authors
        t1, t2, y = loader.create_pairs(num_pairs=1000)
        evaluate_domain("Enron Emails (Zero-Shot)", t1, t2, y, model, vec, scaler)
    else:
        print("Skipping Enron (file not found)")

if __name__ == "__main__":
    run_cross_domain_eval()
