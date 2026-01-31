import torch
import torch.nn as nn
import numpy as np
import pickle
import re
import os
import argparse

# Configuration
# Configuration
MODEL_PATH = "results/siamese_baseline/best_model.pth"
VEC_PATH = "results/siamese_baseline/vectorizer.pkl"
SCALER_PATH = "results/siamese_baseline/scaler.pkl"
MAX_FEATURES = 3000
HIDDEN_DIM = 512
DROPOUT = 0.0  # No dropout during inference
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

# ==============================================================================
# 1. MODEL DEFINITION (Must Match Training)
# ==============================================================================
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SiameseNetwork, self).__init__()
        
        # Branch Network
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
        
        # Head Network
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
        logits = self.head(combined)
        return logits

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def preprocess(text):
    text = text.replace("<nl>", " ")
    text = re.sub(r'<addr\d+_[A-Z]+>', ' <TAG> ', text)
    text = re.sub(r'<[^>]+>', ' <TAG> ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run main_pan.py first.")
        
    print(f"Loading model from {MODEL_PATH}...")
    
    # Load Vectorizer & Scaler
    with open(VEC_PATH, "rb") as f:
        vec = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
        
    # Load Model
    model = SiameseNetwork(input_dim=MAX_FEATURES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    return model, vec, scaler

def verify_authorship(model, vec, scaler, text1, text2):
    # 1. Preprocess
    t1 = preprocess(text1)
    t2 = preprocess(text2)
    
    # 2. Vectorize
    x1 = vec.transform([t1]).toarray()
    x2 = vec.transform([t2]).toarray()
    
    # 3. Scale
    x1 = scaler.transform(x1)
    x2 = scaler.transform(x2)
    
    # 4. Predict
    with torch.no_grad():
        x1_t = torch.tensor(x1, dtype=torch.float32).to(DEVICE)
        x2_t = torch.tensor(x2, dtype=torch.float32).to(DEVICE)
        
        logits = model(x1_t, x2_t)
        prob = torch.sigmoid(logits).item()
        
    return prob

# ==============================================================================
# 3. INTERACTIVE MAIN
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify if two texts are by the same author.")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("files", nargs="*", help="Path to two files to compare")
    
    args = parser.parse_args()
    
    try:
        model, vec, scaler = load_artifacts()
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
        
    if args.interactive:
        print("\n--- Interactive Mode (Ctrl+C to exit) ---")
        while True:
            try:
                print("\nType Text A (or press Enter to paste multiple lines):")
                lines = []
                while True:
                    line = input()
                    if not line: break
                    lines.append(line)
                t1 = "\n".join(lines)
                if not t1.strip(): continue

                print("Type Text B:")
                lines = []
                while True:
                    line = input()
                    if not line: break
                    lines.append(line)
                t2 = "\n".join(lines)
                
                prob = verify_authorship(model, vec, scaler, t1, t2)
                print(f"\nResult: {'SAME AUTHOR' if prob > 0.5 else 'DIFFERENT AUTHORS'}")
                print(f"Confidence: {prob:.4f}")
            except KeyboardInterrupt:
                break
    elif len(args.files) == 2:
        with open(args.files[0], 'r') as f: t1 = f.read()
        with open(args.files[1], 'r') as f: t2 = f.read()
        prob = verify_authorship(model, vec, scaler, t1, t2)
        print(f"Probability of Same Author: {prob:.4f}")
    else:
        print("Usage:")
        print("  python inference.py --interactive")
        print("  python inference.py file1.txt file2.txt")
