import sys
import os
import torch
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader_scie import PAN22Loader, BlogTextLoader, EnronLoader, IMDBLoader
from models.dann_siamese import DANNSiamese
from utils.feature_extraction import EnhancedFeatureExtractor

# Config
OUTPUT_DIR = "results_dann"
BENCHMARK_FILE = f"{OUTPUT_DIR}/comprehensive_benchmark_v2.csv"
DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
MAX_FEATURES = 4308

def pad_features(X, target_dim=4308):
    if X.shape[1] == target_dim: return X
    if X.shape[1] > target_dim: return X[:, :target_dim]
    return np.hstack([X, np.zeros((X.shape[0], target_dim - X.shape[1]))])

def flatten_feats(feats_dict):
    return np.hstack([feats_dict['char'], feats_dict['pos'], feats_dict['lex'], feats_dict['readability']])

def run_benchmark():
    print("Starting Comprehensive Benchmark (SCIE Phase 6)...")
    
    if not os.path.exists(f"{OUTPUT_DIR}/dann_model_v2.pth"):
        print("V2 Model not found.")
        return

    # 1. Load Resources
    with open(f"{OUTPUT_DIR}/extractor.pkl", "rb") as f:
        extractor = pickle.load(f)
        
    model = DANNSiamese(input_dim=MAX_FEATURES, num_domains=4).to(DEVICE)
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/dann_model_v2.pth", map_location=DEVICE))
    model.eval()
    
    # 2. Define Datasets
    # Future work: Add Reddit/Guardian loaders here
    loaders = {
        'PAN22 (Source)': PAN22Loader("pan22-authorship-verification-training.jsonl", "pan22-authorship-verification-training-truth.jsonl"),
        'BlogText (Target)': BlogTextLoader("blogtext.csv"),
        'Enron (Target)': EnronLoader("emails.csv"),
        'IMDB (Target)': IMDBLoader("IMDB Dataset.csv")
    }
    
    records = []
    
    samples_per_domain = 1000 # Enough for significance
    
    for name, loader in loaders.items():
        print(f"Evaluatin {name}...")
        try:
            loader.load(limit=samples_per_domain*5)
        except Exception as e:
            print(f"Skipping {name}: {e}")
            continue
            
        # Create Test Set
        if 'IMDB' in name:
            # IMDB currently has no ground truth for authorship in our loader (dummy -1)
            # We skip 'Accuracy' for IMDB unless we synthesize authorship pairs properly
            # (IMDB Loader currently returns random pairs with -1)
            # We will skin Metric evaluation for IMDB until we build a Proper IMDB Authorship Loader
            print(f"Skipping Metrics for {name} (Unlabeled Pairs).")
            continue
        
        t1, t2, y = loader.create_pairs(num_pairs=samples_per_domain)
        if not t1: continue
        
        # Extract
        f1 = extractor.transform(t1)
        f2 = extractor.transform(t2)
        X1 = pad_features(flatten_feats(f1))
        X2 = pad_features(flatten_feats(f2))
        
        # Predict
        X1_t = torch.tensor(X1, dtype=torch.float32).to(DEVICE)
        X2_t = torch.tensor(X2, dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            preds, _, _ = model(X1_t, X2_t, alpha=0.0)
            
        preds_prob = preds.cpu().numpy().flatten()
        preds_bin = (preds_prob > 0.5).astype(int)
        
        # Metrics
        acc = accuracy_score(y, preds_bin)
        auc = roc_auc_score(y, preds_prob)
        f1 = f1_score(y, preds_bin)
        
        print(f"  Accuracy: {acc:.4f} | AUC: {auc:.4f}")
        
        records.append({
            'Dataset': name,
            'Samples': len(y),
            'Accuracy': f"{acc*100:.2f}%",
            'ROC-AUC': f"{auc:.4f}",
            'F1-Score': f"{f1:.4f}"
        })
        
    # Save Report
    df_res = pd.DataFrame(records)
    print("\nFinal Benchmark Results:")
    print(df_res)
    df_res.to_csv(BENCHMARK_FILE, index=False)
    print(f"Saved to {BENCHMARK_FILE}")

if __name__ == "__main__":
    run_benchmark()
