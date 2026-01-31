import sys
import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from torch.utils.data import TensorDataset, DataLoader

# Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader_scie import PAN22Loader, BlogTextLoader, EnronLoader
from models.dann_siamese import DANNSiamese

# Config
OUTPUT_DIR = "results_dann"
DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
MAX_FEATURES = 4308

def estimate_mutual_information():
    print("Loading Resources...")
    if not os.path.exists(f"{OUTPUT_DIR}/dann_model_v2.pth"):
        print("V2 Model not found. Training incomplete.")
        return

    # 1. Load Extractor
    with open(f"{OUTPUT_DIR}/extractor.pkl", "rb") as f:
        extractor = pickle.load(f)
        
    # 2. Load Model
    model = DANNSiamese(input_dim=MAX_FEATURES, num_domains=4).to(DEVICE)
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/dann_model_v2.pth", map_location=DEVICE))
    model.eval()
    
    # 3. Load Samples (Small subset)
    loaders = [
        PAN22Loader("pan22-authorship-verification-training.jsonl"),
        BlogTextLoader("blogtext.csv"),
        EnronLoader("emails.csv")
    ]
    
    embeddings = []
    domain_labels = []
    
    print("Extracting Embeddings...")
    
    def flatten_feats(feats_dict):
        return np.hstack([feats_dict['char'], feats_dict['pos'], feats_dict['lex'], feats_dict['readability']])
        
    def pad(X, dim=4308):
        if X.shape[1] < dim: return np.hstack([X, np.zeros((X.shape[0], dim - X.shape[1]))])
        return X[:, :dim]

    for i, loader in enumerate(loaders):
        try:
             df = loader.load(limit=2000) # 2000 samples per domain
             if df.empty: continue
             texts = df['text'].tolist()[:1000] # Use 1000
             
             # Extract Features
             f_dict = extractor.transform(texts)
             X = flatten_feats(f_dict)
             X = pad(X, MAX_FEATURES)
             X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
             
             # Get Embedding Z
             with torch.no_grad():
                 z = model.feature_extractor(X_t).cpu().numpy()
                 
             embeddings.append(z)
             domain_labels.extend([i] * len(z))
             
        except Exception as e:
            print(f"Error processing domain {i}: {e}")
            
    Z = np.vstack(embeddings)
    D = np.array(domain_labels)
    
    print(f"Estimating Mutual Information I(Z; D) on {Z.shape[0]} samples...")
    # Estimate MI for each dimension of Z and sum/mean
    # I(Z; D) = sum(I(Z_i; D)) if independent, or use multivariate estimator.
    # mutual_info_classif estimates I(X; Y)
    
    mi_scores = mutual_info_classif(Z, D, discrete_features=False, random_state=42)
    total_mi = mi_scores.sum()
    mean_mi = mi_scores.mean()
    
    print(f"Total MI I(Z; D): {total_mi:.4f}")
    print(f"Mean MI per dimension: {mean_mi:.4f}")
    
    # Interpretation
    # Lower MI = Better Privacy/Invariance
    
    with open(f"{OUTPUT_DIR}/theoretical_analysis.txt", "w") as f:
        f.write("Theoretical Analysis (Information Theoretic)\n")
        f.write("============================================\n")
        f.write(f"Mutual Information I(Z; Domain): {total_mi:.4f}\n")
        f.write(f"Interpretation: Lower values indicate domain invariance.\n")
        
if __name__ == "__main__":
    estimate_mutual_information()
