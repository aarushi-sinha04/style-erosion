import sys
import os
import torch
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import TensorDataset, DataLoader

# Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader_scie import PAN22Loader, BlogTextLoader, EnronLoader, IMDBLoader
from models.dann_siamese import DANNSiamese
from utils.feature_extraction import EnhancedFeatureExtractor

# Config
DATA_DIR = "."
OUTPUT_DIR = "results_dann"
DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
MAX_FEATURES = 4308

def pad_features(X, target_dim=4308):
    """Pad or truncate feature vectors to match model input dim."""
    if X.shape[1] == target_dim:
        return X
    if X.shape[1] > target_dim:
        return X[:, :target_dim]
    else:
        # Pad with zeros
        padding = np.zeros((X.shape[0], target_dim - X.shape[1]))
        return np.hstack([X, padding])

def run_eval():
    print(f"Loading resources from {OUTPUT_DIR}...")
    if not os.path.exists(f"{OUTPUT_DIR}/dann_model_v2.pth"):
        print("V2 Model not found. Training might be incomplete.")
        model_path = f"{OUTPUT_DIR}/dann_model.pth"
    else:
        model_path = f"{OUTPUT_DIR}/dann_model_v2.pth"
        
    print(f"Using model: {model_path}")

    # 1. Load Extractor
    with open(f"{OUTPUT_DIR}/extractor.pkl", "rb") as f:
        extractor = pickle.load(f)
        
    # 2. Load Model
    model = DANNSiamese(input_dim=MAX_FEATURES, num_domains=4).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 3. Load Test Data
    samples = 500
    
    loaders = {
        'PAN22': PAN22Loader("pan22-authorship-verification-training.jsonl", "pan22-authorship-verification-training-truth.jsonl"),
        'BlogText': BlogTextLoader("blogtext.csv"),
        'Enron': EnronLoader("emails.csv"),
        'IMDB': IMDBLoader("IMDB Dataset.csv")
    }
    
    results = {}
    embeddings_list = []
    domain_labels_list = []
    
    print("\nEvaluating Authorship Verification Accuracy (V2):")
    print("-" * 40)
    
    def flatten_feats(feats_dict):
        return np.hstack([
            feats_dict['char'], 
            feats_dict['pos'], 
            feats_dict['lex'], 
            feats_dict['readability']
        ])
    
    for domain, loader in loaders.items():
        print(f"Processing {domain}...")
        try:
            loader.load(limit=5000)
        except Exception as e:
            print(f"Error loading {domain}: {e}")
            continue

        if domain == 'IMDB': 
            t1_list, t2_list, labels = loader.create_pairs(num_pairs=samples)
        else:
            t1_list, t2_list, labels = loader.create_pairs(num_pairs=samples)
            if not t1_list: continue

        # Extract
        f1_dict = extractor.transform(t1_list)
        f2_dict = extractor.transform(t2_list)
        
        X1 = flatten_feats(f1_dict)
        X2 = flatten_feats(f2_dict)
        
        # Ensure dimensions match (in case extractor fit was different)
        X1 = pad_features(X1, MAX_FEATURES)
        X2 = pad_features(X2, MAX_FEATURES)
        
        X1_t = torch.tensor(X1, dtype=torch.float32).to(DEVICE)
        X2_t = torch.tensor(X2, dtype=torch.float32).to(DEVICE)
        
        # Forward
        with torch.no_grad():
            pred_auth, _, _ = model(X1_t, X2_t, alpha=0.0) 
            emb = model.feature_extractor(X1_t)
            
        # Accuracy
        if domain != 'IMDB':
            preds = (pred_auth.squeeze() > 0.5).float().cpu().numpy()
            y_true = np.array(labels)
            acc = (preds == y_true).mean()
            results[domain] = acc
            print(f"{domain}: {acc*100:.1f}%")
        else:
            print(f"{domain}: N/A (Unlabeled target)")
            
        # Store for t-SNE
        embeddings_list.append(emb.cpu().numpy())
        domain_labels_list.extend([domain] * len(X1))
        
    # 4. t-SNE Visualization
    print("\nGenerating t-SNE...")
    all_embeddings = np.vstack(embeddings_list)
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_2d = tsne.fit_transform(all_embeddings)
    
    plt.figure(figsize=(10, 8))
    df_plot = pd.DataFrame({
        'x': X_2d[:,0],
        'y': X_2d[:,1],
        'Domain': domain_labels_list
    })
    sns.scatterplot(data=df_plot, x='x', y='y', hue='Domain', alpha=0.7, palette='bright')
    plt.title("DANN V2 Latent Space (High Dim)")
    plt.savefig(f"{OUTPUT_DIR}/dann_embedding_space_v2.png", dpi=300)
    print(f"Saved t-SNE to {OUTPUT_DIR}/dann_embedding_space_v2.png")
    
    # Save Report
    with open(f"{OUTPUT_DIR}/evaluation_report_v2.txt", "w") as f:
        f.write("DANN V2 Evaluation Results\n")
        f.write("==========================\n")
        for k, v in results.items():
            f.write(f"{k}: {v*100:.2f}%\n")
        
        avg_acc = np.mean(list(results.values()))
        f.write(f"\nAverage Accuracy: {avg_acc*100:.2f}%\n")

if __name__ == "__main__":
    run_eval()
