import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import sys
from tqdm import tqdm

# Fix import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader_scie import PAN22Loader, BlogTextLoader, EnronLoader

# Config
MODEL_DIR = "results_pan_siamese"
OUTPUT_DIR = "results_diagnostics"
DATA_DIR = "."
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ==============================================================================
# MODEL DEF (Must match training)
# ==============================================================================
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

def preprocess(text):
    text = text.replace("<nl>", " ")
    text = re.sub(r'<addr\d+_[A-Z]+>', ' <TAG> ', text)
    text = re.sub(r'<[^>]+>', ' <TAG> ', text)
    return re.sub(r'\s+', ' ', text).strip()

def get_samples(loader_name, loader_obj, n=500):
    print(f"Loading {loader_name}...")
    try:
        if loader_name == "PAN22":
            df = loader_obj.load(limit=3000) 
        else:
            df = loader_obj.load(limit=3000)
            
        texts = df['text'].tolist()
        texts = [str(t) for t in texts if len(str(t)) > 100]
        
        if len(texts) > n:
            texts = np.random.choice(texts, n, replace=False).tolist()
            
        return texts
    except Exception as e:
        print(f"Error loading {loader_name}: {e}")
        return []

def run_viz():
    # 1. Load Resources
    print("Loading Model & Vectorizer...")
    with open(f"{MODEL_DIR}/vectorizer.pkl", "rb") as f: vec = pickle.load(f)
    with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f: scaler = pickle.load(f)
    
    model = SiameseNetwork(input_dim=3000).to(DEVICE)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/best_model.pth", map_location=DEVICE))
    model.eval()
    
    # 2. Load Data
    pan_loader = PAN22Loader("pan22-authorship-verification-training.jsonl")
    blog_loader = BlogTextLoader("blogtext.csv")
    enron_loader = EnronLoader("emails.csv")
    
    texts_pan = get_samples("PAN22", pan_loader, n=500)
    texts_blog = get_samples("BlogText", blog_loader, n=500)
    texts_enron = get_samples("Enron", enron_loader, n=500)
    
    if not texts_pan: return
    
    all_texts = texts_pan + texts_blog + texts_enron
    labels = ["PAN22"]*len(texts_pan) + ["BlogText"]*len(texts_blog) + ["Enron"]*len(texts_enron)
    
    print(f"Total samples: {len(all_texts)}")
    
    # 3. Extract Embeddings
    print("Vectorizing...")
    clean_texts = [preprocess(t) for t in tqdm(all_texts)]
    X_sparse = vec.transform(clean_texts)
    X_dense = X_sparse.toarray()
    X_scaled = scaler.transform(X_dense)
    
    print("Extracting Neural Embeddings...")
    embeddings = []
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, len(X_scaled), batch_size):
            batch = X_scaled[i:i+batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32).to(DEVICE)
            emb = model.forward_one(batch_tensor)
            embeddings.append(emb.cpu().numpy())
            
    embeddings = np.vstack(embeddings)
    print(f"Embedding Shape: {embeddings.shape}")
    
    # 4. t-SNE
    print("Running t-SNE on Embeddings...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_2d = tsne.fit_transform(embeddings)
    
    # 5. Plot
    plt.figure(figsize=(10, 8))
    df_plot = pd.DataFrame({
        'x': X_2d[:,0],
        'y': X_2d[:,1],
        'Domain': labels
    })
    
    sns.scatterplot(data=df_plot, x='x', y='y', hue='Domain', alpha=0.7, palette='deep')
    plt.title("Latent Space Visualization (Siamese Model Embeddings)")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    
    plot_path = f"{OUTPUT_DIR}/embedding_space.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot to {plot_path}")
    
    # 6. Save Report
    with open(f"{OUTPUT_DIR}/embedding_analysis.txt", "w") as f:
        f.write("Phase 1: Embedding Space Analysis\n")
        f.write("=================================\n")
        f.write("Observation: If domains form distinct clusters, the model has learned domain-specific features\n")
        f.write("instead of universal authorship features. This confirms 'Domain Shift' failure.\n")

if __name__ == "__main__":
    run_viz()
