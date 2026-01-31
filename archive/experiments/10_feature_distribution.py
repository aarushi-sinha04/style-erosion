import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from scipy.spatial.distance import jensenshannon
from collections import Counter

# Fix import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader_scie import PAN22Loader, BlogTextLoader, EnronLoader

# Config
DATA_DIR = "."
OUTPUT_DIR = "results_diagnostics"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

PAN_FILE = "pan22-authorship-verification-training.jsonl"
BLOG_FILE = "blogtext.csv"
ENRON_FILE = "emails.csv"

SAMPLES_PER_DOMAIN = 1000

def get_samples(loader_name, loader_obj):
    print(f"Loading {loader_name}...")
    try:
        if loader_name == "PAN22":
            df = loader_obj.load(limit=3000) # returns [id, text]
        else:
            df = loader_obj.load(limit=3000) # returns [id, text] or [id, text, ...]
            
        texts = df['text'].tolist()
        
        # Filter short
        texts = [t for t in texts if len(str(t)) > 100]
        
        # Sample
        if len(texts) > SAMPLES_PER_DOMAIN:
            texts = np.random.choice(texts, SAMPLES_PER_DOMAIN, replace=False).tolist()
            
        return texts
    except Exception as e:
        print(f"Error loading {loader_name}: {e}")
        return []

def run_analysis():
    # 1. Load Data
    pan_loader = PAN22Loader(PAN_FILE)
    blog_loader = BlogTextLoader(BLOG_FILE)
    enron_loader = EnronLoader(ENRON_FILE)
    
    texts_pan = get_samples("PAN22", pan_loader)
    texts_blog = get_samples("BlogText", blog_loader)
    texts_enron = get_samples("Enron", enron_loader)
    
    if not texts_pan or not texts_blog or not texts_enron:
        print("Failed to load adequate data. Check file paths.")
        return

    print(f"Samples: PAN={len(texts_pan)}, Blog={len(texts_blog)}, Enron={len(texts_enron)}")
    
    # 2. Vectorize (Character 4-grams)
    print("Extracting 4-grams...")
    # Use a shared vectorizer to get a common feature space
    all_texts = texts_pan + texts_blog + texts_enron
    labels = ["PAN22"]*len(texts_pan) + ["BlogText"]*len(texts_blog) + ["Enron"]*len(texts_enron)
    
    vec = CountVectorizer(analyzer='char', ngram_range=(4,4), max_features=1000)
    X = vec.fit_transform(all_texts).toarray()
    feature_names = vec.get_feature_names_out()
    
    # 3. Compute Feature Distributions (Distributions over the 1000 features)
    # Normalize per domain to get a probability distribution
    def get_dist(start_idx, end_idx):
        subset = X[start_idx:end_idx]
        mean_vec = subset.mean(axis=0) + 1e-10 # Add smoothing
        return mean_vec / mean_vec.sum()

    idx_pan = len(texts_pan)
    idx_blog = idx_pan + len(texts_blog)
    
    dist_pan = get_dist(0, idx_pan)
    dist_blog = get_dist(idx_pan, idx_blog)
    dist_enron = get_dist(idx_blog, len(all_texts))
    
    # 4. JS Divergence
    js_pan_blog = jensenshannon(dist_pan, dist_blog)
    js_pan_enron = jensenshannon(dist_pan, dist_enron)
    js_blog_enron = jensenshannon(dist_blog, dist_enron)
    
    print("\nJensen-Shannon Divergence (0=Same, 1=Different):")
    print(f"PAN vs Blog:  {js_pan_blog:.4f}")
    print(f"PAN vs Enron: {js_pan_enron:.4f}")
    print(f"Blog vs Enron:{js_blog_enron:.4f}")
    
    # 5. Top Features Analysis
    def get_top_k(dist, k=10):
        indices = np.argsort(dist)[::-1][:k]
        return [(feature_names[i], dist[i]) for i in indices]
        
    top_pan = get_top_k(dist_pan)
    top_blog = get_top_k(dist_blog)
    top_enron = get_top_k(dist_enron)
    
    # 6. t-SNE Visualization
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    
    # Plot
    plt.figure(figsize=(10, 8))
    df_plot = pd.DataFrame({
        'x': X_embedded[:,0],
        'y': X_embedded[:,1],
        'Domain': labels
    })
    sns.scatterplot(data=df_plot, x='x', y='y', hue='Domain', alpha=0.6, palette='bright')
    plt.title("t-SNE of Character 4-gram Features by Domain")
    plt.savefig(f"{OUTPUT_DIR}/feature_tsne.png", dpi=300)
    print(f"Saved t-SNE plot to {OUTPUT_DIR}/feature_tsne.png")
    
    # Save Report
    with open(f"{OUTPUT_DIR}/diagnostic_report.txt", "w") as f:
        f.write("Phase 1: Deep Diagnostic Analysis\n")
        f.write("=================================\n\n")
        f.write("Jensen-Shannon Divergence (Feature Distribution Shift):\n")
        f.write(f"- PAN22 vs BlogText: {js_pan_blog:.4f}\n")
        f.write(f"- PAN22 vs Enron:    {js_pan_enron:.4f}\n")
        f.write(f"- BlogText vs Enron: {js_blog_enron:.4f}\n\n")
        
        f.write("Top 10 Frequent 4-grams per Domain:\n")
        f.write("\nPAN22 (Fanfiction):\n")
        for ft, sc in top_pan: f.write(f"  '{ft}': {sc:.4f}\n")
        f.write("\nBlogText (Personal):\n")
        for ft, sc in top_blog: f.write(f"  '{ft}': {sc:.4f}\n")
        f.write("\nEnron (Business Email):\n")
        for ft, sc in top_enron: f.write(f"  '{ft}': {sc:.4f}\n")

if __name__ == "__main__":
    run_analysis()
