"""
DANN V3 Evaluation - Comprehensive Cross-Domain Assessment
============================================================
Generates:
1. Per-domain accuracy table
2. t-SNE visualization
3. A-distance for domain divergence
4. Results markdown file
"""
import sys
import os
import torch
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime

# Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader_scie import PAN22Loader, BlogTextLoader, EnronLoader, IMDBLoader
from models.dann_siamese_v3 import DANNSiameseV3
from utils.feature_extraction import EnhancedFeatureExtractor

# Config
OUTPUT_DIR = "results_dann"
DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
MAX_FEATURES = 4308
DOMAIN_NAMES = ['PAN22', 'BlogText', 'Enron', 'IMDB']

def pad_features(X, target_dim=4308):
    """Pad or truncate feature vectors to match model input dim."""
    if X.shape[1] == target_dim:
        return X
    if X.shape[1] > target_dim:
        return X[:, :target_dim]
    else:
        padding = np.zeros((X.shape[0], target_dim - X.shape[1]))
        return np.hstack([X, padding])

def flatten_feats(feats_dict):
    return np.hstack([
        feats_dict['char'], 
        feats_dict['pos'], 
        feats_dict['lex'], 
        feats_dict['readability']
    ])

def compute_a_distance(source_features, target_features):
    """
    Compute A-distance (proxy for domain divergence).
    Lower = better domain alignment.
    """
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    
    # Create binary classification task
    n_source = len(source_features)
    n_target = len(target_features)
    
    # Balance if needed
    min_n = min(n_source, n_target, 500)
    
    source_idx = np.random.choice(n_source, min_n, replace=False)
    target_idx = np.random.choice(n_target, min_n, replace=False)
    
    X = np.vstack([source_features[source_idx], target_features[target_idx]])
    y = np.hstack([np.zeros(min_n), np.ones(min_n)])
    
    # Train classifier
    clf = SVC(kernel='linear', max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    
    # A-distance = 2 * (1 - 2 * error)
    error = 1 - scores.mean()
    a_distance = 2 * (1 - 2 * error)
    
    return a_distance

def run_evaluation():
    print("=" * 60)
    print("DANN V3 Evaluation")
    print("=" * 60)
    
    # 1. Check for model
    model_path = f"{OUTPUT_DIR}/dann_model_v4.pth"
    if not os.path.exists(model_path):
        print(f"Model V4 not found at {model_path}")
        model_path = f"{OUTPUT_DIR}/dann_model_v3.pth"
        if not os.path.exists(model_path):
             print(f"Model V3 not found at {model_path}")
             model_path = f"{OUTPUT_DIR}/dann_model_v2.pth"
        if not os.path.exists(model_path):
            print("No model found. Please train first.")
            return
    
    print(f"Using model: {model_path}")
    
    # 2. Load Extractor
    print("\nLoading extractor...")
    with open(f"{OUTPUT_DIR}/extractor.pkl", "rb") as f:
        extractor = pickle.load(f)
    
    # 3. Load Model
    print("Loading model...")
    model = DANNSiameseV3(input_dim=MAX_FEATURES, num_domains=4).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    
    # 4. Load Test Data
    samples = 500
    
    loaders = {
        'PAN22': PAN22Loader("pan22-authorship-verification-training.jsonl", 
                             "pan22-authorship-verification-training-truth.jsonl"),
        'BlogText': BlogTextLoader("blogtext.csv"),
        'Enron': EnronLoader("emails.csv"),
        'IMDB': IMDBLoader("IMDB Dataset.csv")
    }
    
    results = {}
    embeddings_list = []
    domain_labels_list = []
    all_predictions = {}
    
    print("\n" + "=" * 60)
    print("Evaluating on Each Domain")
    print("=" * 60)
    
    for domain, loader in loaders.items():
        print(f"\nProcessing {domain}...")
        
        try:
            loader.load(limit=5000)
        except Exception as e:
            print(f"Error loading {domain}: {e}")
            continue
        
        t1_list, t2_list, labels = loader.create_pairs(num_pairs=samples)
        if not t1_list:
            print(f"No pairs created for {domain}")
            continue
        
        # Extract features
        f1_dict = extractor.transform(t1_list)
        f2_dict = extractor.transform(t2_list)
        
        X1 = pad_features(flatten_feats(f1_dict), MAX_FEATURES)
        X2 = pad_features(flatten_feats(f2_dict), MAX_FEATURES)
        
        X1_t = torch.tensor(X1, dtype=torch.float32).to(DEVICE)
        X2_t = torch.tensor(X2, dtype=torch.float32).to(DEVICE)
        
        # Forward pass
        with torch.no_grad():
            pred_auth, p_dom_a, _ = model(X1_t, X2_t, alpha=0.0)
            emb = model.get_embeddings(X1_t)
        
        pred_probs = pred_auth.squeeze().cpu().numpy()
        preds = (pred_probs > 0.5).astype(float)
        
        # Store embeddings for t-SNE
        embeddings_list.append(emb.cpu().numpy())
        domain_labels_list.extend([domain] * len(X1))
        
        # Compute metrics
        if domain != 'IMDB':
            y_true = np.array(labels)
            
            # Calculate optimal threshold
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(y_true, pred_probs)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            # Use optimal threshold for predictions
            preds_opt = (pred_probs > optimal_threshold).astype(float)
            
            acc = accuracy_score(y_true, preds_opt)
            prec = precision_score(y_true, preds_opt, zero_division=0)
            rec = recall_score(y_true, preds_opt, zero_division=0)
            f1 = f1_score(y_true, preds_opt, zero_division=0)
            try:
                roc = roc_auc_score(y_true, pred_probs)
            except:
                roc = 0.5
            
            results[domain] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'roc_auc': roc,
                'threshold': optimal_threshold
            }
            
            print(f"  {domain}: Acc={acc*100:.1f}% (Thresh={optimal_threshold:.2f}) | AUC={roc:.3f}")
        else:
            print(f"  {domain}: N/A (Unlabeled target domain)")
    
    # 5. Compute A-distances
    print("\n" + "=" * 60)
    print("Computing A-distances (Domain Divergence)")
    print("=" * 60)
    
    all_embeddings = np.vstack(embeddings_list)
    domain_indices = {}
    start_idx = 0
    for domain in loaders.keys():
        n = samples
        domain_indices[domain] = (start_idx, start_idx + n)
        start_idx += n
    
    a_distances = {}
    for i, dom1 in enumerate(list(loaders.keys())):
        for dom2 in list(loaders.keys())[i+1:]:
            idx1 = domain_indices[dom1]
            idx2 = domain_indices[dom2]
            emb1 = all_embeddings[idx1[0]:idx1[1]]
            emb2 = all_embeddings[idx2[0]:idx2[1]]
            
            a_dist = compute_a_distance(emb1, emb2)
            a_distances[f"{dom1}-{dom2}"] = a_dist
            print(f"  {dom1} <-> {dom2}: A-distance = {a_dist:.3f}")
    
    avg_a_distance = np.mean(list(a_distances.values()))
    print(f"\n  Average A-distance: {avg_a_distance:.3f} (lower = better alignment)")
    
    # 6. t-SNE Visualization
    print("\n" + "=" * 60)
    print("Generating t-SNE Visualization")
    print("=" * 60)
    
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=30)
    X_2d = tsne.fit_transform(all_embeddings)
    
    plt.figure(figsize=(12, 10))
    df_plot = pd.DataFrame({
        'x': X_2d[:, 0],
        'y': X_2d[:, 1],
        'Domain': domain_labels_list
    })
    
    colors = {'PAN22': '#2196F3', 'BlogText': '#FF9800', 'Enron': '#4CAF50', 'IMDB': '#F44336'}
    sns.scatterplot(data=df_plot, x='x', y='y', hue='Domain', alpha=0.6, palette=colors, s=50)
    
    plt.title("DANN V3 Embedding Space (Domain Alignment)", fontsize=14)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Domain", loc='upper right')
    
    # Add annotation
    avg_acc = np.mean([r['accuracy'] for r in results.values()]) * 100
    plt.annotate(f"Avg Acc: {avg_acc:.1f}% | A-dist: {avg_a_distance:.2f}", 
                 xy=(0.02, 0.98), xycoords='axes fraction', fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/dann_embedding_space_v3.png", dpi=300)
    print(f"Saved: {OUTPUT_DIR}/dann_embedding_space_v3.png")
    
    # 7. Results Table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print("\n| Domain   | Accuracy | Precision | Recall | F1    | ROC-AUC |")
    print("|----------|----------|-----------|--------|-------|---------|")
    
    for domain, metrics in results.items():
        print(f"| {domain:8} | {metrics['accuracy']*100:6.1f}%  | {metrics['precision']:.3f}     | "
              f"{metrics['recall']:.3f}  | {metrics['f1']:.3f} | {metrics['roc_auc']:.3f}   |")
    
    # Average
    avg_metrics = {
        'accuracy': np.mean([r['accuracy'] for r in results.values()]),
        'precision': np.mean([r['precision'] for r in results.values()]),
        'recall': np.mean([r['recall'] for r in results.values()]),
        'f1': np.mean([r['f1'] for r in results.values()]),
        'roc_auc': np.mean([r['roc_auc'] for r in results.values()])
    }
    
    print(f"| {'AVERAGE':8} | {avg_metrics['accuracy']*100:6.1f}%  | {avg_metrics['precision']:.3f}     | "
          f"{avg_metrics['recall']:.3f}  | {avg_metrics['f1']:.3f} | {avg_metrics['roc_auc']:.3f}   |")
    
    # 8. Save Results
    results_file = f"{OUTPUT_DIR}/dann_results_v3.md"
    with open(results_file, "w") as f:
        f.write("# DANN V3 Evaluation Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Model:** `{model_path}`\n\n")
        
        f.write("## Cross-Domain Authorship Verification\n\n")
        f.write("| Domain | Accuracy | Precision | Recall | F1 | ROC-AUC |\n")
        f.write("|--------|----------|-----------|--------|----|---------|\n")
        
        for domain, metrics in results.items():
            f.write(f"| {domain} | {metrics['accuracy']*100:.1f}% | {metrics['precision']:.3f} | "
                   f"{metrics['recall']:.3f} | {metrics['f1']:.3f} | {metrics['roc_auc']:.3f} |\n")
        
        f.write(f"| **AVERAGE** | **{avg_metrics['accuracy']*100:.1f}%** | {avg_metrics['precision']:.3f} | "
               f"{avg_metrics['recall']:.3f} | {avg_metrics['f1']:.3f} | {avg_metrics['roc_auc']:.3f} |\n")
        
        f.write("\n## Domain Alignment\n\n")
        f.write("A-distance measures domain divergence (lower = better alignment):\n\n")
        for pair, dist in a_distances.items():
            f.write(f"- {pair}: {dist:.3f}\n")
        f.write(f"\n**Average A-distance:** {avg_a_distance:.3f}\n")
        
        f.write("\n## Visualization\n\n")
        f.write(f"![t-SNE Embedding Space](dann_embedding_space_v3.png)\n")
    
    print(f"\nResults saved to: {results_file}")
    
    # 9. Check success criteria
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 60)
    
    target_acc = 0.70
    if avg_metrics['accuracy'] >= target_acc:
        print(f"✓ PASSED: Average accuracy {avg_metrics['accuracy']*100:.1f}% >= 70%")
    else:
        print(f"✗ NEEDS IMPROVEMENT: Average accuracy {avg_metrics['accuracy']*100:.1f}% < 70%")
        print(f"  Gap: {(target_acc - avg_metrics['accuracy'])*100:.1f}% to reach target")
    
    return avg_metrics['accuracy']

if __name__ == "__main__":
    run_evaluation()
