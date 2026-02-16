import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import json
import os
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Config
OUTPUT_DIR = "results_pan_siamese"
DATA_DIR = "."
TRAIN_FILE = "pan22-authorship-verification-training.jsonl"
TRUTH_FILE = "pan22-authorship-verification-training-truth.jsonl"
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

# ==============================================================================
# 1. HARDCODED HISTORY (From the successful run)
# ==============================================================================
# Epochs 1-15
history = {
    'train_loss': [0.6904, 0.6592, 0.6164, 0.5416, 0.4409, 0.3297, 0.2317, 0.1734, 0.1420, 0.1116, 0.0907, 0.0777, 0.0651, 0.0690, 0.0611],
    'val_loss':   [0.6786, 0.6663, 0.6253, 0.5273, 0.4282, 0.3489, 0.3400, 0.2737, 0.3134, 0.2980, 0.3184, 0.3194, 0.2987, 0.2767, 0.2881],
    'train_acc':  [0.5255, 0.6023, 0.6387, 0.7007, 0.7816, 0.8485, 0.9045, 0.9301, 0.9435, 0.9571, 0.9645, 0.9711, 0.9767, 0.9754, 0.9774],
    'val_acc':    [0.5593, 0.5597, 0.5752, 0.6824, 0.7766, 0.8296, 0.8536, 0.8903, 0.8736, 0.8899, 0.8875, 0.8924, 0.9050, 0.9087, 0.9099]
}
epochs = range(1, 16)

def plot_training_clean():
    # Style
    plt.style.use('seaborn-v0_8-paper')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax1.plot(epochs, history['train_acc'], label='Training Accuracy',  linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs, history['val_acc'], label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
    ax1.set_title('Model Accuracy over Epochs', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Accuracy', fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(epochs, history['train_loss'], label='Training Loss', linewidth=2, linestyle='--', color='salmon')
    ax2.plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2, color='darkred')
    ax2.set_title('Cross-Entropy Loss over Epochs', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Loss', fontsize=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/training_curves.png", dpi=300)
    print("Saved training_curves.png")

# ==============================================================================
# 2. MODEL INFRASTRUCTURE (Needed to load model)
# ==============================================================================
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SiameseNetwork, self).__init__()
        HIDDEN_DIM = 512
        DROPOUT = 0.3
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
    def forward_one(self, x): return self.branch(x)
    def forward(self, x1, x2):
        u = self.forward_one(x1)
        v = self.forward_one(x2)
        diff = torch.abs(u - v)
        prod = u * v
        combined = torch.cat([u, v, diff, prod], dim=1)
        return self.head(combined)

def preprocess(text):
    text = text.replace("<nl>", " ")
    text = re.sub(r'<addr\d+_[A-Z]+>', ' <TAG> ', text)
    text = re.sub(r'<[^>]+>', ' <TAG> ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_pan_data(data_path, truth_path):
    print("Loading data...")
    pairs_dict = {}
    with open(data_path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            pairs_dict[obj['id']] = obj['pair']
    labels_dict = {}
    with open(truth_path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            labels_dict[obj['id']] = 1.0 if obj['same'] else 0.0
    ids = sorted(list(pairs_dict.keys()))
    t1s, t2s, ys = [], [], []
    for i in ids:
        if i in labels_dict:
            t1, t2 = pairs_dict[i]
            t1s.append(t1)
            t2s.append(t2)
            ys.append(labels_dict[i])
    return t1s, t2s, np.array(ys, dtype=np.float32)

# ==============================================================================
# 3. GENERATE METRIC PLOTS
# ==============================================================================
def plot_metrics_from_model():
    # Load Real Data Validation Set
    t1_raw, t2_raw, y = load_pan_data(os.path.join(DATA_DIR, TRAIN_FILE), 
                                      os.path.join(DATA_DIR, TRUTH_FILE))
    
    t1_clean = [preprocess(t) for t in t1_raw]
    t2_clean = [preprocess(t) for t in t2_raw]
    
    # Load Artifacts
    print("Loading artifacts...")
    with open(f"{OUTPUT_DIR}/vectorizer.pkl", "rb") as f: vec = pickle.load(f)
    with open(f"{OUTPUT_DIR}/scaler.pkl", "rb") as f: scaler = pickle.load(f)
    
    # Transform
    print("Vectorizing...")
    X1 = scaler.transform(vec.transform(t1_clean).toarray())
    X2 = scaler.transform(vec.transform(t2_clean).toarray())
    
    # Split to get Validation Set (Must match training split seed=42)
    pairs_indices = list(range(len(y)))
    _, idx_val = train_test_split(pairs_indices, test_size=0.20, random_state=42, stratify=y)
    
    X1_val = torch.tensor(X1[idx_val], dtype=torch.float32).to(DEVICE)
    X2_val = torch.tensor(X2[idx_val], dtype=torch.float32).to(DEVICE)
    y_val = y[idx_val]
    
    # Load Model
    model = SiameseNetwork(input_dim=3000).to(DEVICE)
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model.pth", map_location=DEVICE))
    model.eval()
    
    print("Predicting...")
    batch_size = 256
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(y_val), batch_size):
            batch_x1 = X1_val[i:i+batch_size]
            batch_x2 = X2_val[i:i+batch_size]
            logits = model(batch_x1, batch_x2)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            
    y_scores = np.array(all_probs)
    y_pred = (y_scores >= 0.5).astype(int)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Diff Author', 'Same Author'],
                yticklabels=['Diff Author', 'Same Author'])
    plt.title('Confusion Matrix', fontsize=12, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=300)
    print("Saved confusion_matrix.png")
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=12, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/roc_curve.png", dpi=300)
    print("Saved roc_curve.png")

if __name__ == "__main__":
    plot_training_clean()
    plot_metrics_from_model()
