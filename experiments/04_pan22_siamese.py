import json
import numpy as np
import os
import re
import pickle
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Configuration
DATA_DIR = "."
OUTPUT_DIR = "results_pan_siamese"
TRAIN_FILE = "pan22-authorship-verification-training.jsonl"
TRUTH_FILE = "pan22-authorship-verification-training-truth.jsonl"

# Hyperparameters
MAX_FEATURES = 3000  # Input dimension
HIDDEN_DIM = 512
DROPOUT = 0.3
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Using Device: {DEVICE}")

# ==============================================================================
# 1. DATA LOADING
# ==============================================================================
def load_pan_data(data_path, truth_path):
    print(f"Loading {data_path}...")
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

def preprocess(text):
    text = text.replace("<nl>", " ")
    text = re.sub(r'<addr\d+_[A-Z]+>', ' <TAG> ', text)
    text = re.sub(r'<[^>]+>', ' <TAG> ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==============================================================================
# 2. PYTORCH DATASET & MODEL
# ==============================================================================
class PANDataset(Dataset):
    def __init__(self, x1, x2, y):
        # x1, x2 are dense numpy arrays or sparse matrices -> convert to dense tensor
        self.x1 = torch.tensor(x1, dtype=torch.float32)
        self.x2 = torch.tensor(x2, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.y[idx]

class SiameseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SiameseNetwork, self).__init__()
        
        # Branch Network (Feature Extractor)
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
        
        # Head Network (Classifier)
        # Inputs: u, v, |u-v|, u*v -> 4 * HIDDEN_DIM
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Sigmoid applied in loss/predict
        )
        
    def forward_one(self, x):
        return self.branch(x)

    def forward(self, x1, x2):
        u = self.forward_one(x1)
        v = self.forward_one(x2)
        
        # Interaction Features
        diff = torch.abs(u - v)
        prod = u * v
        
        # Concat
        combined = torch.cat([u, v, diff, prod], dim=1)
        
        logits = self.head(combined)
        return logits

# ==============================================================================
# 3. TRAINING LOOP
# ==============================================================================
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for x1, x2, y in loader:
        x1, x2, y = x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(x1, x2)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(y.detach().cpu().numpy())
        
    avg_loss = total_loss / len(loader)
    preds_binary = [1 if p >= 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_labels, preds_binary)
    return avg_loss, acc

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for x1, x2, y in loader:
            x1, x2, y = x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE)
            logits = model(x1, x2)
            loss = criterion(logits, y)
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.cpu().numpy())
            
    avg_loss = total_loss / len(loader)
    preds_binary = [1 if p >= 0.5 else 0 for p in all_probs]
    
    acc = accuracy_score(all_labels, preds_binary)
    roc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, preds_binary)
    
    return avg_loss, acc, roc, f1

# ==============================================================================
# 4. PLOTTING FUNCTIONS
# ==============================================================================
def plot_training_curves(history):
    epochs = range(1, len(history['train_loss']) + 1)
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

def plot_final_metrics(model, loader):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for x1, x2, y in loader:
            x1, x2, y = x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE)
            logits = model(x1, x2)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.cpu().numpy())
            
    y_scores = np.array(all_probs)
    y_true = np.array(all_labels)
    y_pred = (y_scores >= 0.5).astype(int)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
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
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
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

    print("Saved roc_curve.png")

def get_ngram_category(ngram):
    """Categorize the n-gram for better interpretability."""
    import string
    
    # Check for emojis or non-ascii/non-punctuation
    if any(ord(c) > 127 for c in ngram):
        return "Emoji / Special"
        
    has_punct = any(c in string.punctuation for c in ngram)
    has_space = " " in ngram
    has_alpha = any(c.isalpha() for c in ngram)
    has_digit = any(c.isdigit() for c in ngram)
    
    if has_digit: return "Numeric"
    if not has_alpha and has_punct: return "Pure Punctuation"
    if has_space and has_punct: return "Punctuation + Boundary"
    if has_space: return "Word Boundary"
    if has_punct: return "Contraction / Punct"
    return "Word Part"

def analyze_feature_importance(model, vectorizer):
    """
    Analyzes the first layer weights of the Siamese Branch to determine
    which N-grams contribute most to the embedding generation.
    """
    print("Analyzing feature importance...")
    
    # 1. Get Weights from First Linear Layer [Out_Dim, In_Dim]
    # model.branch[0] is the first Linear layer
    weights = model.branch[0].weight.detach().cpu().numpy()
    
    # 2. Calculate Mean Absolute Importance per Input Feature
    # We average across all 1024 output neurons to see generally useful features
    feature_importance = np.mean(np.abs(weights), axis=0)
    
    # 3. Map to Feature Names
    feature_names = vectorizer.get_feature_names_out()
    
    # 4. Sort
    indices = np.argsort(feature_importance)[::-1]
    top_n = 50
    
    # 5. Print to Text File
    with open(f"{OUTPUT_DIR}/feature_importance.txt", "w") as f:
        f.write("TOP 50 MOST IMPORTANT CHAR N-GRAMS\n")
        f.write("============================================================\n")
        f.write("These are 4-character sequences found in the text.\n")
        f.write("'␣' represents a Space. These patterns capture subconscious style.\n")
        f.write("============================================================\n")
        f.write(f"{'Rank':<5} | {'Feature':<10} | {'Category':<22} | {'Score':<8}\n")
        f.write("-" * 60 + "\n")
        
        for i, idx in enumerate(indices[:top_n]):
            feat = feature_names[idx]
            score = feature_importance[idx]
            category = get_ngram_category(feat)
            
            # Replace spaces for readability with a clear symbol
            readable_feat = feat.replace(" ", "␣") 
            
            f.write(f"{i+1:<5d} | {readable_feat:10s} | {category:22s} | {score:.4f}\n")
    
    print(f"Saved extended feature analysis to {OUTPUT_DIR}/feature_importance.txt")
            
    # 6. Plot (Top 20)
    plt.figure(figsize=(10, 8))
    top_indices = indices[:20]
    top_scores = feature_importance[top_indices]
    top_names = [feature_names[i].replace(" ", "␣") for i in top_indices]
    
    sns.barplot(x=top_scores, y=top_names, palette="viridis", hue=top_names, legend=False)
    plt.title(f"Top 20 Stylometric Markers (Character 4-grams)", fontsize=14)
    plt.xlabel("Importance Score (Mean Absolute Weight)", fontsize=10)
    plt.ylabel("Character Sequence (␣ = Space)", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=300)
    print("Saved feature_importance.png")

# ==============================================================================
# 4. MAIN
# ==============================================================================
if __name__ == '__main__':
    print("="*80)
    print("PAN22 SIAMESE NETWORK (PyTorch)")
    print("="*80)
    
    # 1. Load
    t1_raw, t2_raw, y = load_pan_data(os.path.join(DATA_DIR, TRAIN_FILE), 
                                      os.path.join(DATA_DIR, TRUTH_FILE))
    
    print("Preprocessing...")
    t1_clean = [preprocess(t) for t in tqdm(t1_raw)]
    t2_clean = [preprocess(t) for t in tqdm(t2_raw)]
    
    # 2. Vectorize (Input Features)
    print(f"Vectorizing (Char 4-grams, Top {MAX_FEATURES})...")
    # Using Sublinear TF to dampen outlier counts
    vec = TfidfVectorizer(analyzer='char', ngram_range=(4, 4), 
                          max_features=MAX_FEATURES, sublinear_tf=True, min_df=5)
    
    # Fit on all text
    all_text = t1_clean + t2_clean
    vec.fit(all_text)
    
    X1 = vec.transform(t1_clean).toarray() # Dense for PyTorch
    X2 = vec.transform(t2_clean).toarray()
    
    # Scale Features (Critical for Neural Nets)
    print("Scaling features...")
    scaler = StandardScaler()
    # Fit on all vectors stacked
    all_vecs = np.vstack([X1, X2])
    scaler.fit(all_vecs)
    
    X1 = scaler.transform(X1)
    X2 = scaler.transform(X2)
    
    # 3. Split
    pairs_indices = list(range(len(y)))
    idx_train, idx_val = train_test_split(pairs_indices, test_size=0.20, random_state=42, stratify=y)
    
    train_ds = PANDataset(X1[idx_train], X2[idx_train], y[idx_train])
    val_ds = PANDataset(X1[idx_val], X2[idx_val], y[idx_val])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    
    # 4. Model Training
    model = SiameseNetwork(input_dim=MAX_FEATURES).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) # L2 reg
    
    best_val_acc = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    print("\nStarting Training...")
    
    for epoch in range(EPOCHS):
        t_loss, t_acc = train(model, train_loader, criterion, optimizer)
        v_loss, v_acc, v_roc, v_f1 = evaluate(model, val_loader, criterion)
        
        # Save history
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f} ROC: {v_roc:.4f}")
        
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pth")
            
    # Plot curves
    plot_training_curves(history)
            
    print("\n" + "#"*60)
    print("FINAL RESULTS (Best Model)")
    print("#"*60)
    # Load best
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model.pth", weights_only=True))
    _, acc, roc, f1 = evaluate(model, val_loader, criterion)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"ROC-AUC:   {roc:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("#"*60)
    
    # Plot Final Metrics
    print("Generating ROC and Confusion Matrix...")
    plot_final_metrics(model, val_loader)
    
    # Feature Importance
    analyze_feature_importance(model, vec)
    
    with open(f"{OUTPUT_DIR}/report_siamese.txt", "w") as f:
         f.write(f"Accuracy: {acc:.4f}\nROC-AUC: {roc:.4f}\nF1: {f1:.4f}\n")
         
    with open(f"{OUTPUT_DIR}/vectorizer.pkl", "wb") as f: pickle.dump(vec, f)
    with open(f"{OUTPUT_DIR}/scaler.pkl", "wb") as f: pickle.dump(scaler, f)
    
    print("Done!")
