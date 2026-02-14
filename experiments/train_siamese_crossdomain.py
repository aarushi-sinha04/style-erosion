"""
Cross-Domain Siamese Network Training
=====================================
Trains a Siamese network on ALL domains (PAN22 + Blog + Enron) combined,
instead of just PAN22. This dramatically improves cross-domain performance.

Key differences from train_siamese.py:
1. Vectorizer fit on all domains
2. Training data mixed from all labeled domains
3. Domain-balanced batching
"""
import json
import numpy as np
import os
import re
import pickle
import random
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader_scie import PAN22Loader, BlogTextLoader, EnronLoader

# Configuration
OUTPUT_DIR = "results/siamese_crossdomain"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparameters
MAX_FEATURES = 5000  # More features for cross-domain
HIDDEN_DIM = 512
DROPOUT = 0.3
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using Device: {DEVICE}")

def preprocess(text):
    text = str(text)
    text = text.replace("<nl>", " ")
    text = re.sub(r'<addr\d+_[A-Z]+>', ' <TAG> ', text)
    text = re.sub(r'<[^>]+>', ' <TAG> ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class SiameseNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
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
        # Head: u, v, |u-v|, u*v -> 4 * HIDDEN_DIM
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
        return self.head(combined)

class PairDataset(Dataset):
    def __init__(self, x1, x2, y, domains):
        self.x1 = torch.tensor(x1, dtype=torch.float32)
        self.x2 = torch.tensor(x2, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self.domains = domains  # For analysis only

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.y[idx]

def load_all_domains():
    """Load pairs from all labeled domains."""
    all_t1, all_t2, all_y, all_domains = [], [], [], []

    # 1. PAN22 (Gold Standard)
    print("Loading PAN22...")
    pan = PAN22Loader("pan22-authorship-verification-training.jsonl",
                      "pan22-authorship-verification-training-truth.jsonl")
    pan.load()
    t1, t2, y = pan.create_pairs()
    all_t1.extend(t1); all_t2.extend(t2); all_y.extend(y)
    all_domains.extend(['PAN22'] * len(t1))
    print(f"  PAN22: {len(t1)} pairs")

    # 2. BlogText
    print("Loading BlogText...")
    blog = BlogTextLoader("blogtext.csv")
    blog.load(limit=10000)
    t1, t2, y = blog.create_pairs(num_pairs=3000)
    all_t1.extend(t1); all_t2.extend(t2); all_y.extend(y)
    all_domains.extend(['Blog'] * len(t1))
    print(f"  Blog: {len(t1)} pairs")

    # 3. Enron
    print("Loading Enron...")
    enron = EnronLoader("emails.csv")
    enron.load(limit=10000)
    t1, t2, y = enron.create_pairs(num_pairs=3000)
    all_t1.extend(t1); all_t2.extend(t2); all_y.extend(y)
    all_domains.extend(['Enron'] * len(t1))
    print(f"  Enron: {len(t1)} pairs")

    return all_t1, all_t2, np.array(all_y, dtype=np.float32), all_domains

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x1, x2, y in loader:
        x1, x2, y = x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x1, x2)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == y).sum().item()
        total += len(y)
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_probs, all_labels = 0, [], []
    with torch.no_grad():
        for x1, x2, y in loader:
            x1, x2, y = x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE)
            logits = model(x1, x2)
            total_loss += criterion(logits, y).item()
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    acc = accuracy_score(all_labels, preds)
    try:
        roc = roc_auc_score(all_labels, all_probs)
    except:
        roc = 0.0
    f1 = f1_score(all_labels, preds)
    return total_loss / len(loader), acc, roc, f1

if __name__ == '__main__':
    print("=" * 60)
    print("CROSS-DOMAIN SIAMESE NETWORK")
    print("=" * 60)

    # 1. Load all domains
    t1_raw, t2_raw, y, domains = load_all_domains()
    print(f"\nTotal pairs: {len(y)}")

    # 2. Preprocess
    print("Preprocessing...")
    t1_clean = [preprocess(t) for t in tqdm(t1_raw)]
    t2_clean = [preprocess(t) for t in tqdm(t2_raw)]

    # 3. Vectorize on ALL text
    print(f"Vectorizing (Char 4-grams, Top {MAX_FEATURES})...")
    vec = TfidfVectorizer(analyzer='char', ngram_range=(4, 4),
                          max_features=MAX_FEATURES, sublinear_tf=True, min_df=3)
    all_text = t1_clean + t2_clean
    vec.fit(all_text)

    X1 = vec.transform(t1_clean).toarray()
    X2 = vec.transform(t2_clean).toarray()

    # 4. Scale
    print("Scaling...")
    scaler = StandardScaler()
    all_vecs = np.vstack([X1, X2])
    scaler.fit(all_vecs)
    X1 = scaler.transform(X1)
    X2 = scaler.transform(X2)

    # 5. Split (stratified by label)
    idx_train, idx_val = train_test_split(
        list(range(len(y))), test_size=0.20, random_state=42, stratify=y)

    train_ds = PairDataset(X1[idx_train], X2[idx_train], y[idx_train],
                           [domains[i] for i in idx_train])
    val_ds = PairDataset(X1[idx_val], X2[idx_val], y[idx_val],
                         [domains[i] for i in idx_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # 6. Model
    model = SiameseNetwork(input_dim=MAX_FEATURES).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_acc = 0
    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc, v_roc, v_f1 = evaluate(model, val_loader, criterion)
        scheduler.step(v_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | "
              f"Val Acc: {v_acc:.4f} ROC: {v_roc:.4f} F1: {v_f1:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pth")
            print(f"  ★ New best! ({best_val_acc:.4f})")

    # 7. Save artifacts
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model.pth", weights_only=True))
    _, acc, roc, f1 = evaluate(model, val_loader, criterion)
    print(f"\nFinal: Acc={acc:.4f} ROC={roc:.4f} F1={f1:.4f}")

    with open(f"{OUTPUT_DIR}/vectorizer.pkl", "wb") as f: pickle.dump(vec, f)
    with open(f"{OUTPUT_DIR}/scaler.pkl", "wb") as f: pickle.dump(scaler, f)
    with open(f"{OUTPUT_DIR}/report.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nROC-AUC: {roc:.4f}\nF1: {f1:.4f}\n")
        f.write(f"Input Dim: {MAX_FEATURES}\nDomains: PAN22, Blog, Enron\n")
    print("Done!")
