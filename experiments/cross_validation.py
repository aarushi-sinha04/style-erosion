"""
5-Fold Cross Validation
=====================
Addresses the "single train/test split" criticism by running 5-fold CV
for the Rob Siamese and Base DANN models.
"""
import sys
import os
import json
import numpy as np
import random
from collections import defaultdict
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader_scie import PAN22Loader, BlogTextLoader, EnronLoader
from models.dann import DANNSiameseV3
from experiments.train_siamese_crossdomain import SiameseNetwork, PairDataset, preprocess
from utils.feature_extraction import EnhancedFeatureExtractor

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

# Reduced epochs for CV speed
EPOCHS_SIAMESE = 5
EPOCHS_DANN = 5
BATCH_SIZE = 64

def flatten_feats(feats_dict):
    return np.hstack([
        feats_dict['char'], 
        feats_dict['pos'], 
        feats_dict['lex'], 
        feats_dict['readability']
    ])

def train_siam(train_loader, val_loader, input_dim):
    model = SiameseNetwork(input_dim=input_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(EPOCHS_SIAMESE):
        model.train()
        for x1, x2, y in train_loader:
            x1, x2, y = x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x1, x2)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
    # Eval
    model.eval()
    all_preds, all_probs, all_y = [], [], []
    with torch.no_grad():
        for x1, x2, y in val_loader:
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            logits = model(x1, x2)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_y.extend(y.numpy().tolist())
            
    acc = accuracy_score(all_y, all_preds)
    f1 = f1_score(all_y, all_preds, zero_division=0)
    auc = roc_auc_score(all_y, all_probs)
    return acc, f1, auc

def train_dann(train_loader, val_loader, input_dim):
    model = DANNSiameseV3(input_dim=input_dim, num_domains=3).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    criterion_bce = nn.BCELoss()
    
    for epoch in range(EPOCHS_DANN):
        model.train()
        for x1, x2, y, d in train_loader:
            x1, x2, y = x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            p_auth, _, _ = model(x1, x2, alpha=0.0)
            p_auth = p_auth.squeeze()
            
            mask = (y != -1).float()
            l_auth_raw = nn.BCELoss(reduction='none')(p_auth, y)
            l_auth = (l_auth_raw * mask).sum() / (mask.sum() + 1e-8)
            
            l_auth.backward()
            optimizer.step()
            
    # Eval
    model.eval()
    all_preds, all_probs, all_y = [], [], []
    with torch.no_grad():
        for x1, x2, y, _ in val_loader:
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            p_auth, _, _ = model(x1, x2, alpha=0.0)
            probs = p_auth.squeeze().cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_y.extend(y.numpy().tolist())
            
    acc = accuracy_score(all_y, all_preds)
    f1 = f1_score(all_y, all_preds, zero_division=0)
    auc = roc_auc_score(all_y, all_probs)
    return acc, f1, auc

def main():
    print("=" * 60)
    print("5-FOLD CROSS VALIDATION")
    print("=" * 60)
    
    domain_loaders = [
        ('PAN22', PAN22Loader("data/raw/pan22_texts.jsonl", "data/raw/pan22_labels.jsonl")),
        ('Blog', BlogTextLoader("data/raw/blogtext.csv")),
        ('Enron', EnronLoader("data/raw/emails.csv")),
    ]
    
    all_t1, all_t2, all_y, all_domains = [], [], [], []
    
    print("Loading data...")
    for i, (name, loader) in enumerate(domain_loaders):
        loader.load(limit=1500)
        t1, t2, y = loader.create_pairs(num_pairs=400)
        all_t1.extend(t1)
        all_t2.extend(t2)
        all_y.extend(y)
        all_domains.extend([i] * len(y))
        
    all_t1 = np.array(all_t1)
    all_t2 = np.array(all_t2)
    all_y = np.array(all_y)
    all_domains = np.array(all_domains)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {
        'Rob Siamese': {'acc': [], 'f1': [], 'auc': []},
        'Base DANN': {'acc': [], 'f1': [], 'auc': []}
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_t1, all_y)):
        print(f"\n" + "=" * 40)
        print(f"FOLD {fold + 1}/5")
        print("=" * 40)
        
        train_t1, val_t1 = all_t1[train_idx], all_t1[val_idx]
        train_t2, val_t2 = all_t2[train_idx], all_t2[val_idx]
        train_y, val_y = all_y[train_idx], all_y[val_idx]
        train_d, val_d = all_domains[train_idx], all_domains[val_idx]
        
        # ----------------------------------------------------
        # 1. Siamese Model (Rob Siamese simplified proxy)
        # ----------------------------------------------------
        print("  Training Siamese...")
        vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=5000, lowercase=True)
        scaler = MaxAbsScaler()
        
        # Preprocess
        pt1 = [preprocess(t) for t in train_t1]
        pt2 = [preprocess(t) for t in train_t2]
        pv1 = [preprocess(t) for t in val_t1]
        pv2 = [preprocess(t) for t in val_t2]
        
        vec.fit(pt1 + pt2)
        X_train1 = scaler.fit_transform(vec.transform(pt1).toarray())
        X_train2 = scaler.transform(vec.transform(pt2).toarray())
        X_val1 = scaler.transform(vec.transform(pv1).toarray())
        X_val2 = scaler.transform(vec.transform(pv2).toarray())
        
        ds_train = PairDataset(X_train1, X_train2, train_y, train_d)
        ds_val = PairDataset(X_val1, X_val2, val_y, val_d)
        
        acc_s, f1_s, auc_s = train_siam(
            DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True),
            DataLoader(ds_val, batch_size=BATCH_SIZE),
            input_dim=5000
        )
        print(f"    Siamese Fold {fold+1}: Acc={acc_s:.3f}, F1={f1_s:.3f}, AUC={auc_s:.3f}")
        results['Rob Siamese']['acc'].append(acc_s)
        results['Rob Siamese']['f1'].append(f1_s)
        results['Rob Siamese']['auc'].append(auc_s)
        
        # ----------------------------------------------------
        # 2. Base DANN Model
        # ----------------------------------------------------
        print("  Training Base DANN...")
        extractor = EnhancedFeatureExtractor(max_features_char=3000, max_features_pos=1000, max_features_lex=300)
        # Fit extractor on subset to save time
        sample_texts = [str(x) for x in list(train_t1[:500]) + list(train_t2[:500])]
        extractor.fit(sample_texts)
        
        f1_train = extractor.transform([str(x) for x in train_t1])
        f2_train = extractor.transform([str(x) for x in train_t2])
        f1_val = extractor.transform([str(x) for x in val_t1])
        f2_val = extractor.transform([str(x) for x in val_t2])
        
        Xd_train1 = flatten_feats(f1_train)
        Xd_train2 = flatten_feats(f2_train)
        Xd_val1 = flatten_feats(f1_val)
        Xd_val2 = flatten_feats(f2_val)
        
        td_train1 = torch.tensor(Xd_train1, dtype=torch.float32)
        td_train2 = torch.tensor(Xd_train2, dtype=torch.float32)
        yd_train = torch.tensor(train_y, dtype=torch.float32)
        dd_train = torch.tensor(train_d, dtype=torch.long)
        
        td_val1 = torch.tensor(Xd_val1, dtype=torch.float32)
        td_val2 = torch.tensor(Xd_val2, dtype=torch.float32)
        yd_val = torch.tensor(val_y, dtype=torch.float32)
        dd_val = torch.tensor(val_d, dtype=torch.long)
        
        ds_dann_train = TensorDataset(td_train1, td_train2, yd_train, dd_train)
        ds_dann_val = TensorDataset(td_val1, td_val2, yd_val, dd_val)
        
        acc_d, f1_d, auc_d = train_dann(
            DataLoader(ds_dann_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True),
            DataLoader(ds_dann_val, batch_size=BATCH_SIZE),
            input_dim=Xd_train1.shape[1]
        )
        print(f"    DANN Fold {fold+1}: Acc={acc_d:.3f}, F1={f1_d:.3f}, AUC={auc_d:.3f}")
        results['Base DANN']['acc'].append(acc_d)
        results['Base DANN']['f1'].append(f1_d)
        results['Base DANN']['auc'].append(auc_d)
        
    print("\n" + "=" * 60)
    print("CROSS VALIDATION RESULTS")
    print("=" * 60)
    
    summary = {}
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        summary[model_name] = {}
        for m_name, vals in metrics.items():
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            summary[model_name][f"{m_name}_mean"] = round(mean_val, 4)
            summary[model_name][f"{m_name}_std"] = round(std_val, 4)
            print(f"  {m_name.upper()}: {mean_val:.4f} ± {std_val:.4f}")
            
    os.makedirs('results', exist_ok=True)
    with open('results/cross_validation_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nSaved to results/cross_validation_results.json")

if __name__ == "__main__":
    main()
