"""
Robust Siamese Training - Adversarial Fine-Tuning
==================================================
Fine-tunes the Cross-Domain Siamese with adversarial robustness:
1. Clean pairs from PAN22 + Blog + Enron (maintain accuracy)
2. Adversarial pairs from precomputed attacks (improve robustness)
3. Consistency loss: pred(A,P) ≈ pred(A,Adv_P)
"""
import sys
import os
import json
import random
import re
import numpy as np
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader_scie import PAN22Loader, BlogTextLoader, EnronLoader
from experiments.train_siamese_crossdomain import SiameseNetwork, PairDataset, preprocess, evaluate

# Config
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using Device: {DEVICE}")

CD_MODEL_PATH = "results/siamese_crossdomain/best_model.pth"
CD_VEC_PATH = "results/siamese_crossdomain/vectorizer.pkl"
CD_SCALER_PATH = "results/siamese_crossdomain/scaler.pkl"
ADV_DATA = "data/pan22_adversarial.jsonl"
OUTPUT_DIR = "results/robust_siamese"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparams
EPOCHS = 20
BATCH_SIZE = 32
LR = 2e-5      # Very low for fine-tuning
PATIENCE = 6
LAMBDA_ADV = 0.3
LAMBDA_CONS = 0.3


def train_robust_siamese():
    print("=" * 60)
    print("ROBUST SIAMESE TRAINING (Adversarial Fine-Tuning)")
    print("=" * 60)

    # 1. Load vectorizer and scaler
    vec = pickle.load(open(CD_VEC_PATH, 'rb'))
    scaler = pickle.load(open(CD_SCALER_PATH, 'rb'))
    input_dim = len(vec.get_feature_names_out())

    # 2. Load adversarial data
    print(f"Loading adversarial data from {ADV_DATA}...")
    adv_samples = []
    with open(ADV_DATA, 'r') as f:
        for line in f:
            adv_samples.append(json.loads(line))
    print(f"  {len(adv_samples)} adversarial triplets")

    # Pre-vectorize adversarial data
    adv_anchors = [preprocess(s['anchor']) for s in adv_samples]
    adv_pos = [preprocess(s['positive']) for s in adv_samples]
    adv_atk = [preprocess(s['positive_attacked']) for s in adv_samples]

    X_anc = torch.tensor(scaler.transform(vec.transform(adv_anchors).toarray()), dtype=torch.float32).to(DEVICE)
    X_pos = torch.tensor(scaler.transform(vec.transform(adv_pos).toarray()), dtype=torch.float32).to(DEVICE)
    X_atk = torch.tensor(scaler.transform(vec.transform(adv_atk).toarray()), dtype=torch.float32).to(DEVICE)

    # 3. Load clean data for mixed training
    print("\nLoading clean data...")
    clean_t1, clean_t2, clean_y = [], [], []

    for name, loader_cls, args in [
        ('PAN22', PAN22Loader, ["pan22-authorship-verification-training.jsonl",
                                 "pan22-authorship-verification-training-truth.jsonl"]),
        ('Blog', BlogTextLoader, ["blogtext.csv"]),
        ('Enron', EnronLoader, ["emails.csv"])]:
        l = loader_cls(*args)
        l.load(limit=3000)
        t1, t2, y = l.create_pairs(num_pairs=500)
        if t1:
            clean_t1.extend(t1)
            clean_t2.extend(t2)
            clean_y.extend(y)
            print(f"  {name}: {len(t1)} pairs")

    # Pre-vectorize clean data
    ct1_p = [preprocess(t) for t in clean_t1]
    ct2_p = [preprocess(t) for t in clean_t2]
    X_c1 = torch.tensor(scaler.transform(vec.transform(ct1_p).toarray()), dtype=torch.float32).to(DEVICE)
    X_c2 = torch.tensor(scaler.transform(vec.transform(ct2_p).toarray()), dtype=torch.float32).to(DEVICE)
    Y_c = torch.tensor(clean_y, dtype=torch.float32).to(DEVICE)

    # 4. Validation data
    print("\nLoading validation data...")
    val_t1, val_t2, val_y = [], [], []
    for name, loader_cls, args in [
        ('PAN22', PAN22Loader, ["pan22-authorship-verification-training.jsonl",
                                 "pan22-authorship-verification-training-truth.jsonl"]),
        ('Blog', BlogTextLoader, ["blogtext.csv"]),
        ('Enron', EnronLoader, ["emails.csv"])]:
        l = loader_cls(*args)
        l.load(limit=3000)
        t1, t2, y = l.create_pairs(num_pairs=200)
        if t1:
            val_t1.extend(t1)
            val_t2.extend(t2)
            val_y.extend(y)

    vt1_p = [preprocess(t) for t in val_t1]
    vt2_p = [preprocess(t) for t in val_t2]
    X_v1 = torch.tensor(scaler.transform(vec.transform(vt1_p).toarray()), dtype=torch.float32)
    X_v2 = torch.tensor(scaler.transform(vec.transform(vt2_p).toarray()), dtype=torch.float32)
    Y_v = torch.tensor(val_y, dtype=torch.float32).unsqueeze(1)
    val_ds = PairDataset(X_v1.numpy(), X_v2.numpy(), np.array(val_y), ['mixed'] * len(val_y))
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # 5. Load model
    print(f"\nLoading CD Siamese from {CD_MODEL_PATH}...")
    model = SiameseNetwork(input_dim=input_dim).to(DEVICE)
    model.load_state_dict(torch.load(CD_MODEL_PATH, map_location=DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    # Baseline validation
    _, base_acc, base_roc, base_f1 = evaluate(model, val_loader, criterion)
    print(f"Baseline: Acc={base_acc:.4f} ROC={base_roc:.4f} F1={base_f1:.4f}")

    # 6. Training loop
    adv_idx = list(range(len(adv_samples)))
    clean_idx = list(range(len(clean_y)))
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        random.shuffle(adv_idx)
        random.shuffle(clean_idx)
        total_loss, total_cons, n_batches = 0, 0, 0

        for i in range(0, len(adv_idx), BATCH_SIZE):
            batch = adv_idx[i:i + BATCH_SIZE]

            xa = X_anc[batch]
            xp = X_pos[batch]
            xadv = X_atk[batch]

            # Adversarial: A vs P -> same author (1)
            logits_pos = model(xa, xp)
            loss_pos = criterion(logits_pos, torch.ones_like(logits_pos))

            # Adversarial: A vs Adv -> still same author (1)
            logits_adv = model(xa, xadv)
            loss_adv = criterion(logits_adv, torch.ones_like(logits_adv))

            # Consistency
            loss_cons = nn.MSELoss()(torch.sigmoid(logits_pos), torch.sigmoid(logits_adv))

            # Clean batch
            n_clean = max(1, len(batch))
            cb = random.sample(clean_idx, min(n_clean, len(clean_idx)))
            xc1 = X_c1[cb]
            xc2 = X_c2[cb]
            yc = Y_c[cb].unsqueeze(1)
            logits_clean = model(xc1, xc2)
            loss_clean = criterion(logits_clean, yc)

            loss = loss_clean + loss_pos + LAMBDA_ADV * loss_adv + LAMBDA_CONS * loss_cons

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_cons += loss_cons.item()
            n_batches += 1

        # Validate
        _, val_acc, val_roc, val_f1 = evaluate(model, val_loader, criterion)
        avg_loss = total_loss / max(1, n_batches)

        print(f"Epoch {epoch+1}/{EPOCHS}: Loss={avg_loss:.4f} Cons={total_cons/max(1,n_batches):.4f} | "
              f"Val Acc={val_acc:.4f} ROC={val_roc:.4f} F1={val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pth")
            print(f"  ★ New Best! ({best_val_acc:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Save artifacts (reuse same vectorizer/scaler)
    with open(f"{OUTPUT_DIR}/vectorizer.pkl", "wb") as f: pickle.dump(vec, f)
    with open(f"{OUTPUT_DIR}/scaler.pkl", "wb") as f: pickle.dump(scaler, f)

    print(f"\n{'='*40}")
    print(f"Baseline Acc: {base_acc:.4f} -> Robust Best: {best_val_acc:.4f}")
    print(f"Accuracy Change: {(best_val_acc - base_acc)*100:+.1f}%")
    print(f"Saved to: {OUTPUT_DIR}/best_model.pth")


if __name__ == "__main__":
    train_robust_siamese()
