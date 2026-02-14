"""
Robust DANN Training V2 - Balanced Clean + Adversarial
=======================================================
Key fixes from V1:
1. Mix clean data alongside adversarial (prevents catastrophic forgetting)
2. Lower adversarial loss weight (0.5 instead of 1.5)
3. Lower consistency weight (0.5 instead of 2.0)
4. Validate on all 3 labeled domains
5. More epochs with proper early stopping
"""
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
import numpy as np
import pickle
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.dann import DANNSiameseV3
from utils.data_loader_scie import PAN22Loader, BlogTextLoader, EnronLoader

# Config
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
ADV_DATA_FILE = "data/pan22_adversarial.jsonl"
OUTPUT_DIR = "results/robust_dann"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_MODEL_PATH = "results/final_dann/dann_model_v4.pth"
EXTRACTOR_PATH = "results/final_dann/extractor.pkl"

# Hyperparams
EPOCHS = 30
BATCH_SIZE = 32
LR = 5e-6  # Very low LR for fine-tuning
PATIENCE = 8
LAMBDA_ADV = 0.5    # Was 1.5 - too aggressive
LAMBDA_CONS = 0.5   # Was 2.0 - too aggressive
CLEAN_RATIO = 0.5   # 50% clean pairs per batch

def get_feats(extractor, texts):
    f_dict = extractor.transform(texts)
    return np.hstack([f_dict['char'], f_dict['pos'], f_dict['lex'], f_dict['readability']])

def train_robust():
    print("=" * 60)
    print("Robust DANN Training V2 (Balanced Clean + Adversarial)")
    print("=" * 60)

    # 1. Load Extractor
    print(f"Loading Feature Extractor from {EXTRACTOR_PATH}...")
    with open(EXTRACTOR_PATH, 'rb') as f:
        extractor = pickle.load(f)

    # 2. Load Adversarial Data
    print(f"Loading Adversarial Data from {ADV_DATA_FILE}...")
    adv_samples = []
    with open(ADV_DATA_FILE, 'r') as f:
        for line in f:
            adv_samples.append(json.loads(line))
    print(f"  Loaded {len(adv_samples)} adversarial triplets")

    # Pre-extract adversarial features
    print("Pre-extracting adversarial features...")
    anchors = [s['anchor'] for s in adv_samples]
    positives = [s['positive'] for s in adv_samples]
    advs = [s['positive_attacked'] for s in adv_samples]

    f_anchors = get_feats(extractor, anchors)
    f_positives = get_feats(extractor, positives)
    f_advs = get_feats(extractor, advs)

    X_anc = torch.tensor(f_anchors, dtype=torch.float32).to(DEVICE)
    X_pos = torch.tensor(f_positives, dtype=torch.float32).to(DEVICE)
    X_adv = torch.tensor(f_advs, dtype=torch.float32).to(DEVICE)
    print(f"  Adversarial Feature Shape: {X_anc.shape}")

    # 3. Load CLEAN validation + training data from multiple domains
    print("\nLoading Clean Data for mixed training...")
    clean_t1, clean_t2, clean_y = [], [], []

    domain_configs = [
        ('PAN22', PAN22Loader, ["pan22-authorship-verification-training.jsonl", "pan22-authorship-verification-training-truth.jsonl"]),
        ('Blog', BlogTextLoader, ["blogtext.csv"]),
        ('Enron', EnronLoader, ["emails.csv"])
    ]
    for name, loader_cls, args in domain_configs:
        l = loader_cls(*args)
        l.load(limit=3000)
        t1, t2, y = l.create_pairs(num_pairs=500)
        if t1:
            clean_t1.extend(t1)
            clean_t2.extend(t2)
            clean_y.extend(y)
            print(f"  {name}: {len(t1)} clean pairs")

    # Pre-extract clean features
    print("Pre-extracting clean features...")
    f_clean1 = get_feats(extractor, clean_t1)
    f_clean2 = get_feats(extractor, clean_t2)
    X_c1 = torch.tensor(f_clean1, dtype=torch.float32).to(DEVICE)
    X_c2 = torch.tensor(f_clean2, dtype=torch.float32).to(DEVICE)
    Y_c = torch.tensor(clean_y, dtype=torch.float32).to(DEVICE)

    # 4. Validation Data (separate from training clean data)
    print("\nLoading Validation Data...")
    val_loaders = {}
    for name, loader_cls, args in domain_configs:
        l = loader_cls(*args)
        l.load(limit=3000)
        t1, t2, y = l.create_pairs(num_pairs=300)
        if t1:
            f1 = get_feats(extractor, t1)
            f2 = get_feats(extractor, t2)
            x1_v = torch.tensor(f1, dtype=torch.float32).to(DEVICE)
            x2_v = torch.tensor(f2, dtype=torch.float32).to(DEVICE)
            y_v = torch.tensor(y, dtype=torch.float32).to(DEVICE)
            val_loaders[name] = (x1_v, x2_v, y_v)

    # 5. Load Model
    print(f"\nLoading Base DANN from {BASE_MODEL_PATH}...")
    model = DANNSiameseV3(input_dim=X_anc.shape[1], num_domains=4).to(DEVICE)
    model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=DEVICE), strict=False)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 6. Training Loop
    adv_indices = list(range(len(adv_samples)))
    clean_indices = list(range(len(clean_y)))
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        random.shuffle(adv_indices)
        random.shuffle(clean_indices)
        total_loss = 0
        total_cons = 0
        num_batches = 0

        # Process adversarial batches mixed with clean
        for i in range(0, len(adv_indices), BATCH_SIZE):
            batch_adv = adv_indices[i:i + BATCH_SIZE]

            # === ADVERSARIAL COMPONENT ===
            xa = X_anc[batch_adv]
            xp = X_pos[batch_adv]
            xadv = X_adv[batch_adv]

            # Anchor vs Positive -> Same Author (1)
            pred_pos, _, _ = model(xa, xp, alpha=0.0)
            loss_pos = nn.BCELoss()(pred_pos, torch.ones_like(pred_pos))

            # Anchor vs Adversarial -> Still Same Author (1) - ROBUSTNESS
            pred_adv, _, _ = model(xa, xadv, alpha=0.0)
            loss_adv = nn.BCELoss()(pred_adv, torch.ones_like(pred_adv))

            # Consistency: pred(A,P) ≈ pred(A,Adv)
            loss_cons = nn.MSELoss()(pred_pos, pred_adv)

            # === CLEAN COMPONENT (prevents catastrophic forgetting) ===
            # Sample clean batch
            n_clean = max(1, int(len(batch_adv) * CLEAN_RATIO))
            clean_batch = random.sample(clean_indices, min(n_clean, len(clean_indices)))
            xc1 = X_c1[clean_batch]
            xc2 = X_c2[clean_batch]
            yc = Y_c[clean_batch].unsqueeze(1)

            pred_clean, _, _ = model(xc1, xc2, alpha=0.0)
            loss_clean = nn.BCELoss()(pred_clean, yc)

            # === TOTAL LOSS (balanced) ===
            loss = loss_clean + loss_pos + LAMBDA_ADV * loss_adv + LAMBDA_CONS * loss_cons

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_cons += loss_cons.item()
            num_batches += 1

        # Validation
        model.eval()
        val_accs = {}
        with torch.no_grad():
            for name, (vx1, vx2, vy) in val_loaders.items():
                vp, _, _ = model(vx1, vx2, alpha=0.0)
                vpreds = (vp > 0.5).float().squeeze()
                vacc = (vpreds == vy).float().mean().item()
                val_accs[name] = vacc

        avg_val_acc = np.mean(list(val_accs.values()))
        avg_loss = total_loss / max(1, num_batches)
        avg_cons = total_cons / max(1, num_batches)

        val_str = " | ".join([f"{k}={v*100:.1f}%" for k, v in val_accs.items()])
        print(f"Epoch {epoch+1}/{EPOCHS}: Loss {avg_loss:.4f} | Cons {avg_cons:.4f} | {val_str} | Avg: {avg_val_acc*100:.1f}%")

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            patience_counter = 0
            save_path = f"{OUTPUT_DIR}/robust_dann_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  ★ New Best! ({best_val_acc*100:.1f}%)")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"\nBest Robust Model: {best_val_acc*100:.1f}% avg validation accuracy")
    print(f"Saved to: {OUTPUT_DIR}/robust_dann_model.pth")

if __name__ == "__main__":
    train_robust()
