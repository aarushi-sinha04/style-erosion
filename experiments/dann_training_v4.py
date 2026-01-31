"""
DANN Training V4 - Balanced Cross-Domain Authorship Verification
=================================================================
Key changes from V3:
1. Reduced GRL peak (1.5 -> 0.5) to balance domain confusion vs authorship learning
2. Curriculum learning: train authorship first, then add domain adaptation
3. Warmup phase: first 10 epochs focus on authorship only
4. Reduced MMD weight
5. Added gradient monitoring
"""
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, ConcatDataset
import numpy as np
import pickle
import json
from datetime import datetime

# Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader_scie import PAN22Loader, BlogTextLoader, EnronLoader, IMDBLoader
from models.dann_siamese_v3 import DANNSiameseV3, compute_mmd, compute_center_loss, update_centers
from utils.feature_extraction import EnhancedFeatureExtractor

# =============================================================================
# Configuration
# =============================================================================
OUTPUT_DIR = "results_dann"
if not os.path.exists(OUTPUT_DIR): 
    os.makedirs(OUTPUT_DIR)

DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {DEVICE}")

# Training Hyperparameters - REBALANCED
MAX_FEATURES = 4308
BATCH_SIZE = 64
EPOCHS = 100  # More epochs for curriculum learning
PATIENCE = 25  # More patience
SAMPLES_PER_DOMAIN = 5000

# Loss weights - REDUCED for better balance
LAMBDA_MMD = 0.1  # Reduced from 0.5
LAMBDA_CENTER = 0.05  # Reduced from 0.1
GRL_PEAK = 0.5  # Reduced from 1.5
WARMUP_EPOCHS = 15  # Pure authorship training first

DOMAIN_NAMES = ['PAN22', 'Blog', 'Enron', 'IMDB']

# =============================================================================
# Data Loading
# =============================================================================
def extract_features_dataset(loader, extractor, domain_label, samples):
    """Loads data, creates pairs, extracts multi-view features."""
    print(f"\nLoading data for Domain {DOMAIN_NAMES[domain_label]}...")
    loader.load(limit=samples * 3)
    t1_list, t2_list, labels = loader.create_pairs(num_pairs=samples)
    
    if not t1_list:
        return None
    
    print(f"Extracting features for {len(t1_list)} pairs...")
    
    def flatten_feats(feats_dict):
        return np.hstack([
            feats_dict['char'], 
            feats_dict['pos'], 
            feats_dict['lex'], 
            feats_dict['readability']
        ])
        
    f1_dict = extractor.transform(t1_list)
    f2_dict = extractor.transform(t2_list)
    
    X1 = flatten_feats(f1_dict)
    X2 = flatten_feats(f2_dict)
    
    X1_t = torch.tensor(X1, dtype=torch.float32)
    X2_t = torch.tensor(X2, dtype=torch.float32)
    Y_t = torch.tensor(labels, dtype=torch.float32)
    D_t = torch.full((len(labels),), domain_label, dtype=torch.long)
    
    print(f"  {DOMAIN_NAMES[domain_label]}: {len(X1_t)} pairs")
    return TensorDataset(X1_t, X2_t, Y_t, D_t)

# =============================================================================
# Training Functions
# =============================================================================
def compute_grl_lambda(epoch, total_epochs, warmup, peak=0.5):
    """Progressive GRL scheduling with warmup phase."""
    if epoch < warmup:
        return 0.0  # No domain adaptation during warmup
    
    # After warmup: gradual increase
    progress = (epoch - warmup) / (total_epochs - warmup)
    lambda_val = 2.0 / (1.0 + np.exp(-10 * progress)) - 1.0
    return lambda_val * peak

def train_epoch(model, train_loaders, optimizer, criterion_domain, 
                centers, epoch, total_epochs, device):
    """Train for one epoch with curriculum learning."""
    model.train()
    
    # Compute GRL lambda (0 during warmup)
    grl_lambda = compute_grl_lambda(epoch, total_epochs, WARMUP_EPOCHS, GRL_PEAK)
    is_warmup = epoch < WARMUP_EPOCHS
    
    # Statistics
    total_loss_auth = 0
    total_loss_domain = 0
    total_loss_mmd = 0
    total_batches = 0
    correct_auth = 0
    total_auth = 0
    correct_domain = 0
    total_domain = 0
    
    # Interleaved training
    min_len = min([len(l) for l in train_loaders])
    iterators = [iter(l) for l in train_loaders]
    
    for batch_idx in range(min_len):
        total_batches += 1
        X1_list, X2_list, Y_list, D_list = [], [], [], []
        
        for it in iterators:
            try:
                x1, x2, y, d = next(it)
                X1_list.append(x1)
                X2_list.append(x2)
                Y_list.append(y)
                D_list.append(d)
            except StopIteration:
                pass
        
        X1 = torch.cat(X1_list).to(device)
        X2 = torch.cat(X2_list).to(device)
        Y = torch.cat(Y_list).to(device)
        D = torch.cat(D_list).to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        p_auth, p_dom_a, p_dom_b = model(X1, X2, alpha=grl_lambda)
        p_auth = p_auth.squeeze()
        
        # 1. Authorship Loss
        mask = (Y != -1).float()
        y_safe = torch.max(Y, torch.tensor(0.0).to(device))
        l_auth_raw = nn.BCELoss(reduction='none')(p_auth, y_safe)
        l_auth = (l_auth_raw * mask).sum() / (mask.sum() + 1e-8)
        
        if mask.sum() > 0:
            preds = (p_auth > 0.5).float()
            correct_auth += ((preds == y_safe) * mask).sum().item()
            total_auth += mask.sum().item()
        
        # 2. Domain Loss (only after warmup)
        if not is_warmup:
            l_dom = (criterion_domain(p_dom_a, D) + criterion_domain(p_dom_b, D)) / 2
        else:
            l_dom = torch.tensor(0.0, device=device)
        
        pred_d = p_dom_a.argmax(dim=1)
        correct_domain += (pred_d == D).sum().item()
        total_domain += len(D)
        
        # 3. MMD Loss (only after warmup, reduced weight)
        if not is_warmup:
            projections_a = model.get_projections(X1)
            l_mmd = torch.tensor(0.0, device=device)
            num_domains = len(DOMAIN_NAMES)
            
            for i in range(num_domains):
                for j in range(i + 1, num_domains):
                    mask_i = (D == i)
                    mask_j = (D == j)
                    if mask_i.sum() > 0 and mask_j.sum() > 0:
                        l_mmd += compute_mmd(projections_a[mask_i], projections_a[mask_j])
            
            l_mmd = l_mmd / max(1, (num_domains * (num_domains - 1) / 2))
        else:
            l_mmd = torch.tensor(0.0, device=device)
        
        # 4. Center Loss
        features_a = model.get_embeddings(X1)
        auth_mask = (Y != -1)
        
        if auth_mask.sum() > 0:
            l_center = compute_center_loss(features_a[auth_mask], Y[auth_mask].long(), centers)
            with torch.no_grad():
                centers = update_centers(centers, features_a[auth_mask].detach(), Y[auth_mask].long())
        else:
            l_center = torch.tensor(0.0, device=device)
        
        # Total Loss
        if is_warmup:
            # During warmup: only authorship + light center loss
            loss = l_auth + 0.01 * l_center
        else:
            # After warmup: full curriculum
            loss = l_auth + grl_lambda * l_dom + LAMBDA_MMD * l_mmd + LAMBDA_CENTER * l_center
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss_auth += l_auth.item()
        total_loss_domain += l_dom.item() if not is_warmup else 0
        total_loss_mmd += l_mmd.item() if not is_warmup else 0
    
    metrics = {
        'loss_auth': total_loss_auth / total_batches,
        'loss_domain': total_loss_domain / total_batches,
        'loss_mmd': total_loss_mmd / total_batches,
        'acc_auth': correct_auth / max(1, total_auth),
        'acc_domain': correct_domain / max(1, total_domain),
        'grl_lambda': grl_lambda,
        'is_warmup': is_warmup
    }
    
    return metrics, centers

def validate(model, val_loader, device):
    """Validate and return per-domain accuracies."""
    model.eval()
    
    domain_correct = {i: 0 for i in range(4)}
    domain_total = {i: 0 for i in range(4)}
    
    total_domain_correct = 0
    total_domain_total = 0
    
    with torch.no_grad():
        for x1, x2, y, d in val_loader:
            x1, x2, y, d = x1.to(device), x2.to(device), y.to(device), d.to(device)
            
            p_auth, p_dom_a, _ = model(x1, x2, alpha=0.0)
            p_auth = p_auth.squeeze()
            
            preds = (p_auth > 0.5).float()
            
            for dom_id in range(4):
                if dom_id == 3:  # IMDB - no labels
                    continue
                    
                auth_mask = (y != -1) & (d == dom_id)
                if auth_mask.sum() > 0:
                    domain_correct[dom_id] += ((preds[auth_mask] == y[auth_mask]).sum().item())
                    domain_total[dom_id] += auth_mask.sum().item()
            
            pred_d = p_dom_a.argmax(dim=1)
            total_domain_correct += (pred_d == d).sum().item()
            total_domain_total += len(d)
    
    domain_accs = {}
    for dom_id in range(4):
        if dom_id == 3:
            domain_accs[DOMAIN_NAMES[dom_id]] = None
        elif domain_total[dom_id] > 0:
            domain_accs[DOMAIN_NAMES[dom_id]] = domain_correct[dom_id] / domain_total[dom_id]
        else:
            domain_accs[DOMAIN_NAMES[dom_id]] = 0.0
    
    valid_accs = [v for v in domain_accs.values() if v is not None and v > 0]
    avg_acc = np.mean(valid_accs) if valid_accs else 0.0
    
    domain_classifier_acc = total_domain_correct / max(1, total_domain_total)
    
    return {
        'domain_accs': domain_accs,
        'avg_acc': avg_acc,
        'domain_classifier_acc': domain_classifier_acc
    }

# =============================================================================
# Main Training Loop
# =============================================================================
def train_dann_v4():
    print("=" * 60)
    print("DANN V4 Training - Curriculum Learning Approach")
    print("=" * 60)
    print(f"Config: epochs={EPOCHS}, warmup={WARMUP_EPOCHS}, grl_peak={GRL_PEAK}")
    print(f"Loss weights: mmd={LAMBDA_MMD}, center={LAMBDA_CENTER}")
    print("=" * 60)
    
    # 1. Initialize Loaders
    loaders = [
        PAN22Loader("pan22-authorship-verification-training.jsonl", 
                    "pan22-authorship-verification-training-truth.jsonl"),
        BlogTextLoader("blogtext.csv"),
        EnronLoader("emails.csv"),
        IMDBLoader("IMDB Dataset.csv")
    ]
    
    # 2. Load Extractor
    if os.path.exists(f"{OUTPUT_DIR}/extractor.pkl"):
        print(f"\nLoading extractor from {OUTPUT_DIR}/extractor.pkl...")
        with open(f"{OUTPUT_DIR}/extractor.pkl", "rb") as f:
            extractor = pickle.load(f)
    else:
        print("\nFitting EnhancedFeatureExtractor...")
        fit_texts = []
        for l in loaders:
            df = l.load(limit=2000)
            if not df.empty:
                fit_texts.extend(df['text'].tolist())
                
        extractor = EnhancedFeatureExtractor(
            max_features_char=3000, 
            max_features_pos=1000, 
            max_features_lex=300
        )
        extractor.fit(fit_texts)
        
        with open(f"{OUTPUT_DIR}/extractor.pkl", "wb") as f:
            pickle.dump(extractor, f)
    
    # 3. Prepare Datasets
    print("\n" + "=" * 60)
    print("Preparing Datasets...")
    print("=" * 60)
    
    train_datasets = []
    val_datasets = []
    val_split = 0.2
    
    for i, l in enumerate(loaders):
        ds = extract_features_dataset(l, extractor, domain_label=i, samples=SAMPLES_PER_DOMAIN)
        if ds:
            val_size = int(len(ds) * val_split)
            train_size = len(ds) - val_size
            d_train, d_val = random_split(ds, [train_size, val_size])
            train_datasets.append(d_train)
            val_datasets.append(d_val)
    
    train_loaders = [DataLoader(d, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) 
                     for d in train_datasets]
    val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. Initialize Model
    print("\n" + "=" * 60)
    print("Initializing Model...")
    print("=" * 60)
    
    model = DANNSiameseV3(input_dim=MAX_FEATURES, num_domains=4).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)  # Lower LR
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    criterion_domain = nn.CrossEntropyLoss()
    
    centers = torch.zeros(2, 512).to(DEVICE)
    
    # 5. Training
    print("\n" + "=" * 60)
    print(f"Starting Training ({EPOCHS} epochs, {WARMUP_EPOCHS} warmup, patience {PATIENCE})...")
    print("=" * 60)
    
    best_avg_acc = 0.0
    patience_counter = 0
    training_log = []
    
    for epoch in range(EPOCHS):
        train_metrics, centers = train_epoch(
            model, train_loaders, optimizer, criterion_domain,
            centers, epoch, EPOCHS, DEVICE
        )
        
        val_metrics = validate(model, val_loader, DEVICE)
        scheduler.step()
        
        domain_accs = val_metrics['domain_accs']
        log_entry = {
            'epoch': epoch + 1,
            **train_metrics,
            **val_metrics
        }
        training_log.append(log_entry)
        
        # Print progress
        phase = "WARMUP" if train_metrics['is_warmup'] else "ADAPT"
        print(f"\nEpoch {epoch + 1}/{EPOCHS} [{phase}]")
        print(f"  Train: L_auth={train_metrics['loss_auth']:.4f} | Auth={train_metrics['acc_auth']*100:.1f}%")
        
        if not train_metrics['is_warmup']:
            print(f"  Train: L_dom={train_metrics['loss_domain']:.4f} | L_mmd={train_metrics['loss_mmd']:.4f} | λ_grl={train_metrics['grl_lambda']:.2f}")
        
        print(f"  Val:   PAN22={domain_accs['PAN22']*100:.1f}% | Blog={domain_accs['Blog']*100:.1f}% | Enron={domain_accs['Enron']*100:.1f}%")
        print(f"  Val:   Avg={val_metrics['avg_acc']*100:.1f}% | DomClf={val_metrics['domain_classifier_acc']*100:.1f}%")
        
        if val_metrics['domain_classifier_acc'] < 0.40:
            print("  ✓ Domain confusion <40%")
        
        # Early stopping (only after warmup)
        if epoch >= WARMUP_EPOCHS:
            if val_metrics['avg_acc'] > best_avg_acc:
                best_avg_acc = val_metrics['avg_acc']
                patience_counter = 0
                torch.save(model.state_dict(), f"{OUTPUT_DIR}/dann_model_v4.pth")
                print(f"  ★ New best model saved! (Avg: {best_avg_acc*100:.2f}%)")
            else:
                patience_counter += 1
            
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        else:
            # During warmup, just save if better
            if val_metrics['avg_acc'] > best_avg_acc:
                best_avg_acc = val_metrics['avg_acc']
                torch.save(model.state_dict(), f"{OUTPUT_DIR}/dann_model_v4.pth")
                print(f"  ★ Warmup best: {best_avg_acc*100:.2f}%")
    
    # 6. Save training log
    with open(f"{OUTPUT_DIR}/training_log_v4.json", "w") as f:
        json.dump(training_log, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    # 7. Final Report
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Average Cross-Domain Accuracy: {best_avg_acc * 100:.2f}%")
    print(f"Model saved to: {OUTPUT_DIR}/dann_model_v4.pth")
    
    return best_avg_acc

if __name__ == "__main__":
    train_dann_v4()
