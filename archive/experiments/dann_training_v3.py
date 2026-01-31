"""
DANN Training V3 - Enhanced Cross-Domain Authorship Verification
===============================================================
Key improvements:
1. Progressive GRL scheduling (0 -> 1.5)
2. MMD loss for distribution alignment
3. Center loss for intra-class compactness  
4. Weighted domain sampling
5. Cross-domain validation for early stopping
"""
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, ConcatDataset, WeightedRandomSampler
import numpy as np
import pickle
from tqdm import tqdm
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

# Training Hyperparameters
MAX_FEATURES = 4308  # char(3000) + pos(1000) + lex(300) + read(8)
BATCH_SIZE = 64
EPOCHS = 50  # Start with 50,extend if needed
PATIENCE = 15
SAMPLES_PER_DOMAIN = 5000

# Loss weights
LAMBDA_MMD = 0.5
LAMBDA_CENTER = 0.1
GRL_PEAK = 1.5  # Peak GRL lambda

# Domain names for logging
DOMAIN_NAMES = ['PAN22', 'Blog', 'Enron', 'IMDB']

# =============================================================================
# Data Loading & Feature Extraction
# =============================================================================
def extract_features_dataset(loader, extractor, domain_label, samples):
    """Loads data, creates pairs, extracts multi-view features, returns TensorDataset."""
    print(f"\nLoading data for Domain {DOMAIN_NAMES[domain_label]}...")
    loader.load(limit=samples * 3)  # Load enough to sample pairs
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
def compute_grl_lambda(epoch, total_epochs, peak=1.5):
    """Progressive GRL scheduling: 0 -> peak (sigmoid curve)."""
    p = epoch / total_epochs
    lambda_val = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
    return lambda_val * peak

def train_epoch(model, train_loaders, optimizer, criterion_auth, criterion_domain, 
                centers, epoch, total_epochs, device):
    """Train for one epoch with MMD and center loss."""
    model.train()
    
    # Compute GRL lambda for this epoch
    grl_lambda = compute_grl_lambda(epoch, total_epochs, GRL_PEAK)
    
    # Statistics
    total_loss_auth = 0
    total_loss_domain = 0
    total_loss_mmd = 0
    total_loss_center = 0
    total_batches = 0
    correct_auth = 0
    total_auth = 0
    correct_domain = 0
    total_domain = 0
    
    # Interleaved training across domains
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
        
        # =====================================================================
        # 1. Authorship Loss (mask IMDB with -1 labels)
        # =====================================================================
        mask = (Y != -1).float()
        y_safe = torch.max(Y, torch.tensor(0.0).to(device))
        l_auth_raw = nn.BCELoss(reduction='none')(p_auth, y_safe)
        l_auth = (l_auth_raw * mask).sum() / (mask.sum() + 1e-8)
        
        # Track authorship accuracy
        if mask.sum() > 0:
            preds = (p_auth > 0.5).float()
            correct_auth += ((preds == y_safe) * mask).sum().item()
            total_auth += mask.sum().item()
        
        # =====================================================================
        # 2. Domain Loss (adversarial)
        # =====================================================================
        l_dom = (criterion_domain(p_dom_a, D) + criterion_domain(p_dom_b, D)) / 2
        
        # Track domain classifier accuracy
        pred_d = p_dom_a.argmax(dim=1)
        correct_domain += (pred_d == D).sum().item()
        total_domain += len(D)
        
        # =====================================================================
        # 3. MMD Loss (distribution alignment)
        # =====================================================================
        # Align each domain to the mean distribution
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
        
        # =====================================================================
        # 4. Center Loss (intra-class compactness)
        # =====================================================================
        features_a = model.get_embeddings(X1)
        auth_mask = (Y != -1)
        
        if auth_mask.sum() > 0:
            l_center = compute_center_loss(features_a[auth_mask], Y[auth_mask].long(), centers)
            # Update centers
            with torch.no_grad():
                centers = update_centers(centers, features_a[auth_mask].detach(), Y[auth_mask].long())
        else:
            l_center = torch.tensor(0.0, device=device)
        
        # =====================================================================
        # Total Loss
        # =====================================================================
        loss = l_auth + grl_lambda * l_dom + LAMBDA_MMD * l_mmd + LAMBDA_CENTER * l_center
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss_auth += l_auth.item()
        total_loss_domain += l_dom.item()
        total_loss_mmd += l_mmd.item()
        total_loss_center += l_center.item()
    
    # Compute averages
    metrics = {
        'loss_auth': total_loss_auth / total_batches,
        'loss_domain': total_loss_domain / total_batches,
        'loss_mmd': total_loss_mmd / total_batches,
        'loss_center': total_loss_center / total_batches,
        'acc_auth': correct_auth / max(1, total_auth),
        'acc_domain': correct_domain / max(1, total_domain),
        'grl_lambda': grl_lambda
    }
    
    return metrics, centers

def validate(model, val_loader, device):
    """Validate and return per-domain accuracies."""
    model.eval()
    
    domain_correct = {i: 0 for i in range(4)}
    domain_total = {i: 0 for i in range(4)}
    domain_total_with_labels = {i: 0 for i in range(4)}
    
    total_domain_correct = 0
    total_domain_total = 0
    
    with torch.no_grad():
        for x1, x2, y, d in val_loader:
            x1, x2, y, d = x1.to(device), x2.to(device), y.to(device), d.to(device)
            
            p_auth, p_dom_a, _ = model(x1, x2, alpha=0.0)
            p_auth = p_auth.squeeze()
            
            # Authorship accuracy per domain
            preds = (p_auth > 0.5).float()
            
            for dom_id in range(4):
                dom_mask = (d == dom_id)
                if dom_id == 3:  # IMDB - no labels
                    continue
                    
                auth_mask = (y != -1) & dom_mask
                if auth_mask.sum() > 0:
                    domain_correct[dom_id] += ((preds[auth_mask] == y[auth_mask]).sum().item())
                    domain_total[dom_id] += auth_mask.sum().item()
            
            # Domain classifier accuracy
            pred_d = p_dom_a.argmax(dim=1)
            total_domain_correct += (pred_d == d).sum().item()
            total_domain_total += len(d)
    
    # Compute per-domain accuracy
    domain_accs = {}
    for dom_id in range(4):
        if dom_id == 3:  # IMDB
            domain_accs[DOMAIN_NAMES[dom_id]] = None
        elif domain_total[dom_id] > 0:
            domain_accs[DOMAIN_NAMES[dom_id]] = domain_correct[dom_id] / domain_total[dom_id]
        else:
            domain_accs[DOMAIN_NAMES[dom_id]] = 0.0
    
    # Average cross-domain (PAN22, Blog, Enron)
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
def train_dann_v3():
    print("=" * 60)
    print("DANN V3 Training - Cross-Domain Authorship Verification")
    print("=" * 60)
    print(f"Config: epochs={EPOCHS}, batch={BATCH_SIZE}, grl_peak={GRL_PEAK}")
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
    
    # 2. Fit or Load Extractor
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
    
    # Create DataLoaders
    train_loaders = [DataLoader(d, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) 
                     for d in train_datasets]
    val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. Initialize Model
    print("\n" + "=" * 60)
    print("Initializing Model...")
    print("=" * 60)
    
    model = DANNSiameseV3(input_dim=MAX_FEATURES, num_domains=4).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion_domain = nn.CrossEntropyLoss()
    
    # Initialize class centers for center loss
    centers = torch.zeros(2, 512).to(DEVICE)  # 2 classes (same/different), 512 dim
    
    # 5. Training
    print("\n" + "=" * 60)
    print(f"Starting Training ({EPOCHS} epochs, patience {PATIENCE})...")
    print("=" * 60)
    
    best_avg_acc = 0.0
    patience_counter = 0
    training_log = []
    
    for epoch in range(EPOCHS):
        # Train
        train_metrics, centers = train_epoch(
            model, train_loaders, optimizer, None, criterion_domain,
            centers, epoch, EPOCHS, DEVICE
        )
        
        # Validate
        val_metrics = validate(model, val_loader, DEVICE)
        
        # Step scheduler
        scheduler.step()
        
        # Log
        domain_accs = val_metrics['domain_accs']
        log_entry = {
            'epoch': epoch + 1,
            **train_metrics,
            **val_metrics
        }
        training_log.append(log_entry)
        
        # Print progress
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"  Train: L_auth={train_metrics['loss_auth']:.4f} | L_dom={train_metrics['loss_domain']:.4f} | "
              f"L_mmd={train_metrics['loss_mmd']:.4f} | L_center={train_metrics['loss_center']:.4f}")
        print(f"  Train: Auth={train_metrics['acc_auth']*100:.1f}% | Dom={train_metrics['acc_domain']*100:.1f}% | "
              f"λ_grl={train_metrics['grl_lambda']:.2f}")
        print(f"  Val:   PAN22={domain_accs['PAN22']*100:.1f}% | Blog={domain_accs['Blog']*100:.1f}% | "
              f"Enron={domain_accs['Enron']*100:.1f}%")
        print(f"  Val:   Avg={val_metrics['avg_acc']*100:.1f}% | DomClf={val_metrics['domain_classifier_acc']*100:.1f}%")
        
        # Check domain confusion target
        if val_metrics['domain_classifier_acc'] < 0.35:
            print("  ✓ Good domain confusion (<35%)")
        
        # Early stopping based on cross-domain average
        if val_metrics['avg_acc'] > best_avg_acc:
            best_avg_acc = val_metrics['avg_acc']
            patience_counter = 0
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/dann_model_v3.pth")
            print(f"  ★ New best model saved! (Avg: {best_avg_acc*100:.2f}%)")
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    # 6. Save training log
    with open(f"{OUTPUT_DIR}/training_log_v3.json", "w") as f:
        json.dump(training_log, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    # 7. Final Report
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Average Cross-Domain Accuracy: {best_avg_acc * 100:.2f}%")
    print(f"Model saved to: {OUTPUT_DIR}/dann_model_v3.pth")
    print(f"Training log: {OUTPUT_DIR}/training_log_v3.json")
    
    return best_avg_acc

if __name__ == "__main__":
    train_dann_v3()
