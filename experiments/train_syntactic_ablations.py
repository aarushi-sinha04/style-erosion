"""
Train 3 DANN ablation variants with different syntactic feature subsets.
Determines which syntactic features (POS, function words, readability) drive robustness.

Feature dims:
- POS-only: 1000 (POS trigrams)
- Function-words-only (lexical): 300 (function word counts)
- Readability-only: 8 (readability metrics)
- Full multi-view: 4308 (all features - already trained as baseline)

Expected time: ~3 hours per model × 3 = 9 hours total
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader_scie import PAN22Loader, BlogTextLoader, EnronLoader, IMDBLoader
from utils.feature_extraction import EnhancedFeatureExtractor
from models.dann import DANNSiameseV3, GradientReversalLayer

# =============================================================================
# Configuration
# =============================================================================
DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {DEVICE}")

BATCH_SIZE = 64
EPOCHS = 50
PATIENCE = 15
SAMPLES_PER_DOMAIN = 4000
WARMUP_EPOCHS = 5
GRL_PEAK = 0.3

DOMAIN_NAMES = ['PAN22', 'Blog', 'Enron', 'IMDB']

# Ablation configurations
ABLATION_CONFIGS = {
    'pos_only': {
        'input_dim': 1000,
        'feature_key': 'pos',
        'description': 'POS trigrams only'
    },
    'function_only': {
        'input_dim': 300,
        'feature_key': 'lex',
        'description': 'Function/lexical words only'
    },
    'readability_only': {
        'input_dim': 8,
        'feature_key': 'readability',
        'description': 'Readability metrics only'
    }
}


# =============================================================================
# Ablation DANN Model (simplified for smaller input dims)
# =============================================================================
class AblationDANN(nn.Module):
    """
    Simplified DANN for ablation studies.
    Adapts hidden dimensions based on input size.
    """
    def __init__(self, input_dim, num_domains=4):
        super().__init__()
        
        # Scale hidden dim based on input
        if input_dim <= 16:
            hidden_dim = 32
            auth_hidden = 16
        elif input_dim <= 512:
            hidden_dim = 256
            auth_hidden = 128
        else:
            hidden_dim = 512
            auth_hidden = 256
        
        self.hidden_dim = hidden_dim
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Authorship classifier with rich interaction [u, v, |u-v|, u*v]
        self.authorship_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, auth_hidden),
            nn.BatchNorm1d(auth_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(auth_hidden, 1),
            nn.Sigmoid()
        )
        
        # Domain classifier with gradient reversal
        self.grl = GradientReversalLayer(lambda_=1.0)
        self.domain_head = nn.Sequential(
            nn.Linear(hidden_dim, auth_hidden),
            nn.BatchNorm1d(auth_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(auth_hidden, num_domains)
        )
    
    def forward(self, text_a, text_b, alpha=1.0):
        self.grl.lambda_ = alpha
        
        features_a = self.feature_extractor(text_a)
        features_b = self.feature_extractor(text_b)
        
        # Rich interaction
        diff = torch.abs(features_a - features_b)
        prod = features_a * features_b
        combined = torch.cat([features_a, features_b, diff, prod], dim=1)
        
        authorship_pred = self.authorship_classifier(combined)
        
        domain_pred_a = self.domain_head(self.grl(features_a))
        domain_pred_b = self.domain_head(self.grl(features_b))
        
        return authorship_pred, domain_pred_a, domain_pred_b
    
    def get_embeddings(self, x):
        return self.feature_extractor(x)


# =============================================================================
# Data Loading
# =============================================================================
def extract_ablation_features(loader, extractor, domain_label, samples, feature_key):
    """Extract only the specified feature view for this ablation."""
    print(f"\nLoading data for Domain {DOMAIN_NAMES[domain_label]}...")
    loader.load(limit=samples * 3)
    t1_list, t2_list, labels = loader.create_pairs(num_pairs=samples)
    
    if not t1_list:
        return None
    
    print(f"Extracting {feature_key} features for {len(t1_list)} pairs...")
    
    f1_dict = extractor.transform(t1_list)
    f2_dict = extractor.transform(t2_list)
    
    # Only take the specified feature view
    X1 = f1_dict[feature_key]
    X2 = f2_dict[feature_key]
    
    if hasattr(X1, 'toarray'):
        X1 = X1.toarray()
    if hasattr(X2, 'toarray'):
        X2 = X2.toarray()
    
    X1_t = torch.tensor(np.array(X1), dtype=torch.float32)
    X2_t = torch.tensor(np.array(X2), dtype=torch.float32)
    Y_t = torch.tensor(labels, dtype=torch.float32)
    D_t = torch.full((len(labels),), domain_label, dtype=torch.long)
    
    print(f"  {DOMAIN_NAMES[domain_label]}: {len(X1_t)} pairs, dim={X1_t.shape[1]}")
    return TensorDataset(X1_t, X2_t, Y_t, D_t)


# =============================================================================
# Training
# =============================================================================
def compute_grl_lambda(epoch, total_epochs, warmup, peak=0.3):
    if epoch < warmup:
        return 0.0
    progress = (epoch - warmup) / (total_epochs - warmup)
    lambda_val = 2.0 / (1.0 + np.exp(-10 * progress)) - 1.0
    return lambda_val * peak


def train_ablation(ablation_type, extractor):
    """Train one DANN ablation variant."""
    config = ABLATION_CONFIGS[ablation_type]
    input_dim = config['input_dim']
    feature_key = config['feature_key']
    
    print(f"\n{'='*60}")
    print(f"Training {ablation_type} DANN ({config['description']})")
    print(f"Input dim: {input_dim}")
    print(f"{'='*60}")
    
    # Load data for all domains
    loaders = [
        PAN22Loader("pan22-authorship-verification-training.jsonl",
                     "pan22-authorship-verification-training-truth.jsonl"),
        BlogTextLoader("blogtext.csv"),
        EnronLoader("emails.csv"),
        IMDBLoader("IMDB Dataset.csv")
    ]
    
    train_datasets = []
    val_datasets = []
    
    for i, loader in enumerate(loaders):
        ds = extract_ablation_features(
            loader, extractor, domain_label=i, 
            samples=SAMPLES_PER_DOMAIN, feature_key=feature_key
        )
        if ds:
            val_size = int(len(ds) * 0.2)
            train_size = len(ds) - val_size
            d_train, d_val = random_split(ds, [train_size, val_size])
            train_datasets.append(d_train)
            val_datasets.append(d_val)
    
    train_loaders = [DataLoader(d, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
                     for d in train_datasets]
    val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = AblationDANN(input_dim=input_dim, num_domains=4).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion_domain = nn.CrossEntropyLoss()
    
    best_avg_acc = 0.0
    patience_counter = 0
    
    save_path = f"models/dann_{ablation_type}.pth"
    os.makedirs("models", exist_ok=True)
    
    for epoch in range(EPOCHS):
        model.train()
        grl_lambda = compute_grl_lambda(epoch, EPOCHS, WARMUP_EPOCHS, GRL_PEAK)
        is_warmup = epoch < WARMUP_EPOCHS
        
        total_loss_auth = 0
        total_batches = 0
        correct_auth = 0
        total_auth = 0
        
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
            
            X1 = torch.cat(X1_list).to(DEVICE)
            X2 = torch.cat(X2_list).to(DEVICE)
            Y = torch.cat(Y_list).to(DEVICE)
            D = torch.cat(D_list).to(DEVICE)
            
            optimizer.zero_grad()
            
            p_auth, p_dom_a, p_dom_b = model(X1, X2, alpha=grl_lambda)
            p_auth = p_auth.squeeze()
            
            # Authorship loss
            mask = (Y != -1).float()
            y_safe = torch.max(Y, torch.tensor(0.0).to(DEVICE))
            l_auth_raw = nn.BCELoss(reduction='none')(p_auth, y_safe)
            l_auth = (l_auth_raw * mask).sum() / (mask.sum() + 1e-8)
            
            if mask.sum() > 0:
                preds = (p_auth > 0.5).float()
                correct_auth += ((preds == y_safe) * mask).sum().item()
                total_auth += mask.sum().item()
            
            # Domain loss
            if not is_warmup:
                l_dom = (criterion_domain(p_dom_a, D) + criterion_domain(p_dom_b, D)) / 2
                loss = l_auth + grl_lambda * l_dom
            else:
                loss = l_auth
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss_auth += l_auth.item()
        
        scheduler.step()
        
        # Validate
        model.eval()
        domain_correct = {i: 0 for i in range(4)}
        domain_total = {i: 0 for i in range(4)}
        
        with torch.no_grad():
            for x1, x2, y, d in val_loader:
                x1, x2, y, d = x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE), d.to(DEVICE)
                p, _, _ = model(x1, x2, alpha=0.0)
                p = p.squeeze()
                preds = (p > 0.5).float()
                
                for dom_id in range(4):
                    if dom_id == 3:  # IMDB - no labels
                        continue
                    auth_mask = (y != -1) & (d == dom_id)
                    if auth_mask.sum() > 0:
                        domain_correct[dom_id] += ((preds[auth_mask] == y[auth_mask]).sum().item())
                        domain_total[dom_id] += auth_mask.sum().item()
        
        valid_accs = []
        for dom_id in range(3):
            if domain_total[dom_id] > 0:
                valid_accs.append(domain_correct[dom_id] / domain_total[dom_id])
        
        avg_acc = np.mean(valid_accs) if valid_accs else 0.0
        train_acc = correct_auth / max(1, total_auth)
        
        phase = "WARMUP" if is_warmup else "ADAPT"
        print(f"Epoch {epoch+1}/{EPOCHS} [{phase}] | "
              f"TrainAcc={train_acc*100:.1f}% | "
              f"ValAvg={avg_acc*100:.1f}% | "
              f"λ={grl_lambda:.2f}")
        
        # Early stopping after warmup
        if epoch >= WARMUP_EPOCHS:
            if avg_acc > best_avg_acc:
                best_avg_acc = avg_acc
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
                print(f"  ★ New best: {best_avg_acc*100:.1f}% → {save_path}")
            else:
                patience_counter += 1
            
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
        else:
            if avg_acc > best_avg_acc:
                best_avg_acc = avg_acc
                torch.save(model.state_dict(), save_path)
    
    print(f"\n✅ {ablation_type} DANN: Best Val Acc = {best_avg_acc*100:.1f}%")
    print(f"   Model saved to {save_path}")
    return best_avg_acc


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Starting ablation training at {start_time}")
    
    # Load the existing fitted extractor
    extractor_path = "results/final_dann/extractor.pkl"
    print(f"\nLoading extractor from {extractor_path}...")
    with open(extractor_path, 'rb') as f:
        extractor = pickle.load(f)
    
    # Train all ablations
    results = {}
    for ablation_type in ['pos_only', 'function_only', 'readability_only']:
        best_acc = train_ablation(ablation_type, extractor)
        results[ablation_type] = best_acc
    
    # Summary
    elapsed = datetime.now() - start_time
    print(f"\n{'='*60}")
    print(f"ALL ABLATIONS COMPLETE")
    print(f"Total time: {elapsed}")
    print(f"{'='*60}")
    for name, acc in results.items():
        print(f"  {name:20s}: {acc*100:.1f}%")
    
    # Save summary
    os.makedirs("results", exist_ok=True)
    with open("results/ablation_training_summary.json", "w") as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=2)
    
    print(f"\n✅ Saved training summary to results/ablation_training_summary.json")
