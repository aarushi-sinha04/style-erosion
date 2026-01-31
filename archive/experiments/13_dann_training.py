import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pickle
from tqdm import tqdm

# Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader_scie import PAN22Loader, BlogTextLoader, EnronLoader, IMDBLoader
from models.dann_siamese import DANNSiamese
from utils.feature_extraction import EnhancedFeatureExtractor

# Config
OUTPUT_DIR = "results_dann"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

MAX_FEATURES = 4308 # 3000 (Char) + 1000 (POS) + 300 (Lex) + 8 (Read)
BATCH_SIZE = 64
EPOCHS = 200
PATIENCE = 20
SAMPLES_PER_DOMAIN = 5000 

def extract_features_dataset(loader, extractor, domain_label, samples):
    """
    Loads data, creates pairs, extracts multi-view features, returns TensorDataset.
    """
    print(f"Loading data for Domain {domain_label}...")
    loader.load(limit=samples*3) # Load enough to sample pairs
    t1_list, t2_list, labels = loader.create_pairs(num_pairs=samples)
    
    if not t1_list: return None
    
    print(f"Extracting features for {len(t1_list)} pairs (Domain {domain_label})...")
    
    # Extract as dicts
    # We process in batches to avoid RAM explosion if possible, but for 5000 it's fine.
    # Actually EnhancedFeatureExtractor chunks internally for POS tagging.
    
    # We need to flatten the dicts
    def flatten_feats(feats_dict):
        # Concatenate: char, pos, lex, readability
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
    
    return TensorDataset(X1_t, X2_t, Y_t, D_t)

def train_dann():
    # 1. Initialize Loaders
    loaders = [
        PAN22Loader("pan22-authorship-verification-training.jsonl", "pan22-authorship-verification-training-truth.jsonl"),
        BlogTextLoader("blogtext.csv"),
        EnronLoader("emails.csv"),
        IMDBLoader("IMDB Dataset.csv")
    ]
    
    # 2. Fit Extractor
    if os.path.exists(f"{OUTPUT_DIR}/extractor.pkl"):
        print(f"Loading existing extractor from {OUTPUT_DIR}/extractor.pkl...")
        with open(f"{OUTPUT_DIR}/extractor.pkl", "rb") as f:
            extractor = pickle.load(f)
    else:
        print("Fitting EnhancedFeatureExtractor (Multi-View)...")
        # Load samples to fit
        fit_texts = []
        for l in loaders:
            df = l.load(limit=2000)
            if not df.empty:
                fit_texts.extend(df['text'].tolist())
                
        extractor = EnhancedFeatureExtractor(max_features_char=3000, max_features_pos=1000, max_features_lex=300)
        extractor.fit(fit_texts)
        
        with open(f"{OUTPUT_DIR}/extractor.pkl", "wb") as f:
            pickle.dump(extractor, f)
        
    # 3. Prepare Datasets (Train/Val Split)
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
    train_loaders = [DataLoader(d, batch_size=BATCH_SIZE, shuffle=True) for d in train_datasets]
    val_loader = DataLoader(torch.utils.data.ConcatDataset(val_datasets), batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. Model
    model = DANNSiamese(input_dim=MAX_FEATURES, num_domains=4).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion_auth = nn.BCELoss(reduction='none')
    criterion_domain = nn.CrossEntropyLoss()
    
    print(f"Starting Training (200 Epochs, Patience {PATIENCE})...")
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss_auth = 0
        train_loss_dom = 0
        batches = 0
        
        # Correct Schedule: 0 -> 1
        p = epoch / EPOCHS
        loss_lambda = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        
        # Interleaved Training
        min_len = min([len(l) for l in train_loaders])
        iterators = [iter(l) for l in train_loaders]
        
        for _ in range(min_len):
            batches += 1
            X1_list, X2_list, Y_list, D_list = [], [], [], []
            
            for it in iterators:
                try:
                    x1, x2, y, d = next(it)
                    X1_list.append(x1); X2_list.append(x2); Y_list.append(y); D_list.append(d)
                except StopIteration:
                    pass
            
            X1 = torch.cat(X1_list).to(DEVICE)
            X2 = torch.cat(X2_list).to(DEVICE)
            Y = torch.cat(Y_list).to(DEVICE)
            D = torch.cat(D_list).to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            # GRL Alpha = 1.0 (Standard) or Scheduled? 
            # We control impact via loss_lambda, so let's keep alpha=1.0 for GRL to work fully.
            p_auth, p_dom_a, p_dom_b = model(X1, X2, alpha=1.0)
            p_auth = p_auth.squeeze()
            
            # Auth Loss (Mask IMDB -1)
            mask = (Y != -1).float()
            l_auth_raw = criterion_auth(p_auth, torch.max(Y, torch.tensor(0.0).to(DEVICE)))
            l_auth = (l_auth_raw * mask).sum() / (mask.sum() + 1e-8)
            
            # Domain Loss
            l_dom = (criterion_domain(p_dom_a, D) + criterion_domain(p_dom_b, D)) / 2
            
            # Total Loss
            loss = l_auth + loss_lambda * l_dom
            
            loss.backward()
            optimizer.step()
            
            train_loss_auth += l_auth.item()
            train_loss_dom += l_dom.item()
            
        # Validation
        model.eval()
        val_correct_auth = 0
        val_total_auth = 0
        val_correct_dom = 0
        val_total_dom = 0
        domain_accs = {0: [], 1: [], 2: [], 3: []}
        
        with torch.no_grad():
            for x1, x2, y, d in val_loader:
                x1, x2, y, d = x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE), d.to(DEVICE)
                p_auth, p_dom_a, _ = model(x1, x2, alpha=0.0)
                
                # Auth Acc
                mask = (y != -1)
                preds = (p_auth.squeeze() > 0.5).long()
                
                if mask.sum() > 0:
                    val_correct_auth += (preds[mask] == y[mask].long()).sum().item()
                    val_total_auth += mask.sum().item()
                    
                    # Per domain
                    for dom_id in [0, 1, 2]: # Skip IMDB (3)
                        dom_mask = (d == dom_id) & mask
                        if dom_mask.sum() > 0:
                            acc = (preds[dom_mask] == y[dom_mask].long()).float().mean().item()
                            domain_accs[dom_id].append(acc)
                            
                # Domain Acc
                pred_d = p_dom_a.argmax(dim=1)
                val_correct_dom += (pred_d == d).sum().item()
                val_total_dom += len(d)
                
        val_acc = val_correct_auth / max(1, val_total_auth)
        val_dom_acc = val_correct_dom / max(1, val_total_dom)
        
        epoch_str = f"Epoch {epoch+1}/{EPOCHS}"
        loss_str = f"L_Auth: {train_loss_auth/batches:.3f} | L_Dom: {train_loss_dom/batches:.3f} (w={loss_lambda:.2f})"
        val_str = f"Val Auth: {val_acc*100:.1f}% | Val Dom: {val_dom_acc*100:.1f}%"
        
        print(f"{epoch_str} | {loss_str} | {val_str}")
        print(f"  PAN: {np.mean(domain_accs[0])*100:.1f}% | Blog: {np.mean(domain_accs[1])*100:.1f}% | Enron: {np.mean(domain_accs[2])*100:.1f}%")
        
        # Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/dann_model_v2.pth")
            print("  [Saved Best Model]")
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print("Early Stopping.")
            break

if __name__ == "__main__":
    train_dann()
