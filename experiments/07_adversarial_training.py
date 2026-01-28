import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import re

# Configuration
MODEL_DIR = "results_pan_siamese"
OUTPUT_DIR = "results_pan_adversarial_training"
ORIG_TRAIN_FILE = "pan22-authorship-verification-training.jsonl"
ORIG_TRUTH_FILE = "pan22-authorship-verification-training-truth.jsonl"
ADV_TRAIN_FILE = "pan22_adversarial_train.jsonl"

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 1e-5 # Low LR for fine-tuning

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==============================================================================
# MODEL DEFINITION (Must match original)
# ==============================================================================
HIDDEN_DIM = 512
DROPOUT = 0.3
MAX_FEATURES = 3000

class SiameseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SiameseNetwork, self).__init__()
        self.branch = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(1024, HIDDEN_DIM), nn.BatchNorm1d(HIDDEN_DIM), nn.ReLU(), nn.Dropout(DROPOUT)
        )
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 4, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1)
        )
    def forward_one(self, x): return self.branch(x)
    def forward(self, x1, x2):
        u, v = self.forward_one(x1), self.forward_one(x2)
        return self.head(torch.cat([u, v, torch.abs(u - v), u * v], dim=1))

class PANDataset(Dataset):
    def __init__(self, x1, x2, y):
        self.x1 = torch.tensor(x1, dtype=torch.float32)
        self.x2 = torch.tensor(x2, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x1[idx], self.x2[idx], self.y[idx]

def preprocess(text):
    text = text.replace("<nl>", " ")
    text = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def load_combined_data():
    print("Loading Original Data...")
    pairs_dict = {}
    with open(ORIG_TRAIN_FILE, 'r') as f:
        for line in f:
            obj = json.loads(line)
            pairs_dict[obj['id']] = obj['pair']
    labels_dict = {}
    with open(ORIG_TRUTH_FILE, 'r') as f:
        for line in f:
            obj = json.loads(line)
            labels_dict[obj['id']] = 1.0 if obj['same'] else 0.0
            
    t1s, t2s, ys = [], [], []
    for pid, label in labels_dict.items():
        if pid in pairs_dict:
            t1s.append(pairs_dict[pid][0])
            t2s.append(pairs_dict[pid][1])
            ys.append(label)
            
    print(f"Original Samples: {len(ys)}")
    
    if os.path.exists(ADV_TRAIN_FILE):
        print("Loading Adversarial Data...")
        with open(ADV_TRAIN_FILE, 'r') as f:
            count = 0
            for line in f:
                obj = json.loads(line)
                t1s.append(obj['pair'][0])
                t2s.append(obj['pair'][1])
                ys.append(1.0) # Adv data is paraphrased same-author, so label=1
                count += 1
        print(f"Adversarial Samples: {count}")
    else:
        print("WARNING: Adversarial data not found!")
        
    return t1s, t2s, np.array(ys, dtype=np.float32)

def train_adversarial():
    # 1. Load Resources
    print("Loading Pre-trained Resources...")
    with open(f"{MODEL_DIR}/vectorizer.pkl", "rb") as f: vec = pickle.load(f)
    with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f: scaler = pickle.load(f)
    
    model = SiameseNetwork(input_dim=MAX_FEATURES).to(DEVICE)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/best_model.pth", map_location=DEVICE))
    
    # 2. Load Data
    t1_raw, t2_raw, y = load_combined_data()
    
    # 3. Vectorize (Use EXISTING vectorizer - do not retrain)
    print("Vectorizing...")
    t1_clean = [preprocess(t) for t in tqdm(t1_raw)]
    t2_clean = [preprocess(t) for t in tqdm(t2_raw)]
    
    X1 = vec.transform(t1_clean).toarray()
    X2 = vec.transform(t2_clean).toarray()
    
    # Scale
    X1 = scaler.transform(X1)
    X2 = scaler.transform(X2)
    
    # 4. Split
    indices = list(range(len(y)))
    idx_train, idx_val = train_test_split(indices, test_size=0.15, random_state=42, stratify=y)
    
    train_ds = PANDataset(X1[idx_train], X2[idx_train], y[idx_train])
    val_ds = PANDataset(X1[idx_val], X2[idx_val], y[idx_val])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 5. Fine-Tune
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Fine-Tuning on {len(train_ds)} samples for {EPOCHS} epochs...")
    best_acc = 0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x1, x2, label in train_loader:
            x1, x2, label = x1.to(DEVICE), x2.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x1, x2)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Eval
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x1, x2, label in val_loader:
                x1, x2, label = x1.to(DEVICE), x2.to(DEVICE), label.to(DEVICE)
                logits = model(x1, x2)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend([1 if p>=0.5 else 0 for p in probs])
                all_labels.extend(label.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f} | Val Acc {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model_robust.pth")
            
    print(f"Training Complete. Best Acc: {best_acc:.4f}")
    print(f"Saved robust model to {OUTPUT_DIR}/best_model_robust.pth")

if __name__ == "__main__":
    train_adversarial()
