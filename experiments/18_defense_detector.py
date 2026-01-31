import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader_scie import PAN22Loader
from models.dann_siamese import DANNSiamese
from attacks.t5_paraphraser import T5Paraphraser
from utils.feature_extraction import EnhancedFeatureExtractor

# Config
OUTPUT_DIR = "results_defense"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
DANN_DIR = "results_dann"

DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
MAX_FEATURES = 4308

class AttackDetector(nn.Module):
    """
    Binary Classifier to detect if a text has been adversarially modified.
    Input: Feature vector (4308 dim).
    Output: Probability of being 'Attacked' (1) vs 'Clean' (0).
    """
    def __init__(self, input_dim=4308):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

def train_detector():
    print("Training Attack Detector (Phase 5)...")
    
    # 1. Load Resources
    with open(f"{DANN_DIR}/extractor.pkl", "rb") as f:
        extractor = pickle.load(f)
        
    paraphraser = T5Paraphraser()
    
    # 2. Generate Dataset (Clean vs Attacked)
    loader = PAN22Loader("pan22-authorship-verification-training.jsonl")
    loader.load(limit=2000)
    texts_clean = loader.df['text'].tolist()[:1000]
    
    print("Generating Attacked Samples...")
    texts_attacked = []
    
    for t in texts_clean:
        try:
            # We assume successful paraphrase is an attack
            adv = paraphraser.paraphrase(t)
            texts_attacked.append(adv)
        except:
            pass
            
    # Balance
    min_len = min(len(texts_clean), len(texts_attacked))
    texts_clean = texts_clean[:min_len]
    texts_attacked = texts_attacked[:min_len]
    
    combined_texts = texts_clean + texts_attacked
    labels = [0]*len(texts_clean) + [1]*len(texts_attacked) # 0=Clean, 1=Attacked
    
    # 3. Extract Features
    def flatten_feats(feats_dict):
        return np.hstack([feats_dict['char'], feats_dict['pos'], feats_dict['lex'], feats_dict['readability']])
    
    def pad(X):
        if X.shape[1] < MAX_FEATURES: 
            return np.hstack([X, np.zeros((X.shape[0], MAX_FEATURES - X.shape[1]))])
        return X[:, :MAX_FEATURES]
        
    print("Extracting Features...")
    f_dict = extractor.transform(combined_texts)
    X = flatten_feats(f_dict)
    X = pad(X)
    y = np.array(labels)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Tensor
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    # 4. Train
    model = AttackDetector().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    for epoch in range(10):
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            pred = model(bx).squeeze()
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}")
        
    # 5. Eval
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        preds = (model(xt).squeeze() > 0.5).cpu().numpy().astype(int)
        
    print("Detector Performance:")
    print(classification_report(y_test, preds, target_names=['Clean', 'Attacked']))
    
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/attack_detector.pth")
    print("Saved Attack Detector.")

if __name__ == "__main__":
    train_detector()
