import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from tqdm import tqdm

# Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader_scie import PAN22Loader, BlogTextLoader
from models.dann_siamese import DANNSiamese
from attacks.t5_paraphraser import T5Paraphraser
from utils.feature_extraction import EnhancedFeatureExtractor

# Config
OUTPUT_DIR = "results_dann"
ADV_OUTPUT_DIR = "results_dann_adv"
if not os.path.exists(ADV_OUTPUT_DIR): os.makedirs(ADV_OUTPUT_DIR)

DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
MAX_FEATURES = 4308
BATCH_SIZE = 32
EPOCHS = 10 # Fine-tuning
SAMPLES = 1000

def train_adversarial():
    print("Loading Resources...")
    if not os.path.exists(f"{OUTPUT_DIR}/dann_model_v2.pth"):
        print("V2 Model not found. Please wait for optimization to finish.")
        return

    # 1. Load Extractor
    with open(f"{OUTPUT_DIR}/extractor.pkl", "rb") as f:
        extractor = pickle.load(f)
        
    # 2. Load Model
    model = DANNSiamese(input_dim=MAX_FEATURES, num_domains=4).to(DEVICE)
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/dann_model_v2.pth", map_location=DEVICE))
    model.train()
    
    # 3. Load Paraphraser
    paraphraser = T5Paraphraser()
    
    # 4. Load Data (Focus on PAN22 for now)
    loader = PAN22Loader("pan22-authorship-verification-training.jsonl", "pan22-authorship-verification-training-truth.jsonl")
    loader.load(limit=2000)
    t1_list, t2_list, labels = loader.create_pairs(num_pairs=SAMPLES)
    
    # 5. Generate Adversarial Data (Offline for speed)
    print("Generating Adversarial Examples...")
    adv_t1 = []
    adv_t2 = []
    adv_y = []
    
    # Helper to flatten
    def flatten_feats(feats_dict):
        return np.hstack([
            feats_dict['char'], 
            feats_dict['pos'], 
            feats_dict['lex'], 
            feats_dict['readability']
        ])
    
    for i in tqdm(range(len(t1_list))):
        t1, t2, y = t1_list[i], t2_list[i], labels[i]
        
        # Clean Pair
        adv_t1.append(t1)
        adv_t2.append(t2)
        adv_y.append(y)
        
        # Adversarial Pair (Only for Positive Pairs)
        # If Same Author, Paraphrasing T2 should STILL be Same Author.
        if y == 1:
            try:
                t2_adv = paraphraser.paraphrase(t2)
                adv_t1.append(t1)
                adv_t2.append(t2_adv)
                adv_y.append(1) # Label is still 1
            except:
                pass
                
    # 6. Extract Features
    print("Extracting Features...")
    f1_dict = extractor.transform(adv_t1)
    f2_dict = extractor.transform(adv_t2)
    X1 = flatten_feats(f1_dict)
    X2 = flatten_feats(f2_dict)
    
    # Padding if needed
    def pad(X, dim):
        if X.shape[1] < dim:
             return np.hstack([X, np.zeros((X.shape[0], dim - X.shape[1]))])
        return X[:, :dim]
        
    X1 = pad(X1, MAX_FEATURES)
    X2 = pad(X2, MAX_FEATURES)
    
    dataset = TensorDataset(
        torch.tensor(X1, dtype=torch.float32), 
        torch.tensor(X2, dtype=torch.float32), 
        torch.tensor(adv_y, dtype=torch.float32)
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 7. Fine-Tune Loop
    optimizer = optim.Adam(model.parameters(), lr=5e-5) # Smaller LR
    criterion = nn.BCELoss()
    
    print("Starting Adversarial Fine-Tuning...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for x1, x2, y in dataloader:
            x1, x2, y = x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            p_auth, _, _ = model(x1, x2, alpha=0.0) # Disable domain loss for fine-tuning? Or keep?
            # User wants robust model. Let's keep domain loss low or 0.
            # Focus on robustness now. Alpha=0.0 effectively freezes domain branch influence.
            
            loss = criterion(p_auth.squeeze(), y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}: Loss {total_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), f"{ADV_OUTPUT_DIR}/dann_adv_model.pth")
    print("Saved Adversarial Model.")

if __name__ == "__main__":
    train_adversarial()
