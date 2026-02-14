
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.adversarial_contrastive_encoder import AdversarialContrastiveEncoder
from utils.paraphraser import Paraphraser
from utils.data_loader_scie import PAN22Loader
from utils.feature_extraction import EnhancedFeatureExtractor

# Config
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
OUTPUT_DIR = "results/ace_training"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

MAX_FEATURES = 3000
BATCH_SIZE = 16 # Small batch size due to on-the-fly T5 generation
EPOCHS = 2 # Demo run

class TripletDataset:
    def __init__(self, loader, samples_per_author=10):
        """
        Creates triplets (Anchor, Positive, Negative)
        """
        print("Loading data for triplets...")
        self.data_by_author = {}
        
        # Load raw data
        df = loader.load(limit=100) # Limit for initial phase
        
        # Group by author (using 'id' as proxy if author id not explicit, 
        # but PAN22 JSONL usually implies pair logic.
        # Actually PAN22 loader creates PAIRS. We need individual texts with author labels.
        # PAN22 training data is pairs. We may not have author IDs explicitly for all.
        # But we have 'same=True/False'.
        
        # Strategy: Use 'same=True' pairs as (Anchor, Positive).
        # Use 'same=False' pairs? No, negative should be random or hard negative.
        # Or just pick a random text from the dataset as negative.
        
        self.positive_pairs = []
        all_texts = []
        
        # Extract from loader dataframe (which has 'text', 'id' etc? No, loader returns DF or lists)
        # PAN22Loader specific: loads pairs.
        # Let's use the loader's internal structure or raw file.
        # Actually, let's just use the `create_pairs` output.
        
        t1, t2, labels = loader.create_pairs(num_pairs=50) # Demo limit
        
        for i in range(len(labels)):
            if labels[i] == 1:
                self.positive_pairs.append((t1[i], t2[i]))
            all_texts.append(t1[i])
            all_texts.append(t2[i])
            
        self.all_texts = all_texts
        print(f"Created {len(self.positive_pairs)} positive pairs for anchors.")
        
    def __len__(self):
        return len(self.positive_pairs)
        
    def __getitem__(self, idx):
        anchor, positive = self.positive_pairs[idx]
        # Negative: random text
        negative = random.choice(self.all_texts)
        # Check simple equality (unlikely collision but good practice)
        while negative == anchor or negative == positive:
            negative = random.choice(self.all_texts)
            
        return anchor, positive, negative

def train_ace():
    print("="*60)
    print("RAVT Phase 1: Adversarial Contrastive Encoder Training")
    print("="*60)
    
    # 1. Initialize Components
    print("\nInitializing Components...")
    
    # Feature Extractor
    extractor = EnhancedFeatureExtractor(max_features_char=MAX_FEATURES)
    # We need to fit it first.
    # Load some data to fit
    loader = PAN22Loader("pan22-authorship-verification-training.jsonl", 
                         "pan22-authorship-verification-training-truth.jsonl")
    
    # Dataset
    triplet_ds = TripletDataset(loader)
    
    # Fit extractor
    print("Fitting feature extractor...")
    extractor.fit(triplet_ds.all_texts)
    
    # Model
    # Calculate dimension dynamically
    input_dim = (len(extractor.char_vectorizer.get_feature_names_out()) + 
                 len(extractor.pos_vectorizer.get_feature_names_out()) + 
                 len(extractor.lex_vectorizer.get_feature_names_out()) + 
                 8)
    print(f"Feature Dimension: {input_dim}")
    model = AdversarialContrastiveEncoder(input_dim=input_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Lower LR for stability
    
    # Attacker
    # Only load if we have GPU or acceptable performance. 
    # For now, let's try.
    attacker = Paraphraser(device=DEVICE)
    
    # 2. Training Loop
    print("\nStarting Training...")
    loss_history = []
    
    # Helper to vectorize batch of texts
    def vectorize(text_list):
        feats = extractor.transform(text_list)
        # Flatten (assuming extractor returns dict)
        # Reuse logic from other scripts if possible, or simple hstack
        # EnhancedFeatureExtractor returns dict with 'char', 'pos' etc.
        # We need to concatenate everything flattened.
        # Wait, ACE model expects single vector.
        # Let's inspect extractor structure again.
        # It usually aligns with the model input.
        # Implementation Plan says "Input dim=3000".
        # EnhancedFeatureExtractor allows configuring dimensions.
        
        # Flatten logic:
        vecs = []
        for k in ['char', 'pos', 'lex', 'readability']:
            if k in feats:
                vecs.append(feats[k])
        
        flat = np.hstack(vecs)
        return torch.tensor(flat, dtype=torch.float32).to(DEVICE)

    model.train()
    
    # Manual batching since we have text
    indices = list(range(len(triplet_ds)))
    
    for epoch in range(EPOCHS):
        random.shuffle(indices)
        total_loss = 0
        steps = 0
        
        # Simplified batch loop
        for i in tqdm(range(0, len(indices), BATCH_SIZE), desc=f"Epoch {epoch+1}"):
            batch_indices = indices[i:i+BATCH_SIZE]
            
            anchors_txt = []
            positives_txt = []
            negatives_txt = []
            
            for idx in batch_indices:
                a, p, n = triplet_ds[idx]
                anchors_txt.append(a)
                positives_txt.append(p)
                negatives_txt.append(n)
            
            # ATTACK (Robustness step)
            # Only attack the positives
            try:
                # We attack the WHOLE batch of positives?
                # T5 generation is slow. Maybe limit to 50% or fewer?
                # Let's attack 50%
                attack_count = len(positives_txt) // 2
                to_attack = positives_txt[:attack_count]
                rest = positives_txt[attack_count:]
                
                attacked_part = attacker.attack(to_attack)
                positives_attacked_txt = attacked_part + rest # "attacked" corresponds to augmented view
                
                # Wait, "positive_attacked" argument in loss implies a specifically attacked version 
                # separate from "positive".
                # The loss signature: contrastive_loss(anchor, positive, positive_attacked, negative)
                # This means we need BOTH Clean Positive AND Attacked Positive for the SAME Anchor.
                # So we must attack ALL positives? Or just use Clean Positive as placeholder if not attacking?
                # Using Clean as Attacked (identity attack) is valid for 50% of data.
                
                # So:
                # 1. Take all positives.
                # 2. Attack ALL of them? Too slow.
                # 3. Attack subset?
                # Let's attack ALL but maybe reduce batch size if slow.
                # Or just attack subset and duplicate clean for the rest.
                
                # Let's try attacking subset.
                attacked_subset = attacker.attack(to_attack)
                # If attack fails (returns original list or None), handle it.
                if not attacked_subset: attacked_subset = to_attack
                
                positives_attacked_final = attacked_subset + rest # First half attacked, second half clean
                
            except Exception as e:
                print(f"Attack failed: {e}")
                positives_attacked_final = positives_txt # Fallback
            
            # Vectorize everything
            # This is the heavy part if extractor is slow (it uses sklearn, so CPU bound)
            emb_anchor = model(vectorize(anchors_txt))
            emb_positive = model(vectorize(positives_txt))
            emb_pos_attacked = model(vectorize(positives_attacked_final)) # Robust view
            emb_negative = model(vectorize(negatives_txt))
            
            # Loss
            loss = model.contrastive_loss(emb_anchor, emb_positive, emb_pos_attacked, emb_negative)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
        avg_loss = total_loss / steps
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        # Save checkcpoint
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/ace_model_epoch_{epoch+1}.pth")

    # 3. Visualization & Validation
    print("\nGenerating Robustness Visualization...")
    visualization(model, extractor, attacker, triplet_ds)
    
def visualization(model, extractor, attacker, ds):
    # Select 50 pairs
    indices = random.sample(range(len(ds)), 50)
    anchors = []
    positives = []
    
    for idx in indices:
        a, p, _ = ds[idx]
        anchors.append(a)
        positives.append(p)
        
    # Attack positives
    print("Generating attacks for visualization...")
    positives_adv = attacker.attack(positives)
    
    # Get embeddings
    with torch.no_grad():
        def vec(txts):
            # Same vectorize logic (need to refactor to reuse)
            feats = extractor.transform(txts)
            vecs = []
            for k in ['char', 'pos', 'lex', 'readability']:
                if k in feats: vecs.append(feats[k])
            flat = np.hstack(vecs)
            return torch.tensor(flat, dtype=torch.float32).to(DEVICE)

        e_clean = model(vec(positives)).cpu().numpy()
        e_adv = model(vec(positives_adv)).cpu().numpy()
        
    # Distances
    dists = np.linalg.norm(e_clean - e_adv, axis=1)
    avg_dist = np.mean(dists)
    print(f"Average Clean-Adv Distance: {avg_dist:.4f} (Lower is better)")
    
    # Plot t-SNE
    all_embs = np.vstack([e_clean, e_adv])
    labels = ["Clean"] * 50 + ["Attacked"] * 50
    
    tsne = TSNE(n_components=2)
    x_2d = tsne.fit_transform(all_embs)
    
    plt.figure(figsize=(10, 8))
    # Draw lines connecting pairs
    for i in range(50):
        plt.plot([x_2d[i,0], x_2d[i+50,0]], [x_2d[i,1], x_2d[i+50,1]], 'k-', alpha=0.1)
        
    plt.scatter(x_2d[:50, 0], x_2d[:50, 1], c='blue', label='Clean')
    plt.scatter(x_2d[50:, 0], x_2d[50:, 1], c='red', label='Attacked')
    plt.legend()
    plt.title(f"Embedding Robustness (Avg Move: {avg_dist:.3f})")
    plt.savefig(f"{OUTPUT_DIR}/robustness_tsne.png")
    print("Saved robustness_tsne.png")


if __name__ == "__main__":
    train_ace()
