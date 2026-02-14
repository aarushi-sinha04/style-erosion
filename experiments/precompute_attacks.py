import sys
import os
import json
from tqdm import tqdm
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from utils.data_loader_scie import PAN22Loader
from utils.paraphraser import Paraphraser
from utils.preprocessing import preprocess_text

def precompute_attacks(limit=1000):
    output_file = "data/pan22_adversarial.jsonl"
    if not os.path.exists("data"): os.makedirs("data")
    
    # 1. Load Data
    loader = PAN22Loader("pan22-authorship-verification-training.jsonl", 
                         "pan22-authorship-verification-training-truth.jsonl")
    
    # Get pairs
    t1_list, t2_list, labels = loader.create_pairs(num_pairs=limit)
    
    # Filter for SAME author (we only attack positive pairs for contrastive learning)
    positive_indices = [i for i, label in enumerate(labels) if label == 1]
    # Apply preprocessing
    texts_to_attack = [preprocess_text(t2_list[i]) for i in positive_indices]
    anchors_list = [preprocess_text(t1_list[i]) for i in positive_indices]
    
    print(f"Found {len(texts_to_attack)} positive pairs to attack.")
    
    # 2. Initialize Paraphraser
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    paraphraser = Paraphraser(device=device)
    
    # 3. Generate Loop
    results = []
    
    # Batch size for generation
    BATCH_SIZE = 8
    
    for i in tqdm(range(0, len(texts_to_attack), BATCH_SIZE)):
        batch = texts_to_attack[i:i+BATCH_SIZE]
        
        try:
            attacked_batch = paraphraser.attack(batch)
        except Exception as e:
            print(f"Error iterating batch {i}: {e}")
            attacked_batch = batch # Fallback
            
        # Store
        for j, original_text in enumerate(batch):
            anchor = anchors_list[i+j]
            orig_idx = positive_indices[i+j] # Keep ID ref if needed
            
            entry = {
                'id': f"aug_{orig_idx}",
                'anchor': anchor,
                'positive': original_text,
                'positive_attacked': attacked_batch[j], # Attack output is usually cleanish T5 text
                'label': 1
            }
            results.append(entry)
            
        # Save incrementally
        if i % 100 == 0:
             with open(output_file, 'w') as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
                    
    # Final save
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    print(f"Saved {len(results)} adversarial training examples to {output_file}")

if __name__ == "__main__":
    precompute_attacks()
