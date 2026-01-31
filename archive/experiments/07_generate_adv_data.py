import json
import os
import sys
import random
from tqdm import tqdm
import torch

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.attack_models import MultiPassParaphraser

# Config
DATA_DIR = "."
TRAIN_FILE = "pan22-authorship-verification-training.jsonl"
TRUTH_FILE = "pan22-authorship-verification-training-truth.jsonl"
OUTPUT_FILE = "pan22_adversarial_train.jsonl"
NUM_aug_SAMPLES = 50

def generate_adversarial_data():
    print(f"Loading {TRAIN_FILE}...")
    
    # Load Training Pairs
    pairs_dict = {}
    with open(os.path.join(DATA_DIR, TRAIN_FILE), 'r') as f:
        for line in f:
            obj = json.loads(line)
            pairs_dict[obj['id']] = obj['pair']
            
    # Load Labels
    labels_dict = {}
    with open(os.path.join(DATA_DIR, TRUTH_FILE), 'r') as f:
        for line in f:
            obj = json.loads(line)
            labels_dict[obj['id']] = 1.0 if obj['same'] else 0.0
            
    # Filter for SAME Author only (can't attack Diff Author easily without changing meaning too much or just keeping it diff)
    # Actually, attacking SAME -> DIFF (obfuscation) is the goal.
    same_author_ids = [pid for pid, label in labels_dict.items() if label == 1.0]
    
    # Select subset
    if len(same_author_ids) > NUM_aug_SAMPLES:
        selected_ids = random.sample(same_author_ids, NUM_aug_SAMPLES)
    else:
        selected_ids = same_author_ids
        
    print(f"Selected {len(selected_ids)} same-author pairs for augmentation.")
    
    # Load Attacker
    attacker = MultiPassParaphraser()
    
    augmented_data = []
    
    print("Generating Adversarial Samples...")
    for pid in tqdm(selected_ids):
        t1, t2 = pairs_dict[pid]
        
        try:
            # Attack t2
            # We use 1 pass for speed
            t2_aug = attacker.attack(t2, passes=1)
            
            # Create new entry
            # ID format: original_ID + _aug
            new_id = f"{pid}_aug"
            
            entry = {
                'id': new_id,
                'pair': [t1, t2_aug],
                'same': True, # Still same author content-wise, but style shifted
                'is_adversarial': True
            }
            augmented_data.append(entry)
            
        except Exception as e:
            print(f"Error processing {pid}: {e}")
            continue
            
    # Save
    print(f"Saving {len(augmented_data)} augmented pairs to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for item in augmented_data:
            f.write(json.dumps(item) + "\n")
            
    print("Done generating adversarial data.")

if __name__ == "__main__":
    generate_adversarial_data()
