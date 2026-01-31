import json
import os
import random
from tqdm import tqdm

# Config
DATA_DIR = "."
TRAIN_FILE = "pan22-authorship-verification-training.jsonl"
TRUTH_FILE = "pan22-authorship-verification-training-truth.jsonl"
OUTPUT_FILE = "pan22_adversarial_train.jsonl"
NUM_aug_SAMPLES = 2000

class HeuristicAttacker:
    def attack(self, text):
        words = text.split()
        new_words = []
        for i, w in enumerate(words):
            # Random Deletion (10%)
            if random.random() < 0.10:
                continue
            
            # Random Swap (10%)
            if i < len(words)-1 and random.random() < 0.10:
                new_words.append(words[i+1])
                new_words.append(words[i])
                # Skip next
                words[i+1] = "" 
            else:
                if w != "":
                    new_words.append(w)
        return " ".join(new_words)

def generate_adversarial_data():
    print(f"Loading {TRAIN_FILE}...")
    
    pairs_dict = {}
    with open(os.path.join(DATA_DIR, TRAIN_FILE), 'r') as f:
        for line in f:
            obj = json.loads(line)
            pairs_dict[obj['id']] = obj['pair']
            
    labels_dict = {}
    with open(os.path.join(DATA_DIR, TRUTH_FILE), 'r') as f:
        for line in f:
            obj = json.loads(line)
            labels_dict[obj['id']] = 1.0 if obj['same'] else 0.0
            
    same_author_ids = [pid for pid, label in labels_dict.items() if label == 1.0]
    
    if len(same_author_ids) > NUM_aug_SAMPLES:
        selected_ids = random.sample(same_author_ids, NUM_aug_SAMPLES)
    else:
        selected_ids = same_author_ids
        
    print(f"Selected {len(selected_ids)} pairs for Fast Augmentation.")
    
    attacker = HeuristicAttacker()
    augmented_data = []
    
    print("Generating FAST Adversarial Samples...")
    for pid in tqdm(selected_ids):
        t1, t2 = pairs_dict[pid]
        
        # Attack t2
        t2_aug = attacker.attack(t2)
        
        entry = {
            'id': f"{pid}_aug",
            'pair': [t1, t2_aug],
            'same': True,
            'is_adversarial': True
        }
        augmented_data.append(entry)
            
    print(f"Saving {len(augmented_data)} pairs to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for item in augmented_data:
            f.write(json.dumps(item) + "\n")
            
    print("Done.")

if __name__ == "__main__":
    generate_adversarial_data()
