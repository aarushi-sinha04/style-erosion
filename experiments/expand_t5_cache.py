"""
Expand T5 adversarial cache from 50 to 100 samples.
Generates 50 NEW attacks using the existing Paraphraser and appends to cache.

Expected time: ~2 hours (50 samples × ~2 min each with 3-pass paraphrasing)
"""
import sys
import os
import json
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from utils.data_loader_scie import PAN22Loader
from utils.paraphraser import Paraphraser
from utils.preprocessing import preprocess_text


def expand_cache():
    """Generate 50 NEW T5 attacks, append to existing cache."""
    cache_path = "data/eval_adversarial_cache.jsonl"
    
    # Load existing cache
    existing = []
    with open(cache_path, 'r') as f:
        for line in f:
            if line.strip():
                existing.append(json.loads(line))
    
    print(f"Existing cache: {len(existing)} samples")
    need = 100 - len(existing)
    if need <= 0:
        print(f"✅ Cache already has {len(existing)} entries (>= 100). Nothing to do.")
        return
    print(f"Target: 100 samples (need {need} new)")
    
    # Build set of existing pairs (use first 100 chars as key)
    existing_keys = set()
    for item in existing:
        key = (item.get('anchor', '')[:100], item.get('positive', '')[:100])
        existing_keys.add(key)
    
    # Load PAN22 test data
    loader = PAN22Loader(
        "pan22-authorship-verification-training.jsonl",
        "pan22-authorship-verification-training-truth.jsonl"
    )
    loader.load(limit=6000)
    t1_list, t2_list, labels = loader.create_pairs(num_pairs=2000)
    
    # Find same-author pairs NOT already in cache
    candidates = []
    for i, label in enumerate(labels):
        if label == 1:  # Same author only
            key = (t1_list[i][:100], t2_list[i][:100])
            if key not in existing_keys:
                candidates.append((t1_list[i], t2_list[i]))
    
    print(f"Found {len(candidates)} candidate same-author pairs not in cache")
    
    if len(candidates) < need:
        print(f"⚠️  Only {len(candidates)} candidates available, will generate all of them")
        need = len(candidates)
    
    # Initialize paraphraser
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    paraphraser = Paraphraser(device=device)
    
    # Generate attacks with 3-pass paraphrasing
    new_attacks = []
    
    for idx, (text_a, text_b) in enumerate(tqdm(candidates[:need], desc="Generating T5 attacks")):
        try:
            # 3-pass aggressive paraphrasing for stronger attacks
            current = text_b
            for pass_num in range(3):
                result = paraphraser.attack([current])
                current = result[0] if result else current
            
            new_attacks.append({
                'anchor': text_a,
                'positive': text_b,
                'attacked': current
            })
        except Exception as e:
            print(f"Error on sample {idx}: {e}")
            # Fallback: single-pass
            try:
                result = paraphraser.attack([text_b])
                new_attacks.append({
                    'anchor': text_a,
                    'positive': text_b,
                    'attacked': result[0] if result else text_b
                })
            except:
                continue
        
        # Incremental save every 10 samples
        if (idx + 1) % 10 == 0:
            with open(cache_path, 'a') as f:
                for item in new_attacks[-(min(10, len(new_attacks))):]:
                    pass  # We'll batch-write at the end
            print(f"  Progress: {idx + 1}/{need} generated")
    
    # Append new attacks to cache
    with open(cache_path, 'a') as f:
        for item in new_attacks:
            f.write(json.dumps(item) + '\n')
    
    final_count = len(existing) + len(new_attacks)
    print(f"\n✅ Expanded cache from {len(existing)} to {final_count} samples")
    print(f"Saved to {cache_path}")
    
    if final_count < 100:
        print(f"⚠️  Only reached {final_count}/100 (not enough unique candidate pairs)")


if __name__ == "__main__":
    expand_cache()
