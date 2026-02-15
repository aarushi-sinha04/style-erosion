"""
Evaluate BERT Siamese on Existing Attack Caches
=================================================
Runs BERT through synonym and T5 adversarial caches
to complete the attack comparison table.

Usage:
    python experiments/eval_bert_attacks.py
"""
import sys
import os
import json
import numpy as np
import torch
from tqdm import tqdm

from transformers import BertTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.train_bert_baseline import BERTSiamese

DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'results/bert_baseline/bert_siamese_best.pth'
MAX_LEN = 256


def predict_pair(model, tokenizer, text_a, text_b):
    """Predict same-author probability for one pair."""
    text_a = str(text_a)[:1000]
    text_b = str(text_b)[:1000]
    enc1 = tokenizer(text_a, max_length=MAX_LEN, padding='max_length',
                      truncation=True, return_tensors='pt')
    enc2 = tokenizer(text_b, max_length=MAX_LEN, padding='max_length',
                      truncation=True, return_tensors='pt')
    with torch.no_grad():
        logits = model(
            enc1['input_ids'].to(DEVICE), enc1['attention_mask'].to(DEVICE),
            enc2['input_ids'].to(DEVICE), enc2['attention_mask'].to(DEVICE))
        prob = torch.sigmoid(logits).cpu().item()
    return prob


def eval_on_cache(model, tokenizer, cache_path, attack_name):
    """Evaluate BERT ASR on an adversarial cache."""
    if not os.path.exists(cache_path):
        print(f"  ⚠ {cache_path} not found")
        return None

    with open(cache_path) as f:
        samples = [json.loads(line) for line in f]

    print(f"\n  {attack_name}: {len(samples)} samples")

    clean_correct = 0
    attacked_fooled = 0
    total = 0

    for s in tqdm(samples, desc=f"  {attack_name}"):
        anchor = s['anchor']
        positive = s['positive']
        attacked = s['attacked']

        clean_prob = predict_pair(model, tokenizer, anchor, positive)
        total += 1

        if clean_prob > 0.5:  # BERT says "same author"
            clean_correct += 1
            atk_prob = predict_pair(model, tokenizer, anchor, attacked)
            if atk_prob <= 0.5:  # Attack flips to "different author"
                attacked_fooled += 1

    asr = attacked_fooled / max(clean_correct, 1)
    result = {
        'asr': round(float(asr), 4),
        'clean_correct': clean_correct,
        'attacked_fooled': attacked_fooled,
        'total': total,
    }
    print(f"    ASR={asr:.1%} ({attacked_fooled}/{clean_correct})")
    return result


def main():
    print("=" * 60)
    print("BERT SIAMESE ATTACK EVALUATION")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # Load BERT
    if not os.path.exists(MODEL_PATH):
        print(f"✗ Model not found at {MODEL_PATH}")
        return

    print("Loading BERT Siamese...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BERTSiamese(freeze_bert_layers=8).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("  ✓ Loaded")

    results = {}

    # Synonym attack
    r = eval_on_cache(model, tokenizer, 'data/synonym_adversarial_cache.jsonl', 'Synonym')
    if r: results['Synonym'] = r

    # T5 paraphrase attack
    r = eval_on_cache(model, tokenizer, 'data/eval_adversarial_cache.jsonl', 'T5 Paraphrase')
    if r: results['T5 Paraphrase'] = r

    # Back-translation (if available)
    r = eval_on_cache(model, tokenizer, 'data/backtranslation_adversarial_cache.jsonl', 'Back-Translation')
    if r: results['Back-Translation'] = r

    # Save
    with open('results/bert_attack_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("BERT Siamese Attack Results:")
    for attack, data in results.items():
        print(f"  {attack:20s}: ASR={data['asr']:.1%} ({data['attacked_fooled']}/{data['clean_correct']})")
    print(f"\n✅ Saved to results/bert_attack_results.json")


if __name__ == "__main__":
    main()
