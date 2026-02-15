"""
Error Analysis for Rob Siamese
================================
Categorizes false positives and false negatives, analyzes
patterns by text length, n-gram overlap, and domain.
"""
import sys
import os
import json
import re
import numpy as np
import pickle
from collections import Counter

import torch
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader_scie import PAN22Loader, BlogTextLoader, EnronLoader

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

def preprocess(text):
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text[:5000]

def compute_char_4gram_overlap(a, b):
    a_grams = set(a[i:i+4] for i in range(len(a)-3))
    b_grams = set(b[i:i+4] for i in range(len(b)-3))
    if not a_grams or not b_grams:
        return 0.0
    return len(a_grams & b_grams) / len(a_grams | b_grams)

def run_error_analysis():
    print("=" * 60)
    print("ERROR ANALYSIS: Rob Siamese")
    print("=" * 60)

    # Load model
    from experiments.train_siamese_crossdomain import SiameseNetwork
    model_path = "results/robust_siamese/best_model.pth"
    vec = pickle.load(open("results/robust_siamese/vectorizer.pkl", "rb"))
    scaler = pickle.load(open("results/robust_siamese/scaler.pkl", "rb"))
    input_dim = len(vec.get_feature_names_out())
    model = SiameseNetwork(input_dim=input_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Collect predictions per domain
    domain_loaders = [
        ('PAN22', PAN22Loader("pan22-authorship-verification-training.jsonl",
                               "pan22-authorship-verification-training-truth.jsonl")),
        ('Blog', BlogTextLoader("blogtext.csv")),
        ('Enron', EnronLoader("emails.csv")),
    ]

    all_results = {}
    fp_examples, fn_examples = [], []

    for domain_name, dl in domain_loaders:
        print(f"\n--- {domain_name} ---")
        dl.load(limit=5000)
        t1, t2, y = dl.create_pairs(num_pairs=500)
        if not t1:
            continue

        y = np.array(y)
        valid = y != -1
        t1 = [t1[i] for i in range(len(t1)) if valid[i]]
        t2 = [t2[i] for i in range(len(t2)) if valid[i]]
        y = y[valid]

        # Predict
        pt1 = [preprocess(t) for t in t1]
        pt2 = [preprocess(t) for t in t2]
        X1 = torch.tensor(scaler.transform(vec.transform(pt1).toarray()), dtype=torch.float32).to(DEVICE)
        X2 = torch.tensor(scaler.transform(vec.transform(pt2).toarray()), dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            logits = model(X1, X2)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

        preds = (probs > 0.5).astype(int)
        cm = confusion_matrix(y, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        print(f"  TP={tp} TN={tn} FP={fp} FN={fn}")

        # Analyze errors
        for i in range(len(y)):
            overlap = compute_char_4gram_overlap(pt1[i], pt2[i])
            len_a = len(pt1[i])
            len_b = len(pt2[i])
            len_ratio = max(len_a, len_b) / (min(len_a, len_b) + 1)

            info = {
                'domain': domain_name,
                'true_label': int(y[i]),
                'pred_prob': float(round(probs[i], 4)),
                'overlap': round(overlap, 4),
                'len_a': len_a,
                'len_b': len_b,
                'len_ratio': round(len_ratio, 2),
                'text_a_snippet': pt1[i][:100],
                'text_b_snippet': pt2[i][:100],
            }

            if preds[i] == 1 and y[i] == 0:  # FP: predicted same, actually different
                fp_examples.append(info)
            elif preds[i] == 0 and y[i] == 1:  # FN: predicted different, actually same
                fn_examples.append(info)

        # Domain stats
        domain_fp = [e for e in fp_examples if e['domain'] == domain_name]
        domain_fn = [e for e in fn_examples if e['domain'] == domain_name]
        all_results[domain_name] = {
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
            'n_pairs': len(y),
        }

    # Summary
    print("\n" + "=" * 60)
    print("ERROR PATTERNS")
    print("=" * 60)

    if fp_examples:
        fp_overlaps = [e['overlap'] for e in fp_examples]
        fp_ratios = [e['len_ratio'] for e in fp_examples]
        print(f"\nFALSE POSITIVES ({len(fp_examples)} total): Predicted SAME, Actually DIFFERENT")
        print(f"  Avg 4-gram overlap: {np.mean(fp_overlaps):.3f} (±{np.std(fp_overlaps):.3f})")
        print(f"  Avg length ratio:   {np.mean(fp_ratios):.2f}")
        print(f"  By domain: {Counter(e['domain'] for e in fp_examples)}")
        print(f"  Avg confidence:     {np.mean([e['pred_prob'] for e in fp_examples]):.3f}")

    if fn_examples:
        fn_overlaps = [e['overlap'] for e in fn_examples]
        fn_ratios = [e['len_ratio'] for e in fn_examples]
        print(f"\nFALSE NEGATIVES ({len(fn_examples)} total): Predicted DIFFERENT, Actually SAME")
        print(f"  Avg 4-gram overlap: {np.mean(fn_overlaps):.3f} (±{np.std(fn_overlaps):.3f})")
        print(f"  Avg length ratio:   {np.mean(fn_ratios):.2f}")
        print(f"  By domain: {Counter(e['domain'] for e in fn_examples)}")
        print(f"  Avg confidence:     {np.mean([e['pred_prob'] for e in fn_examples]):.3f}")

    # Key finding
    if fp_examples and fn_examples:
        print(f"\nKEY FINDING:")
        print(f"  FP overlap ({np.mean(fp_overlaps):.3f}) vs FN overlap ({np.mean(fn_overlaps):.3f})")
        if np.mean(fp_overlaps) > np.mean(fn_overlaps):
            print(f"  → FPs have HIGHER n-gram overlap: different authors with similar style")
            print(f"  → FNs have LOWER n-gram overlap: same author writing differently (topic shift)")

    # Save
    output = {
        'domain_stats': all_results,
        'false_positives': fp_examples[:20],  # Top 20
        'false_negatives': fn_examples[:20],
        'summary': {
            'total_fp': len(fp_examples),
            'total_fn': len(fn_examples),
            'fp_avg_overlap': round(np.mean([e['overlap'] for e in fp_examples]), 4) if fp_examples else 0,
            'fn_avg_overlap': round(np.mean([e['overlap'] for e in fn_examples]), 4) if fn_examples else 0,
            'fp_avg_confidence': round(np.mean([e['pred_prob'] for e in fp_examples]), 4) if fp_examples else 0,
            'fn_avg_confidence': round(np.mean([e['pred_prob'] for e in fn_examples]), 4) if fn_examples else 0,
        }
    }
    with open("results/error_analysis.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to results/error_analysis.json")

if __name__ == "__main__":
    run_error_analysis()
