"""
BERTScore Semantic Preservation Measurement
============================================
Computes BERTScore between original and T5-attacked texts
to verify that attacks preserve meaning.
"""
import json
import numpy as np

def compute_bertscore():
    print("=" * 60)
    print("BERTSCORE SEMANTIC PRESERVATION")
    print("=" * 60)

    # Load cached attacks
    cache_file = "data/eval_adversarial_cache.jsonl"
    samples = []
    with open(cache_file, 'r') as f:
        for line in f:
            samples.append(json.loads(line))

    originals = [s['positive'] for s in samples]
    attacked = [s['attacked'] for s in samples]
    print(f"Loaded {len(samples)} attack pairs")

    # Compute BERTScore
    from bert_score import score
    print("Computing BERTScore (this may take a minute)...")
    P, R, F1 = score(attacked, originals, lang='en', verbose=True)

    results = {
        'precision': round(P.mean().item(), 4),
        'recall': round(R.mean().item(), 4),
        'f1': round(F1.mean().item(), 4),
        'n_samples': len(samples),
        'per_sample_f1': [round(f.item(), 4) for f in F1]
    }

    print(f"\n{'='*40}")
    print(f"BERTScore Precision: {results['precision']:.4f}")
    print(f"BERTScore Recall:    {results['recall']:.4f}")
    print(f"BERTScore F1:        {results['f1']:.4f}")
    print(f"Min F1: {F1.min().item():.4f} | Max F1: {F1.max().item():.4f}")

    with open("results/bertscore_detailed.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to results/bertscore_detailed.json")

if __name__ == "__main__":
    compute_bertscore()
