"""
Classical Baseline: Logistic Regression + Character 3-grams
============================================================
Provides a traditional ML baseline for comparison against
neural approaches. Evaluates on all 3 domains + ASR.
"""
import sys
import os
import json
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader_scie import PAN22Loader, BlogTextLoader, EnronLoader
from utils.preprocessing import preprocess_text

def eval_baselines():
    print("=" * 60)
    print("CLASSICAL BASELINE: LogReg + Char 3-grams")
    print("=" * 60)

    # 1. Train on PAN22 (primary training domain)
    print("\nLoading Training Data (PAN22)...")
    loader = PAN22Loader("pan22-authorship-verification-training.jsonl",
                         "pan22-authorship-verification-training-truth.jsonl")
    loader.load(limit=8000)
    t1_train, t2_train, y_train = loader.create_pairs(num_pairs=3000)

    # Feature: |tfidf(A) - tfidf(B)| (absolute difference of char 3-gram vectors)
    print("Vectorizing (Char 3-grams, 5000 features)...")
    vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 3),
                          max_features=5000, sublinear_tf=True)
    all_texts = t1_train + t2_train
    vec.fit(all_texts)

    X1_train = vec.transform(t1_train).toarray()
    X2_train = vec.transform(t2_train).toarray()
    X_train = np.abs(X1_train - X2_train)  # Feature difference
    y_train = np.array(y_train)

    print(f"  Training set: {len(y_train)} pairs")

    # Train Logistic Regression
    print("Training LogReg...")
    clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    print(f"  Train accuracy: {train_acc*100:.1f}%")

    # 2. Evaluate on all domains
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    domain_loaders = [
        ('PAN22', PAN22Loader("pan22-authorship-verification-training.jsonl",
                               "pan22-authorship-verification-training-truth.jsonl")),
        ('Blog', BlogTextLoader("blogtext.csv")),
        ('Enron', EnronLoader("emails.csv")),
    ]

    results = {}
    for name, dl in domain_loaders:
        print(f"\n  Evaluating {name}...")
        dl.load(limit=5000)
        t1, t2, y = dl.create_pairs(num_pairs=500)
        if not t1:
            print(f"    No pairs")
            continue

        y = np.array(y)
        valid = y != -1
        if valid.sum() == 0:
            print(f"    No labels")
            continue

        X1 = vec.transform(t1).toarray()
        X2 = vec.transform(t2).toarray()
        X_test = np.abs(X1 - X2)

        probs = clf.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)

        acc = accuracy_score(y[valid], preds[valid])
        try:
            roc = roc_auc_score(y[valid], probs[valid])
        except:
            roc = 0.0
        f1 = f1_score(y[valid], preds[valid], zero_division=0)

        results[name] = {'acc': round(acc, 4), 'roc': round(roc, 4), 'f1': round(f1, 4)}
        print(f"    Acc={acc*100:.1f}% ROC={roc:.3f} F1={f1:.3f}")

    # 3. ASR evaluation
    print("\n  Adversarial Robustness...")
    cache_file = "data/eval_adversarial_cache.jsonl"
    if os.path.exists(cache_file):
        cache = []
        with open(cache_file, 'r') as f:
            for line in f:
                cache.append(json.loads(line))

        anchors = [c['anchor'] for c in cache]
        originals = [c['positive'] for c in cache]
        attacked = [c['attacked'] for c in cache]

        success = 0
        valid_orig = 0
        for i in range(len(anchors)):
            x1 = vec.transform([anchors[i]]).toarray()
            x2_orig = vec.transform([originals[i]]).toarray()
            x2_atk = vec.transform([attacked[i]]).toarray()

            p_orig = clf.predict_proba(np.abs(x1 - x2_orig))[0, 1]
            p_atk = clf.predict_proba(np.abs(x1 - x2_atk))[0, 1]

            if p_orig > 0.5:
                valid_orig += 1
                if p_atk < 0.5:
                    success += 1

        asr = success / valid_orig if valid_orig > 0 else 0
        print(f"    ASR: {asr*100:.1f}% ({success}/{valid_orig})")
    else:
        asr = None
        print("    No adversarial cache found")

    # Save
    output = {'accuracy': results, 'asr': round(asr, 4) if asr is not None else None}
    with open("results/baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to results/baseline_results.json")

if __name__ == "__main__":
    eval_baselines()
