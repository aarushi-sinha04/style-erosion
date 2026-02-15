"""
Back-Translation Attack for Authorship Verification
=====================================================
English → German → English using Helsinki-NLP Marian models.
Sentence-level attack (like T5) but different mechanism.

Install first: pip install sentencepiece sacremoses

Usage:
    python attacks/back_translation.py
"""
import sys
import os
import json
import numpy as np
import pickle
import torch
from tqdm import tqdm

from transformers import MarianMTModel, MarianTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.dann import DANNSiameseV3
from utils.data_loader_scie import PAN22Loader
from experiments.train_siamese import SiameseNetwork, preprocess
from utils.feature_extraction import EnhancedFeatureExtractor

DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')


class BackTranslationAttack:
    """English → German → English translation attack."""

    def __init__(self):
        print("Loading Marian EN→DE model...")
        self.en_de_tok = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
        self.en_de_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de').to(DEVICE)

        print("Loading Marian DE→EN model...")
        self.de_en_tok = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-de-en')
        self.de_en_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-de-en').to(DEVICE)

        self.en_de_model.eval()
        self.de_en_model.eval()
        print("Back-translation models ready.")

    def _translate(self, text, model, tokenizer, max_len=512):
        # Truncate long texts
        text = str(text)[:2000]
        inputs = tokenizer(text, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_len).to(DEVICE)
        with torch.no_grad():
            translated = model.generate(**inputs, max_new_tokens=512)
        return tokenizer.decode(translated[0], skip_special_tokens=True)

    def attack(self, text):
        """English → German → English"""
        german = self._translate(text, self.en_de_model, self.en_de_tok)
        back_en = self._translate(german, self.de_en_model, self.de_en_tok)
        return back_en

    def attack_batch(self, texts):
        return [self.attack(t) for t in texts]


# ==============================================================================
# Prediction helpers (same as eval_robust_all.py)
# ==============================================================================
def get_feats(extractor, texts):
    f_dict = extractor.transform(texts)
    return np.hstack([f_dict['char'], f_dict['pos'], f_dict['lex'], f_dict['readability']])


def predict_dann(model, extractor, t1, t2):
    f1 = get_feats(extractor, t1)
    f2 = get_feats(extractor, t2)
    x1 = torch.tensor(f1, dtype=torch.float32).to(DEVICE)
    x2 = torch.tensor(f2, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        p, _, _ = model(x1, x2, alpha=0.0)
    return p.cpu().numpy().reshape(-1)


def predict_siamese(model, vec, scaler, t1, t2):
    p1 = [preprocess(t) for t in t1]
    p2 = [preprocess(t) for t in t2]
    v1 = scaler.transform(vec.transform(p1).toarray())
    v2 = scaler.transform(vec.transform(p2).toarray())
    x1 = torch.tensor(v1, dtype=torch.float32).to(DEVICE)
    x2 = torch.tensor(v2, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(x1, x2)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    return probs


def predict_bert(model, tokenizer, t1, t2, max_len=256):
    """Predict with BERT Siamese."""
    probs_list = []
    for a, b in zip(t1, t2):
        enc1 = tokenizer(str(a)[:1000], max_length=max_len, padding='max_length',
                          truncation=True, return_tensors='pt')
        enc2 = tokenizer(str(b)[:1000], max_length=max_len, padding='max_length',
                          truncation=True, return_tensors='pt')
        with torch.no_grad():
            logits = model(
                enc1['input_ids'].to(DEVICE), enc1['attention_mask'].to(DEVICE),
                enc2['input_ids'].to(DEVICE), enc2['attention_mask'].to(DEVICE))
            prob = torch.sigmoid(logits).cpu().item()
        probs_list.append(prob)
    return np.array(probs_list)


# ==============================================================================
# Main evaluation
# ==============================================================================
def evaluate_backtranslation():
    print("=" * 60)
    print("BACK-TRANSLATION ATTACK EVALUATION")
    print("=" * 60)

    # Load extractor
    extractor = pickle.load(open('results/final_dann/extractor.pkl', 'rb'))

    # Load models
    models = {}

    # DANN models
    for name, path in [('Base DANN', 'results/final_dann/dann_model_v4.pth'),
                       ('Robust DANN', 'results/robust_dann/robust_dann_model.pth')]:
        if os.path.exists(path):
            m = DANNSiameseV3(input_dim=4308, num_domains=4).to(DEVICE)
            m.load_state_dict(torch.load(path, map_location=DEVICE)); m.eval()
            models[name] = {'model': m, 'type': 'dann'}
            print(f"  ✓ {name}")

    # Siamese models
    for name, mpath, vpath, spath, dim in [
        ('PAN22 Siamese', 'results/siamese_baseline/best_model.pth',
         'results/siamese_baseline/vectorizer.pkl', 'results/siamese_baseline/scaler.pkl', 3000),
        ('CD Siamese', 'results/siamese_crossdomain/best_model.pth',
         'results/siamese_crossdomain/vectorizer.pkl', 'results/siamese_crossdomain/scaler.pkl', 5000),
        ('Rob Siamese', 'results/robust_siamese/best_model.pth',
         'results/robust_siamese/vectorizer.pkl', 'results/robust_siamese/scaler.pkl', 5000),
    ]:
        if os.path.exists(mpath):
            vec = pickle.load(open(vpath, 'rb'))
            scaler = pickle.load(open(spath, 'rb'))
            m = SiameseNetwork(input_dim=dim).to(DEVICE)
            m.load_state_dict(torch.load(mpath, map_location=DEVICE)); m.eval()
            models[name] = {'model': m, 'type': 'siamese', 'vec': vec, 'scaler': scaler}
            print(f"  ✓ {name}")

    # BERT Siamese
    bert_path = 'results/bert_baseline/bert_siamese_best.pth'
    if os.path.exists(bert_path):
        try:
            from transformers import BertTokenizer
            from experiments.train_bert_baseline import BERTSiamese
            bert_tok = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_m = BERTSiamese(freeze_bert_layers=8).to(DEVICE)
            bert_m.load_state_dict(torch.load(bert_path, map_location=DEVICE)); bert_m.eval()
            models['BERT Siamese'] = {'model': bert_m, 'type': 'bert', 'tokenizer': bert_tok}
            print(f"  ✓ BERT Siamese")
        except Exception as e:
            print(f"  ✗ BERT Siamese: {e}")

    # Load PAN22 positive pairs
    print("\nLoading PAN22 positive pairs...")
    loader = PAN22Loader("pan22-authorship-verification-training.jsonl",
                         "pan22-authorship-verification-training-truth.jsonl")
    loader.load(limit=6000)
    t1, t2, labels = loader.create_pairs(num_pairs=500)
    labels = np.array(labels)

    # Filter to positive (same-author) pairs only — first 100
    pos_idx = [i for i, l in enumerate(labels) if l == 1][:100]
    anchors = [t1[i] for i in pos_idx]
    positives = [t2[i] for i in pos_idx]
    print(f"  Using {len(anchors)} positive pairs")

    # Generate back-translated versions
    print("\nGenerating back-translations...")
    attacker = BackTranslationAttack()
    attacked = []
    for i, text in enumerate(tqdm(positives, desc="Back-translating")):
        attacked.append(attacker.attack(text))

    # Save cache
    cache = []
    for a, p, atk in zip(anchors, positives, attacked):
        cache.append({'anchor': a, 'positive': p, 'attacked': atk, 'attack_type': 'back_translation'})
    with open('data/backtranslation_adversarial_cache.jsonl', 'w') as f:
        for entry in cache:
            f.write(json.dumps(entry) + '\n')
    print(f"  Saved {len(cache)} samples to data/backtranslation_adversarial_cache.jsonl")

    # Evaluate each model
    print("\n" + "=" * 60)
    print("ASR RESULTS")
    print("=" * 60)
    results = {}

    for name, cfg in models.items():
        print(f"\n  Evaluating {name}...")

        # Get clean predictions
        if cfg['type'] == 'dann':
            clean_p = predict_dann(cfg['model'], extractor, anchors, positives)
            atk_p = predict_dann(cfg['model'], extractor, anchors, attacked)
        elif cfg['type'] == 'siamese':
            clean_p = predict_siamese(cfg['model'], cfg['vec'], cfg['scaler'], anchors, positives)
            atk_p = predict_siamese(cfg['model'], cfg['vec'], cfg['scaler'], anchors, attacked)
        elif cfg['type'] == 'bert':
            clean_p = predict_bert(cfg['model'], cfg['tokenizer'], anchors, positives)
            atk_p = predict_bert(cfg['model'], cfg['tokenizer'], anchors, attacked)

        # ASR: fraction where clean says same-author (>0.5) but attacked says different (<=0.5)
        clean_correct = clean_p > 0.5
        atk_fooled = atk_p <= 0.5
        n_correct = int(clean_correct.sum())
        n_fooled = int((clean_correct & atk_fooled).sum())
        asr = n_fooled / max(n_correct, 1)

        results[name] = {
            'asr': round(float(asr), 4),
            'clean_correct': n_correct,
            'attacked_fooled': n_fooled,
            'total': len(anchors),
        }
        print(f"    ASR={asr:.1%} (fooled {n_fooled}/{n_correct})")

    # Save
    with open('results/backtranslation_attack.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print comparison
    print(f"\n{'='*60}")
    print(f"{'Model':<20} {'BackTrans ASR':>14}")
    print("-" * 34)
    for name, r in results.items():
        print(f"{name:<20} {r['asr']:>13.1%}")

    print(f"\n✅ Results saved to results/backtranslation_attack.json")
    return results


if __name__ == "__main__":
    evaluate_backtranslation()
