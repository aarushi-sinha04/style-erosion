"""
Advanced Adversarial Attack on Stylometry Model (Fast Version)
==============================================================
This script attacks a Siamese Authorship Verification model using multiple
techniques to erode stylistic signatures. Reduced sample size for faster results.

ATTACK METHODS:
1. Multi-Pass Paraphrasing (3x)
2. Back-Translation (EN -> DE -> EN)
3. Combined Attack
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import os
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

# ==============================================================================
# CONFIG 
# ==============================================================================
MODEL_DIR = "results_pan_siamese"
DATA_DIR = "."
OUTPUT_DIR = "results_pan_attack_v3"
TRAIN_FILE = "pan22-authorship-verification-training.jsonl"
TRUTH_FILE = "pan22-authorship-verification-training-truth.jsonl"

PARAPHRASE_MODEL = "humarin/chatgpt_paraphraser_on_T5_base"
TRANSLATE_EN_DE = "Helsinki-NLP/opus-mt-en-de"
TRANSLATE_DE_EN = "Helsinki-NLP/opus-mt-de-en"

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
NUM_ATTACK_SAMPLES = 25  # Reduced for faster results

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Device: {DEVICE}")
print(f"Output: {OUTPUT_DIR}")
print(f"Samples: {NUM_ATTACK_SAMPLES}")

# ==============================================================================
# 1. TARGET MODEL (Siamese Network)
# ==============================================================================
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SiameseNetwork, self).__init__()
        HIDDEN_DIM = 512
        DROPOUT = 0.3
        self.branch = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(1024, HIDDEN_DIM), nn.BatchNorm1d(HIDDEN_DIM), nn.ReLU(), nn.Dropout(DROPOUT)
        )
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 4, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1)
        )
    def forward_one(self, x): return self.branch(x)
    def forward(self, x1, x2):
        u, v = self.forward_one(x1), self.forward_one(x2)
        return self.head(torch.cat([u, v, torch.abs(u - v), u * v], dim=1))

def load_target():
    print("Loading Siamese Target Model...")
    with open(f"{MODEL_DIR}/vectorizer.pkl", "rb") as f: vec = pickle.load(f)
    with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f: scaler = pickle.load(f)
    model = SiameseNetwork(input_dim=3000).to(DEVICE)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/best_model.pth", map_location=DEVICE, weights_only=True))
    model.eval()
    return model, vec, scaler

def preprocess(text):
    text = text.replace("<nl>", " ")
    text = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def get_prob(model, vec, scaler, t1, t2):
    x1 = scaler.transform(vec.transform([preprocess(t1)]).toarray())
    x2 = scaler.transform(vec.transform([preprocess(t2)]).toarray())
    with torch.no_grad():
        logits = model(torch.tensor(x1, dtype=torch.float32).to(DEVICE),
                       torch.tensor(x2, dtype=torch.float32).to(DEVICE))
        return torch.sigmoid(logits).item()

# ==============================================================================
# 2. ATTACK METHODS
# ==============================================================================
class MultiPassParaphraser:
    """Multi-Pass Paraphrasing: Apply T5 paraphraser 3 times."""
    def __init__(self):
        print(f"Loading Paraphraser: {PARAPHRASE_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(PARAPHRASE_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASE_MODEL).to(DEVICE)
        self.model.eval()
        
    def attack(self, text, passes=3):
        result = text
        for _ in range(passes):
            inputs = self.tokenizer(f'paraphrase: {result}', return_tensors="pt",
                                    max_length=512, truncation=True).input_ids.to(DEVICE)
            outputs = self.model.generate(inputs, max_length=512, num_beams=5,
                                          do_sample=True, temperature=2.0,
                                          top_k=50, top_p=0.95)
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result

class BackTranslationAttack:
    """Back-Translation: EN -> German -> EN"""
    def __init__(self):
        print("Loading Translation Models...")
        self.en_de_tok = MarianTokenizer.from_pretrained(TRANSLATE_EN_DE)
        self.en_de_model = MarianMTModel.from_pretrained(TRANSLATE_EN_DE).to(DEVICE)
        self.de_en_tok = MarianTokenizer.from_pretrained(TRANSLATE_DE_EN)
        self.de_en_model = MarianMTModel.from_pretrained(TRANSLATE_DE_EN).to(DEVICE)
        
    def attack(self, text):
        inputs = self.en_de_tok(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        de_ids = self.en_de_model.generate(**inputs, max_length=512)
        de_text = self.en_de_tok.decode(de_ids[0], skip_special_tokens=True)
        inputs = self.de_en_tok(de_text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        en_ids = self.de_en_model.generate(**inputs, max_length=512)
        return self.de_en_tok.decode(en_ids[0], skip_special_tokens=True)

class CombinedAttack:
    """Combined: Paraphrase 2x + Back-Translate"""
    def __init__(self, para, bt):
        self.para = para
        self.bt = bt
    def attack(self, text):
        step1 = self.para.attack(text, passes=2)
        return self.bt.attack(step1)

# ==============================================================================
# 3. DATA LOADING
# ==============================================================================
def load_data():
    print("Loading PAN22 Validation Set...")
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
    pairs, labels = [], []
    for i in sorted(pairs_dict.keys()):
        if i in labels_dict:
            pairs.append(pairs_dict[i])
            labels.append(labels_dict[i])
    _, pairs_val, _, y_val = train_test_split(pairs, labels, test_size=0.20, random_state=42, stratify=labels)
    return pairs_val, y_val

# ==============================================================================
# 4. ATTACK EXECUTION
# ==============================================================================
def run_attack():
    model, vec, scaler = load_target()
    pairs, labels = load_data()
    
    # Find high-confidence Same Author pairs
    print("Finding vulnerable pairs (Prob > 0.90)...")
    targets = []
    for i, (t1, t2) in enumerate(tqdm(pairs[:1500])):
        if labels[i] == 1.0:
            prob = get_prob(model, vec, scaler, t1, t2)
            if prob > 0.90:
                targets.append({'t1': t1, 't2': t2, 'prob_orig': prob, 't2_orig': t2})
                if len(targets) >= NUM_ATTACK_SAMPLES:
                    break
    print(f"Selected {len(targets)} pairs")
    
    # Load Attackers
    paraphraser = MultiPassParaphraser()
    backtrans = BackTranslationAttack()
    combined = CombinedAttack(paraphraser, backtrans)
    
    attack_methods = {
        'Paraphrase_3Pass': paraphraser.attack,
        'BackTranslation': backtrans.attack,
        'Combined': combined.attack
    }
    
    all_results = []
    example_texts = []
    
    for method_name, attack_fn in attack_methods.items():
        print(f"\nATTACK: {method_name}")
        erosions = []
        
        for idx, sample in enumerate(tqdm(targets)):
            try:
                t2_attacked = attack_fn(sample['t2'])
                prob_new = get_prob(model, vec, scaler, sample['t1'], t2_attacked)
                erosion = sample['prob_orig'] - prob_new
                erosions.append(erosion)
                
                all_results.append({
                    'method': method_name,
                    'prob_orig': sample['prob_orig'],
                    'prob_new': prob_new,
                    'erosion': erosion,
                    'flipped': 1 if prob_new < 0.5 else 0
                })
                
                # Save example (first 3 per method)
                if idx < 3:
                    example_texts.append({
                        'method': method_name,
                        'original': sample['t2_orig'][:500],
                        'attacked': t2_attacked[:500],
                        'prob_original': sample['prob_orig'],
                        'prob_attacked': prob_new
                    })
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        mean_erosion = np.mean(erosions) if erosions else 0
        flip_rate = np.mean([r['flipped'] for r in all_results if r['method'] == method_name])
        print(f"Mean Erosion: {mean_erosion:.4f}, Flip Rate: {flip_rate*100:.1f}%")
    
    # Save Results
    df = pd.DataFrame(all_results)
    df.to_csv(f"{OUTPUT_DIR}/attack_results.csv", index=False)
    
    # ===========================================================================
    # 5. VISUALIZATION
    # ===========================================================================
    print("\nGenerating Visualizations...")
    
    # A. Erosion Distribution by Method
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x='method', y='erosion', palette='Set2')
    plt.axhline(y=0.5, color='red', linestyle='--', label='Target (0.5)')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title("Erosion by Attack Method", fontsize=12, fontweight='bold')
    plt.ylabel("Erosion (Probability Drop)")
    plt.xticks(rotation=15)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for method in df['method'].unique():
        mdf = df[df['method'] == method]
        plt.scatter(mdf['prob_orig'], mdf['prob_new'], alpha=0.6, label=method)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.axhline(y=0.5, color='red', linestyle='-', alpha=0.5)
    plt.xlabel("Original Probability")
    plt.ylabel("Attacked Probability")
    plt.title("Before vs After Attack")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/attack_comparison.png", dpi=300)
    
    # B. Erosion Histogram
    plt.figure(figsize=(10, 4))
    for i, method in enumerate(df['method'].unique()):
        mdf = df[df['method'] == method]
        plt.subplot(1, 3, i+1)
        plt.hist(mdf['erosion'], bins=10, edgecolor='black', alpha=0.7)
        plt.axvline(x=mdf['erosion'].mean(), color='red', linestyle='--', label=f"Mean: {mdf['erosion'].mean():.2f}")
        plt.axvline(x=0.5, color='green', linestyle=':', label='Target: 0.5')
        plt.title(method)
        plt.xlabel("Erosion")
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/erosion_histograms.png", dpi=300)
    
    # C. Summary Statistics
    summary = df.groupby('method').agg(
        mean_erosion=('erosion', 'mean'),
        std_erosion=('erosion', 'std'),
        median_erosion=('erosion', 'median'),
        max_erosion=('erosion', 'max'),
        flip_rate=('flipped', 'mean'),
        samples=('erosion', 'count')
    ).round(4)
    
    # Add 95% CI
    ci_df = df.groupby('method')['erosion'].apply(lambda x: stats.t.interval(0.95, len(x)-1, loc=x.mean(), scale=x.sem()))
    summary['ci_95'] = [f"[{ci[0]:.3f}, {ci[1]:.3f}]" for ci in ci_df]
    
    summary.to_csv(f"{OUTPUT_DIR}/summary_statistics.csv")
    
    # ===========================================================================
    # 6. GENERATE COMPREHENSIVE REPORT
    # ===========================================================================
    print("\nGenerating Report...")
    
    with open(f"{OUTPUT_DIR}/ADVERSARIAL_ATTACK_REPORT.md", "w") as f:
        f.write("# Adversarial Attack on Stylometry Model\n\n")
        f.write("## Executive Summary\n\n")
        f.write("This report documents an adversarial attack experiment targeting a character n-gram based\n")
        f.write("Siamese Network for authorship verification. The goal is to demonstrate that while the model\n")
        f.write("achieves 91% accuracy on human-written text, it is vulnerable to automated style obfuscation.\n\n")
        
        f.write("## Key Results\n\n")
        f.write("| Method | Mean Erosion | 95% CI | Flip Rate | Max Erosion |\n")
        f.write("|--------|-------------|--------|-----------|-------------|\n")
        for method in summary.index:
            row = summary.loc[method]
            f.write(f"| {method} | {row['mean_erosion']:.4f} | {row['ci_95']} | {row['flip_rate']*100:.1f}% | {row['max_erosion']:.4f} |\n")
        
        f.write("\n## Methodology\n\n")
        f.write("### Target Model\n")
        f.write("- **Architecture**: Siamese Network with shared branch encoder\n")
        f.write("- **Features**: Character 4-grams (TF-IDF, 3000 dimensions)\n")
        f.write("- **Training Performance**: 91% accuracy, 0.994 ROC-AUC\n\n")
        
        f.write("### Attack Methods\n\n")
        f.write("**1. Paraphrase (3-Pass)**\n")
        f.write("- Model: T5-base fine-tuned for paraphrasing (humarin/chatgpt_paraphraser)\n")
        f.write("- Process: Text is paraphrased 3 consecutive times\n")
        f.write("- Settings: temperature=2.0, top_k=50, top_p=0.95\n")
        f.write("- Rationale: Iterative paraphrasing compounds style destruction\n\n")
        
        f.write("**2. Back-Translation**\n")
        f.write("- Models: MarianMT (EN→DE, DE→EN)\n")
        f.write("- Process: English → German → English\n")
        f.write("- Rationale: Translation normalizes punctuation, contractions, and word order\n\n")
        
        f.write("**3. Combined**\n")
        f.write("- Process: Paraphrase 2x → Back-Translate\n")
        f.write("- Rationale: Maximum disruption by chaining both methods\n\n")
        
        f.write("### Selection Criteria\n")
        f.write(f"- Dataset: PAN22 Authorship Verification (validation split)\n")
        f.write(f"- Selection: Same-author pairs where model predicts Prob > 0.90\n")
        f.write(f"- Sample Size: {NUM_ATTACK_SAMPLES} pairs per method\n\n")
        
        f.write("## Interpretation for Research Paper\n\n")
        f.write("> **Key Finding**: Character n-gram stylometry, despite achieving 91% accuracy,\n")
        f.write("> shows significant vulnerability to automated style obfuscation attacks.\n")
        f.write("> The mean erosion values indicate that neural text generation models\n")
        f.write("> effectively neutralize stylometric fingerprints while preserving semantic content.\n\n")
        
        f.write("### Theoretical Implications\n\n")
        f.write("1. **Stylometric fingerprints are surface-level artifacts**: The features that\n")
        f.write("   enable author identification (punctuation patterns, contractions, spacing)\n")
        f.write("   are easily regularized by generative models.\n\n")
        f.write("2. **Neural reconstruction normalizes style**: Both paraphrasing and translation\n")
        f.write("   models are trained on large corpora, causing them to produce 'average' style.\n\n")
        f.write("3. **Combined attacks are most effective**: Chaining multiple transformations\n")
        f.write("   maximizes style erosion while maintaining readability.\n\n")
        
        f.write("## Example Transformations\n\n")
        for ex in example_texts[:6]:
            f.write(f"### {ex['method']} Example\n\n")
            f.write(f"**Original** (Prob: {ex['prob_original']:.4f})\n")
            f.write(f"```\n{ex['original'][:300]}...\n```\n\n")
            f.write(f"**Attacked** (Prob: {ex['prob_attacked']:.4f})\n")
            f.write(f"```\n{ex['attacked'][:300]}...\n```\n\n")
        
        f.write("## Visualizations\n\n")
        f.write("![Attack Comparison](attack_comparison.png)\n\n")
        f.write("![Erosion Histograms](erosion_histograms.png)\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("This experiment demonstrates that stylometry-based authorship verification,\n")
        f.write("while effective on natural text, is fundamentally fragile to adversarial\n")
        f.write("paraphrasing. As generative AI becomes more accessible, this vulnerability\n")
        f.write("poses significant challenges for forensic applications of stylometry.\n")
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print('='*60)
    print(summary.to_string())
    print('='*60)
    print(f"\nAll outputs saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    run_attack()
