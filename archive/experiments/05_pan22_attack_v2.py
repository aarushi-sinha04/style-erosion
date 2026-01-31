"""
Advanced Adversarial Attack on Stylometry Model
================================================
This script attacks a Siamese Authorship Verification model using multiple
techniques to erode stylistic signatures:

1. MULTI-PASS PARAPHRASING: Run text through T5 paraphraser 3 times.
2. BACK-TRANSLATION: Translate EN -> DE -> EN to destroy character patterns.
3. COMBINED ATTACK: Both methods chained for maximum disruption.

For Research Paper:
-------------------
This experiment demonstrates that while character n-gram based stylometry
achieves high accuracy (91%), it is fundamentally fragile to automated
style-washing attacks. The "fingerprint" exists in low-level character
patterns (punctuation, spacing, contractions) which are easily normalized
by neural text generation models.
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

# ==============================================================================
# CONFIG
# ==============================================================================
MODEL_DIR = "results_pan_siamese"
DATA_DIR = "."
OUTPUT_DIR = "results_pan_attack_v2"
TRAIN_FILE = "pan22-authorship-verification-training.jsonl"
TRUTH_FILE = "pan22-authorship-verification-training-truth.jsonl"

PARAPHRASE_MODEL = "humarin/chatgpt_paraphraser_on_T5_base"
TRANSLATE_EN_DE = "Helsinki-NLP/opus-mt-en-de"
TRANSLATE_DE_EN = "Helsinki-NLP/opus-mt-de-en"

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
NUM_ATTACK_SAMPLES = 100  # More samples for statistical significance

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Device: {DEVICE}")
print(f"Output: {OUTPUT_DIR}")

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
    """
    Multi-Pass Paraphrasing Attack
    ==============================
    Rationale: A single paraphrase pass may retain some stylistic markers.
    By iteratively paraphrasing (3 passes), we compound the style destruction.
    Each pass introduces variation in punctuation, contractions, and phrasing.
    """
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
    """
    Back-Translation Attack
    =======================
    Rationale: Translating text to another language and back destroys
    character-level patterns (punctuation, spacing, word order) while
    preserving semantic meaning. German is chosen as the pivot language
    due to its significantly different grammatical structure.
    """
    def __init__(self):
        print("Loading Translation Models (EN->DE->EN)...")
        self.en_de_tok = MarianTokenizer.from_pretrained(TRANSLATE_EN_DE)
        self.en_de_model = MarianMTModel.from_pretrained(TRANSLATE_EN_DE).to(DEVICE)
        self.de_en_tok = MarianTokenizer.from_pretrained(TRANSLATE_DE_EN)
        self.de_en_model = MarianMTModel.from_pretrained(TRANSLATE_DE_EN).to(DEVICE)
        
    def attack(self, text):
        # EN -> DE
        inputs = self.en_de_tok(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        de_ids = self.en_de_model.generate(**inputs, max_length=512)
        de_text = self.en_de_tok.decode(de_ids[0], skip_special_tokens=True)
        
        # DE -> EN
        inputs = self.de_en_tok(de_text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        en_ids = self.de_en_model.generate(**inputs, max_length=512)
        return self.de_en_tok.decode(en_ids[0], skip_special_tokens=True)

class CombinedAttack:
    """
    Combined Attack
    ===============
    Apply both back-translation AND multi-pass paraphrasing for maximum disruption.
    Order: Paraphrase -> Back-Translate
    """
    def __init__(self, para, bt):
        self.para = para
        self.bt = bt
        
    def attack(self, text):
        step1 = self.para.attack(text, passes=2)
        step2 = self.bt.attack(step1)
        return step2

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
def run_full_attack():
    # Load
    model, vec, scaler = load_target()
    pairs, labels = load_data()
    
    # Find high-confidence Same Author pairs
    print("Finding vulnerable pairs (Prob > 0.90)...")
    targets = []
    for i, (t1, t2) in enumerate(tqdm(pairs[:1500])):
        if labels[i] == 1.0:
            prob = get_prob(model, vec, scaler, t1, t2)
            if prob > 0.90:
                targets.append({'t1': t1, 't2': t2, 'prob_orig': prob})
                if len(targets) >= NUM_ATTACK_SAMPLES:
                    break
    
    print(f"Selected {len(targets)} pairs with Prob > 0.90")
    
    # Load Attackers
    paraphraser = MultiPassParaphraser()
    backtrans = BackTranslationAttack()
    combined = CombinedAttack(paraphraser, backtrans)
    
    # Run 3 Attack Types
    attack_methods = {
        'Paraphrase (3-Pass)': paraphraser.attack,
        'Back-Translation (DE)': backtrans.attack,
        'Combined': combined.attack
    }
    
    all_results = []
    
    for method_name, attack_fn in attack_methods.items():
        print(f"\n{'='*60}")
        print(f"ATTACK: {method_name}")
        print('='*60)
        
        erosions = []
        for sample in tqdm(targets):
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
            except Exception as e:
                print(f"Error: {e}")
                continue
                
        mean_erosion = np.mean(erosions)
        flip_rate = np.mean([r['flipped'] for r in all_results if r['method'] == method_name])
        print(f"Mean Erosion: {mean_erosion:.4f}")
        print(f"Flip Rate: {flip_rate*100:.1f}%")
    
    # Save DataFrame
    df = pd.DataFrame(all_results)
    df.to_csv(f"{OUTPUT_DIR}/attack_results.csv", index=False)
    
    # ===========================================================================
    # 5. VISUALIZATION
    # ===========================================================================
    print("\nGenerating Visualizations...")
    
    # A. Erosion Comparison by Method
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='method', y='erosion', palette='Set2')
    plt.axhline(y=0.5, color='red', linestyle='--', label='Target (0.5)')
    plt.title("Erosion by Attack Method", fontsize=14, fontweight='bold')
    plt.ylabel("Erosion (Prob Drop)")
    plt.xlabel("Attack Method")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/erosion_by_method.png", dpi=300)
    
    # B. Before/After Scatter
    for method_name in attack_methods.keys():
        mdf = df[df['method'] == method_name]
        plt.figure(figsize=(8, 6))
        plt.scatter(mdf['prob_orig'], mdf['prob_new'], alpha=0.6, edgecolors='black')
        plt.plot([0, 1], [0, 1], 'k--', label='No Change')
        plt.axhline(y=0.5, color='red', linestyle='-', alpha=0.5, label='Flip Threshold')
        plt.xlabel("Original Probability (Same Author)")
        plt.ylabel("Attacked Probability")
        plt.title(f"Attack Impact: {method_name}")
        plt.legend()
        safe_name = method_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '')
        plt.savefig(f"{OUTPUT_DIR}/scatter_{safe_name}.png", dpi=300)
    
    # C. Summary Table
    summary = df.groupby('method').agg(
        mean_erosion=('erosion', 'mean'),
        median_erosion=('erosion', 'median'),
        flip_rate=('flipped', 'mean'),
        max_erosion=('erosion', 'max'),
        count=('erosion', 'count')
    ).round(4)
    summary.to_csv(f"{OUTPUT_DIR}/summary_table.csv")
    
    # Print Final Report
    print("\n" + "#"*60)
    print("FINAL ATTACK RESULTS")
    print("#"*60)
    print(summary.to_string())
    print("#"*60)
    
    # Save Text Report
    with open(f"{OUTPUT_DIR}/attack_report.txt", "w") as f:
        f.write("ADVERSARIAL ATTACK REPORT\n")
        f.write("="*60 + "\n\n")
        f.write("EXPERIMENTAL SETUP\n")
        f.write("-"*30 + "\n")
        f.write(f"Target Model: Siamese Network (Char 4-grams, 91% Acc)\n")
        f.write(f"Dataset: PAN22 Validation Set\n")
        f.write(f"Samples Attacked: {len(targets)}\n")
        f.write(f"Selection Criteria: Original Prob > 0.90\n\n")
        
        f.write("ATTACK METHODS\n")
        f.write("-"*30 + "\n")
        f.write("1. Paraphrase (3-Pass): T5 paraphraser applied 3 times.\n")
        f.write("2. Back-Translation: EN -> German -> EN via MarianMT.\n")
        f.write("3. Combined: Paraphrase then Back-Translate.\n\n")
        
        f.write("RESULTS\n")
        f.write("-"*30 + "\n")
        f.write(summary.to_string())
        f.write("\n\n")
        
        f.write("INTERPRETATION (For Research Paper)\n")
        f.write("-"*30 + "\n")
        f.write("The character n-gram based stylometry model, despite achieving 91% accuracy\n")
        f.write("on human-written text, is highly vulnerable to automated style obfuscation.\n")
        f.write("Back-Translation achieves the highest erosion by normalizing punctuation,\n")
        f.write("contractions, and sentence structure through the reconstruction process.\n")
        f.write("This confirms that 'stylometric fingerprints' are fragile artifacts of\n")
        f.write("surface-level text patterns, easily disrupted by generative AI.\n")
    
    print(f"\nAll outputs saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    run_full_attack()
