import numpy as np
import pandas as pd
import pickle
import random
import warnings
import re
import os
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk import word_tokenize, sent_tokenize
from collections import Counter
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "results"
DATA_DIR = "."
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
warnings.filterwarnings('ignore')

# Download NLTK
for package in ['punkt', 'averaged_perceptron_tagger', 'stopwords', 
                'punkt_tab', 'averaged_perceptron_tagger_eng']:
    try:
        nltk.download(package, quiet=True)
    except:
        pass

# ==============================================================================
# 1. CLEANING & STYLO EXTRACTOR
# ==============================================================================
def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'(?i)^(from|to|subject|sent|cc|bcc):.*', '', text, flags=re.MULTILINE)
    text = re.sub(r'-{3,}.*?-{3,}', '', text, flags=re.DOTALL)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class StylometricExtractor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.function_words = ['i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their', 'this', 'that', 'on', 'in', 'at', 'to', 'for', 'of', 'and', 'but', 'or']

    def extract_features(self, text):
        if not text: return self._get_zeros()
        try:
            tokens = word_tokenize(text.lower())
            words = [w for w in tokens if w.isalpha()]
            if not words: return self._get_zeros()
            
            # Lexical
            word_lens = [len(w) for w in words]
            avg_word_len = np.mean(word_lens)
            ttr = len(set(words)) / len(words)
            
            # Character
            total_chars = len(text)
            char_feats = {
                'c_comma': text.count(',') / total_chars,
                'c_period': text.count('.') / total_chars,
                'c_quest': text.count('?') / total_chars,
                'c_excl': text.count('!') / total_chars,
                'c_digit': sum(c.isdigit() for c in text) / total_chars
            }
            
            # Function words
            fw_counts = {f'fw_{fw}': words.count(fw)/len(words) for fw in self.function_words}
            
            features = {
                'lex_awl': avg_word_len,
                'lex_ttr': ttr,
                **char_feats,
                **fw_counts
            }
            return np.array(list(features.values()))
        except:
            return self._get_zeros()

    def _get_zeros(self):
        return np.zeros(2 + 5 + len(self.function_words))

# ==============================================================================
# 2. MAIN PIPELINE
# ==============================================================================
if __name__ == "__main__":
    print(f"{'='*80}\nSCIENCE-GRADE VERIFICATION (v5 - Hybrid Ensemble)\n{'='*80}")
    
    # Load
    df = pd.read_csv(os.path.join(DATA_DIR, "emails.csv"))
    df['author'] = df['file'].apply(lambda x: x.split('/')[0])
    
    # Select
    MIN_EMAILS = 80
    top_authors = df['author'].value_counts()[lambda x: x >= MIN_EMAILS].head(20).index.tolist()
    
    print(f"Authors: {len(top_authors)}")
    
    # Prepared
    data = []
    print("Preparing & Cleaning...")
    for author in top_authors:
        msgs = df[df['author'] == author]['message'].tolist()
        clean = [clean_text(m) for m in msgs]
        # Strict length filter
        clean = [m for m in clean if len(m) >= 300]
        
        if len(clean) >= MIN_EMAILS:
            sample = random.sample(clean, MIN_EMAILS)
            for m in sample:
                data.append({'author': author, 'text': m})
                
    df_clean = pd.DataFrame(data)
    print(f"Dataset: {len(df_clean)} emails.")
    
    # --- FEATURES ---
    print("\n1. SBERT Embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df_clean['text'].tolist(), show_progress_bar=True)
    
    print("2. Stylometric Features...")
    stylo = StylometricExtractor()
    stylo_feats = np.array([stylo.extract_features(t) for t in df_clean['text']])
    scaler = MinMaxScaler()
    stylo_feats = scaler.fit_transform(stylo_feats)
    
    # --- PAIRS ---
    print("\nGenerating Pairs...")
    X_train, y_train, X_test, y_test = [], [], [], []
    
    # Split Author-based
    # We split texts per author 70/30
    indices_by_auth = {}
    for i, row in df_clean.iterrows():
        a = row['author']
        if a not in indices_by_auth: indices_by_auth[a] = []
        indices_by_auth[a].append(i)
        
    train_indices = []
    test_indices = []
    
    for a, idxs in indices_by_auth.items():
        split = int(len(idxs) * 0.70)
        train_indices.extend(idxs[:split])
        test_indices.extend(idxs[split:])
        
    def get_vec(idx):
        # Concatenate Stylo + SBERT
        return np.concatenate([stylo_feats[idx], embeddings[idx]])
    
    feature_dim = len(get_vec(0))
    print(f"Combined Feature Dim: {feature_dim}")
    
    def diff_features(v1, v2):
        return np.abs(v1 - v2)
        
    # Generate
    # Train
    for _ in range(3000): # 3000 pairs
        # Pos
        a = random.choice(top_authors)
        idxs = [i for i in indices_by_auth[a] if i in train_indices]
        if len(idxs) < 2: continue
        i1, i2 = random.sample(idxs, 2)
        X_train.append(diff_features(get_vec(i1), get_vec(i2)))
        y_train.append(1)
        
        # Neg
        a1, a2 = random.sample(top_authors, 2)
        idx1 = random.choice([i for i in indices_by_auth[a1] if i in train_indices])
        idx2 = random.choice([i for i in indices_by_auth[a2] if i in train_indices])
        X_train.append(diff_features(get_vec(idx1), get_vec(idx2)))
        y_train.append(0)
        
    # Test
    for _ in range(1000):
        a = random.choice(top_authors)
        idxs = [i for i in indices_by_auth[a] if i in test_indices]
        if len(idxs) < 2: continue
        i1, i2 = random.sample(idxs, 2)
        X_test.append(diff_features(get_vec(i1), get_vec(i2)))
        y_test.append(1)
        
        a1, a2 = random.sample(top_authors, 2)
        idx1 = random.choice([i for i in indices_by_auth[a1] if i in test_indices])
        idx2 = random.choice([i for i in indices_by_auth[a2] if i in test_indices])
        X_test.append(diff_features(get_vec(idx1), get_vec(idx2)))
        y_test.append(0)
        
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    print(f"Training Pairs: {X_train.shape}")
    
    # --- ENSEMBLE TRAINING ---
    print("\nTraining Ensemble (Gradient Boosting + RF)...")
    clf1 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    clf2 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    eclf = VotingClassifier(estimators=[('gb', clf1), ('rf', clf2)], voting='soft')
    
    eclf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = eclf.predict(X_test)
    y_prob = eclf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    print("\n" + "#"*60)
    print("FINAL RESULTS (v5 - Hybrid Ensemble)")
    print("#"*60)
    print(f"Accuracy:  {acc:.4f}")
    print(f"ROC-AUC:   {roc:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("#"*60)
    
    with open(f"{OUTPUT_DIR}/report_v5.txt", "w") as f:
         f.write(f"Accuracy: {acc:.4f}\nROC-AUC: {roc:.4f}\nF1: {f1:.4f}\n")
         
    # Save model
    with open(f"{OUTPUT_DIR}/model_ensemble_v5.pkl", "wb") as f: pickle.dump(eclf, f)
    
    print("Done!")