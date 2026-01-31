import spacy
import textstat
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
import os

# Download nltk resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model not found. Downloading...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class EnhancedFeatureExtractor:
    """
    Extracts multi-view stylometric features for Domain-Adversarial Verifiction.
    Views:
    1. Character: 4-grams (Standard)
    2. Syntactic: POS-tag trigrams (e.g. DET-ADJ-NOUN)
    3. Lexical: Function word frequencies
    4. Readability: Complexity metrics
    """
    def __init__(self, max_features_char=3000, max_features_pos=1000, max_features_lex=300):
        self.max_features_char = max_features_char
        self.max_features_pos = max_features_pos
        self.max_features_lex = max_features_lex
        
        # Sub-Vectorizers
        self.char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(4,4), max_features=max_features_char, sublinear_tf=True)
        self.pos_vectorizer = TfidfVectorizer(ngram_range=(3,3), max_features=max_features_pos)
        self.lex_vectorizer = CountVectorizer(max_features=max_features_lex) # Will allow it to learn common words
        
        # State
        self.is_fitted = False
        
    def _get_pos_tags(self, text):
        """Extracts space-separated POS tags."""
        doc = nlp(str(text)[:10000]) # Truncate for speed if necessary
        return " ".join([token.pos_ for token in doc])
        
    def _get_readability_vector(self, text):
        """Extracts scalar readability metrics."""
        text = str(text)
        try:
            fk = textstat.flesch_kincaid_grade(text)
            ari = textstat.automated_readability_index(text)
            dc = textstat.dale_chall_readability_score(text)
            cli = textstat.coleman_liau_index(text)
            
            # Simple stats
            chars = len(text)
            words = len(text.split())
            sents = textstat.sentence_count(text)
            avg_sent_len = words / max(1, sents)
            avg_word_len = chars / max(1, words)
            
            return np.array([fk, ari, dc, cli, avg_sent_len, avg_word_len, words, chars], dtype=np.float32)
        except:
            return np.zeros(8, dtype=np.float32)

    def fit(self, texts):
        print("Fitting EnhancedFeatureExtractor...")
        
        # 1. Char 4-grams
        print("Fitting Char Vectorizer...")
        self.char_vectorizer.fit(texts)
        
        # 2. POS Trigrams (Expensive)
        print("Extracting POS tags (this may take a while)...")
        # In production, use nlp.pipe for batching
        pos_texts = []
        for doc in nlp.pipe(texts, batch_size=50, disable=["ner", "parser", "lemmatizer"]):
             pos_texts.append(" ".join([t.pos_ for t in doc]))
        
        print("Fitting POS Vectorizer...")
        self.pos_vectorizer.fit(pos_texts)
        
        # 3. Lexical (Function words / common words)
        print("Fitting Lexical Vectorizer...")
        self.lex_vectorizer.fit(texts)
        
        self.is_fitted = True
        print("Extractor Fitted.")
        
    def transform(self, texts):
        if not self.is_fitted:
            raise ValueError("Extractor not fitted.")
            
        print("Transforming texts...")
        
        # 1. Char
        char_feats = self.char_vectorizer.transform(texts).toarray()
        
        # 2. POS
        pos_texts = []
        for doc in nlp.pipe(texts, batch_size=50, disable=["ner", "parser", "lemmatizer"]):
             pos_texts.append(" ".join([t.pos_ for t in doc]))
        pos_feats = self.pos_vectorizer.transform(pos_texts).toarray()
        
        # 3. Lexical
        lex_feats = self.lex_vectorizer.transform(texts).toarray()
        
        # 4. Readability
        read_feats = np.array([self._get_readability_vector(t) for t in texts])
        
        return {
            'char': char_feats,
            'pos': pos_feats,
            'lex': lex_feats,
            'readability': read_feats
        }

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
