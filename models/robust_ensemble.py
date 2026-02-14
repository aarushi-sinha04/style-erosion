"""
Robust Multi-Expert Ensemble V2
================================
Uses hard routing + confidence thresholding:
- Siamese specialist for PAN22-like text
- Cross-Domain Siamese for general verification
- Base DANN for cross-domain adaptation
- Weighted voting based on calibrated confidence
"""
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.dann import DANNSiameseV3
from experiments.train_siamese import SiameseNetwork, preprocess

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

# Paths
BASE_DANN_PATH = "results/final_dann/dann_model_v4.pth"
ROBUST_DANN_PATH = "results/robust_dann/robust_dann_model.pth"
SIAMESE_PATH = "results/siamese_baseline/best_model.pth"
SIAMESE_VEC_PATH = "results/siamese_baseline/vectorizer.pkl"
SIAMESE_SCALER_PATH = "results/siamese_baseline/scaler.pkl"
CD_SIAMESE_PATH = "results/siamese_crossdomain/best_model.pth"
CD_SIAMESE_VEC_PATH = "results/siamese_crossdomain/vectorizer.pkl"
CD_SIAMESE_SCALER_PATH = "results/siamese_crossdomain/scaler.pkl"
DANN_EXTRACTOR_PATH = "results/final_dann/extractor.pkl"


class RobustMultiExpertEnsemble:
    """
    Ensemble that combines:
    1. PAN22 Siamese Specialist (97%+ on PAN22)
    2. Cross-Domain Siamese (balanced across domains)
    3. Base DANN (strong on Enron/Blog)
    
    Routing: confidence-weighted voting with domain priors.
    """
    def __init__(self):
        self.device = DEVICE
        self.experts = {}
        
        # Load DANN Extractor
        print("Loading DANN Extractor...")
        self.dann_extractor = pickle.load(open(DANN_EXTRACTOR_PATH, 'rb'))
        
        # Load Base DANN
        self._load_dann("Base DANN", BASE_DANN_PATH)
        
        # Load Robust DANN (optional)
        if os.path.exists(ROBUST_DANN_PATH):
            self._load_dann("Robust DANN", ROBUST_DANN_PATH)
        
        # Load PAN22 Siamese
        self._load_siamese("PAN22 Siamese", SIAMESE_PATH, SIAMESE_VEC_PATH, SIAMESE_SCALER_PATH, 3000)
        
        # Load Cross-Domain Siamese (optional)
        if os.path.exists(CD_SIAMESE_PATH):
            self._load_siamese("CD Siamese", CD_SIAMESE_PATH, CD_SIAMESE_VEC_PATH, CD_SIAMESE_SCALER_PATH, 5000)
    
    def _load_dann(self, name, path):
        try:
            model = DANNSiameseV3(input_dim=4308, num_domains=4).to(self.device)
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()
            self.experts[name] = {'model': model, 'type': 'dann'}
            print(f"  ✓ Loaded {name}")
        except Exception as e:
            print(f"  ✗ Failed to load {name}: {e}")
    
    def _load_siamese(self, name, model_path, vec_path, scaler_path, input_dim):
        try:
            vec = pickle.load(open(vec_path, 'rb'))
            scaler = pickle.load(open(scaler_path, 'rb'))
            model = SiameseNetwork(input_dim=input_dim).to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            self.experts[name] = {'model': model, 'type': 'siamese', 'vec': vec, 'scaler': scaler}
            print(f"  ✓ Loaded {name}")
        except Exception as e:
            print(f"  ✗ Failed to load {name}: {e}")
    
    def _get_dann_features(self, text):
        f_dict = self.dann_extractor.transform([text])
        return np.hstack([f_dict['char'], f_dict['pos'], f_dict['lex'], f_dict['readability']])
    
    def _predict_expert(self, name, text_a, text_b):
        """Get prediction + confidence from one expert."""
        info = self.experts[name]
        
        with torch.no_grad():
            if info['type'] == 'dann':
                fa = self._get_dann_features(text_a)
                fb = self._get_dann_features(text_b)
                xa = torch.tensor(fa, dtype=torch.float32).to(self.device)
                xb = torch.tensor(fb, dtype=torch.float32).to(self.device)
                prob, _, _ = info['model'](xa, xb, alpha=0.0)
                prob = prob.item()
            else:
                ta = preprocess(text_a)
                tb = preprocess(text_b)
                va = info['scaler'].transform(info['vec'].transform([ta]).toarray())
                vb = info['scaler'].transform(info['vec'].transform([tb]).toarray())
                xa = torch.tensor(va, dtype=torch.float32).to(self.device)
                xb = torch.tensor(vb, dtype=torch.float32).to(self.device)
                logits = info['model'](xa, xb)
                prob = torch.sigmoid(logits).item()
        
        # Confidence = distance from decision boundary (0.5)
        confidence = abs(prob - 0.5) * 2  # Normalized to [0, 1]
        return prob, confidence
    
    def predict(self, text_a, text_b, domain='unknown'):
        """
        Weighted voting across all experts with domain priors.
        """
        # Domain-specific weight priors
        domain_weights = {
            'pan22': {'PAN22 Siamese': 3.0, 'CD Siamese': 1.5, 'Base DANN': 0.5, 'Robust DANN': 0.5},
            'blog': {'PAN22 Siamese': 0.5, 'CD Siamese': 2.0, 'Base DANN': 2.0, 'Robust DANN': 1.0},
            'enron': {'PAN22 Siamese': 0.3, 'CD Siamese': 2.0, 'Base DANN': 2.5, 'Robust DANN': 1.0},
            'imdb': {'PAN22 Siamese': 0.3, 'CD Siamese': 1.5, 'Base DANN': 2.0, 'Robust DANN': 1.0},
        }
        
        weights = domain_weights.get(domain.lower(), 
                                     {'PAN22 Siamese': 1.0, 'CD Siamese': 1.5, 'Base DANN': 1.5, 'Robust DANN': 0.5})
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for name in self.experts:
            prob, confidence = self._predict_expert(name, text_a, text_b)
            prior = weights.get(name, 1.0)
            w = prior * (0.3 + 0.7 * confidence)  # Blend prior with confidence
            weighted_sum += w * prob
            total_weight += w
        
        return weighted_sum / (total_weight + 1e-8)
