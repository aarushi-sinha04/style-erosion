import torch
import numpy as np

class EnsembleVerifier:
    """
    Ensemble Defense Mechanism (SCIE Phase 5).
    Combines predictions from multiple experts:
    1. DANN (Robust, Domain-Invariant)
    2. Char-Level (Classic Stylometry)
    3. Syntax-Level (POS - Structural)
    """
    def __init__(self, dann_model, extractor, device='cpu'):
        self.dann = dann_model
        self.extractor = extractor
        self.device = device
        self.dann.eval()
        
    def predict(self, t1_list, t2_list):
        """
        Returns Ensemble Prediction (Probability).
        Strategy: Weighted Average.
        """
        # 1. DANN Prediction (The Heavy Lifter)
        f1_dict = self.extractor.transform(t1_list)
        f2_dict = self.extractor.transform(t2_list)
        
        def flatten(d): 
            return np.hstack([d['char'], d['pos'], d['lex'], d['readability']])
        
        # Pad
        def pad(X):
             dim = 4308
             if X.shape[1] < dim: return np.hstack([X, np.zeros((X.shape[0], dim - X.shape[1]))])
             return X[:, :dim]

        X1 = pad(flatten(f1_dict))
        X2 = pad(flatten(f2_dict))
        
        X1_t = torch.tensor(X1, dtype=torch.float32).to(self.device)
        X2_t = torch.tensor(X2, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            pred_dann, _, _ = self.dann(X1_t, X2_t, alpha=0.0)
            prob_dann = pred_dann.cpu().numpy().flatten()
            
        # 2. Syntax Expert (POS Cosine Similarity)
        # Simple heuristic: If POS distribution is very different, likely different authors.
        pos1 = f1_dict['pos']
        pos2 = f2_dict['pos']
        
        # Cosine Sim
        from sklearn.metrics.pairwise import cosine_similarity
        sim_pos = np.diag(cosine_similarity(pos1, pos2))
        prob_pos = (sim_pos + 1) / 2 # Normalize -1..1 to 0..1
        
        # 3. Char Expert
        char1 = f1_dict['char']
        char2 = f2_dict['char']
        sim_char = np.diag(cosine_similarity(char1, char2))
        prob_char = (sim_char + 1) / 2
        
        # Ensemble Weights
        # DANN is trained, so it gets highest weight.
        # Syntax is good for checking structural consistency.
        w_dann = 0.6
        w_pos = 0.2
        w_char = 0.2
        
        ensemble_prob = (w_dann * prob_dann) + (w_pos * prob_pos) + (w_char * prob_char)
        return ensemble_prob

if __name__ == "__main__":
    # Test logic
    print("Ensemble Verifier Drafted.")
