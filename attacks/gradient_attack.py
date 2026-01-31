import torch
import numpy as np
import re

class GradientAttacker:
    """
    White-Box Attack using Feature-Space Gradients.
    Since the model uses TF-IDF/Count vectors, we compute gradients w.r.t the input vector.
    
    Algorithm:
    1. Forward pass pair (X1, X2).
    2. Compute Loss (Maximize error -> Minimize -Loss).
    3. Backward pass to get Gradient w.r.t X2 (Target).
    4. Find features with highest Gradient (Sensitivity).
    5. Map features back to words.
    6. Insert high-gradient words to flip prediction.
    """
    def __init__(self, model, extractor, device='cpu'):
        self.model = model
        self.extractor = extractor
        self.device = device
        self.feature_names = self._get_feature_mapping()
        
    def _get_feature_mapping(self):
        """Map feature indices to names (e.g., 'word', '4gram')."""
        # EnhancedFeatureExtractor flattens: Char (3000) + POS (1000) + Lex (300) + Read (8)
        # We need to map global index to specific vocab
        
        mapping = {}
        offset = 0
        
        # Char 4-grams
        if hasattr(self.extractor, 'char_vectorizer'):
            for k, v in self.extractor.char_vectorizer.vocabulary_.items():
                mapping[v + offset] = {'type': 'char', 'value': k}
            offset += self.extractor.max_features_char
            
        # POS (Hard to manipulate directly, skipping)
        offset += self.extractor.max_features_pos
        
        # Lexical (Function words - EASY to insert)
        if hasattr(self.extractor, 'lex_vectorizer'):
            for k, v in self.extractor.lex_vectorizer.vocabulary_.items():
                mapping[v + offset] = {'type': 'lex', 'value': k}
            offset += self.extractor.max_features_lex
            
        return mapping

    def attack(self, t1_text, t2_text, true_label, max_changes=5):
        """
        Modifies t2_text to change the model's prediction.
        """
        self.model.eval()
        
        # 1. Initial State
        f1 = self._vectorize(t1_text)
        f2 = self._vectorize(t2_text)
        
        # Enable grad for X2
        f2.requires_grad = True
        
        # 2. Forward
        pred_auth, _, _ = self.model(f1, f2, alpha=0.0)
        
        # 3. Loss (We want to FLIP the prediction)
        # If True=1 (Same), we want Pred to be 0 (Diff).
        # So we maximize BCELoss(Pred, 1) or Minimize BCELoss(Pred, 0)
        target_adv = 1.0 - true_label
        target_tensor = torch.tensor([target_adv], device=self.device)
        
        loss = torch.nn.BCELoss()(pred_auth.squeeze(0), target_tensor)
        
        # 4. Backward
        loss.backward()
        grad = f2.grad.cpu().numpy().flatten()
        
        # 5. Identify Features
        # We want to change X2 such that Loss decreases (towards target).
        # X_new = X_old - lr * grad
        # Since TF-IDF is non-negative and typically sparse:
        # - Gradient < 0: Increasing this feature reduces loss (Good for insertion).
        # - Gradient > 0: Decreasing this feature reduces loss (Good for deletion).
        
        # We focus on INSERTION (Lexical features) as it's easier than deletion.
        # Find negative gradients with largest magnitude.
        
        valid_indices = [i for i in range(len(grad)) if i in self.feature_names and self.feature_names[i]['type'] == 'lex']
        
        if not valid_indices:
            print("No modifiable lexical features found.")
            return t2_text
            
        # Sort by gradient (Lowest/Most Negative first)
        # Most negative grad => Increasing this feature minimizes loss towards target.
        sorted_indices = sorted(valid_indices, key=lambda i: grad[i])
        
        # Pick top changes
        changes = []
        for idx in sorted_indices[:max_changes]:
            # If grad is negative, we want to INSERT
            if grad[idx] < 0:
                word = self.feature_names[idx]['value']
                changes.append(word)
                
        # 6. Apply Changes (Append words)
        adv_text = t2_text + " " + " ".join(changes)
        
        return adv_text

    def _vectorize(self, text):
        # Helper to bypass full extractor fitting and jus transform single
        # Assumes extractor fits
        d = self.extractor.transform([text])
        x = np.hstack([d['char'], d['pos'], d['lex'], d['readability']])
        
        # Pad
        dim = 4308
        if x.shape[1] < dim:
            x = np.hstack([x, np.zeros((1, dim - x.shape[1]))])
        else:
            x = x[:, :dim]
            
        return torch.tensor(x, dtype=torch.float32, device=self.device)
