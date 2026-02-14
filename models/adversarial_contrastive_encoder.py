import torch
import torch.nn as nn
import torch.nn.functional as F

class AdversarialContrastiveEncoder(nn.Module):
    """
    Learns embeddings robust to adversarial perturbations
    using contrastive learning.
    """
    def __init__(self, input_dim=3000, hidden_dim=512, embed_dim=256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # Temperature for contrastive loss
        self.temperature = 0.07
    
    def forward(self, x):
        return F.normalize(self.encoder(x), p=2, dim=1)
    
    def contrastive_loss(self, anchor, positive, positive_attacked, negative):
        """
        Triplet contrastive loss with adversarial augmentation.
        
        Key idea: 
        - anchor and positive should be close
        - anchor and positive_attacked should ALSO be close (robustness)
        - anchor and negative should be far
        """
        # Compute similarities
        sim_pos = F.cosine_similarity(anchor, positive, dim=1)
        sim_pos_attacked = F.cosine_similarity(anchor, positive_attacked, dim=1)
        sim_neg = F.cosine_similarity(anchor, negative, dim=1)
        
        # Contrastive loss (NT-Xent style)
        logits_pos = torch.exp(sim_pos / self.temperature)
        logits_pos_attacked = torch.exp(sim_pos_attacked / self.temperature)
        logits_neg = torch.exp(sim_neg / self.temperature)
        
        # Numerator: should be high
        numerator = logits_pos + logits_pos_attacked
        
        # Denominator: normalize
        denominator = numerator + logits_neg + 1e-8
        
        loss = -torch.log(numerator / denominator).mean()
        
        return loss
