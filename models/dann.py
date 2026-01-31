"""
DANN Siamese V3 - Enhanced with Attention and Spectral Normalization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientReversalFunction(torch.autograd.Function):
    """Reverses gradients during backprop."""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class SelfAttention(nn.Module):
    """Self-attention to focus on domain-invariant features."""
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x: [batch, embed_dim] -> [batch, 1, embed_dim]
        x = x.unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        return x.squeeze(1)

class DANNSiameseV3(nn.Module):
    """
    Enhanced Domain-Adversarial Siamese Network.
    
    Improvements over V2:
    1. Self-attention for domain-invariant feature selection
    2. Spectral normalization for training stability
    3. Deeper domain classifier (harder adversarial task)
    4. Projection head for contrastive alignment
    """
    def __init__(self, input_dim=4308, hidden_dim=512, num_domains=4):
        super().__init__()
        
        # Shared feature extractor with spectral normalization
        self.feature_extractor = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_dim, 1024)),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Linear(1024, hidden_dim)),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # Self-attention for domain-invariant focus
        self.attention = SelfAttention(hidden_dim, num_heads=4)
        
        # Projection head for distribution alignment
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Authorship classifier (predicts same vs different author)
        self.authorship_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Deeper domain classifier with gradient reversal (harder adversarial task)
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(lambda_=1.0),
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(256, num_domains)
        )
    
    def forward(self, text_a, text_b, alpha=1.0):
        """
        alpha: scaling factor for gradient reversal
        """
        # Set lambda for GRL (first layer of domain_classifier)
        self.domain_classifier[0].lambda_ = alpha
        
        # Extract features for both texts
        features_a = self.feature_extractor(text_a)
        features_b = self.feature_extractor(text_b)
        
        # Apply attention
        features_a = self.attention(features_a)
        features_b = self.attention(features_b)
        
        # Authorship prediction (based on feature difference)
        diff = torch.abs(features_a - features_b)
        authorship_pred = self.authorship_classifier(diff)
        
        # Domain prediction
        domain_pred_a = self.domain_classifier(features_a)
        domain_pred_b = self.domain_classifier(features_b)
        
        return authorship_pred, domain_pred_a, domain_pred_b
    
    def get_embeddings(self, x):
        """Get embeddings for visualization."""
        features = self.feature_extractor(x)
        features = self.attention(features)
        return features
    
    def get_projections(self, x):
        """Get projections for MMD alignment."""
        features = self.feature_extractor(x)
        features = self.attention(features)
        return self.projection(features)


def compute_mmd(source, target, kernel='rbf', sigma=1.0):
    """
    Maximum Mean Discrepancy loss for distribution alignment.
    Aligns feature distributions across domains.
    """
    n_source = source.size(0)
    n_target = target.size(0)
    
    if n_source == 0 or n_target == 0:
        return torch.tensor(0.0, device=source.device)
    
    # Compute kernel matrices
    if kernel == 'rbf':
        # Source-Source
        ss = torch.cdist(source, source, p=2)
        k_ss = torch.exp(-ss ** 2 / (2 * sigma ** 2))
        
        # Target-Target
        tt = torch.cdist(target, target, p=2)
        k_tt = torch.exp(-tt ** 2 / (2 * sigma ** 2))
        
        # Source-Target
        st = torch.cdist(source, target, p=2)
        k_st = torch.exp(-st ** 2 / (2 * sigma ** 2))
    else:
        # Linear kernel
        k_ss = source @ source.t()
        k_tt = target @ target.t()
        k_st = source @ target.t()
    
    # MMD formula
    mmd = k_ss.mean() + k_tt.mean() - 2 * k_st.mean()
    return mmd


def compute_center_loss(features, labels, centers, num_classes=2):
    """
    Center loss to reduce intra-class variation.
    Pulls features towards their class centers.
    """
    batch_size = features.size(0)
    
    # Get centers for each sample
    centers_batch = centers[labels.long()]
    
    # L2 distance to centers
    loss = ((features - centers_batch) ** 2).sum() / batch_size
    
    return loss


def update_centers(centers, features, labels, alpha=0.5):
    """Update class centers with exponential moving average."""
    for i in range(centers.size(0)):
        mask = (labels == i)
        if mask.sum() > 0:
            centers[i] = alpha * centers[i] + (1 - alpha) * features[mask].mean(0)
    return centers
