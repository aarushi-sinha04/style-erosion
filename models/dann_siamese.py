import torch
import torch.nn as nn

class GradientReversalFunction(torch.autograd.Function):
    """
    Reverses gradients during backprop.
    This forces the feature extractor to learn features that:
    - Are good for authorship verification
    - Are bad for domain classification
    """
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

class DANNSiamese(nn.Module):
    """
    Domain-Adversarial Siamese Network.
    
    Architecture:
    1. Feature Extractor (shared for both texts)
    2. Authorship Classifier (same vs different author)
    3. Domain Classifier (PAN22 vs Blog vs Enron vs IMDB) with gradient reversal
    """
    def __init__(self, input_dim=4308, hidden_dim=512, num_domains=4):
        super().__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Authorship classifier (predicts same vs different author)
        self.authorship_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Domain classifier with gradient reversal
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(lambda_=1.0),
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_domains) 
            # Output raw logits (CrossEntropyLoss includes Softmax)
        )
    
    def forward(self, text_a, text_b, alpha=1.0):
        """
        alpha: scaling factor for gradient reversal (lambda)
        """
        # Set lambda for GRL
        self.domain_classifier[0].lambda_ = alpha
        
        # Extract features for both texts
        features_a = self.feature_extractor(text_a)
        features_b = self.feature_extractor(text_b)
        
        # Authorship prediction (based on feature difference)
        diff = torch.abs(features_a - features_b)
        authorship_pred = self.authorship_classifier(diff)
        
        # Domain prediction (for both texts separately)
        # We want to confuse domain classifier on individual texts too.
        domain_pred_a = self.domain_classifier(features_a)
        domain_pred_b = self.domain_classifier(features_b)
        
        return authorship_pred, domain_pred_a, domain_pred_b
