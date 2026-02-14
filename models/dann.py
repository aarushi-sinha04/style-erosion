"""
DANN Siamese V4 - Fixed Architecture
=====================================
Changes from V3:
1. REMOVED spectral normalization (was constraining learning)
2. ADDED rich interaction features (u, v, |u-v|, u*v) like the Siamese
3. REMOVED self-attention on single vector (wasteful)
4. Kept gradient reversal for domain adaptation
5. Kept projection head for MMD
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


class DANNSiameseV3(nn.Module):
    """
    Fixed Domain-Adversarial Siamese Network.

    Key fix: Uses rich interaction features [u, v, |u-v|, u*v] instead of
    just |u-v|, and removes spectral norm that was preventing learning.
    """
    def __init__(self, input_dim=4308, hidden_dim=512, num_domains=4):
        super().__init__()

        # Shared feature extractor - NO spectral norm
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

        # Projection head for MMD alignment
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # Authorship classifier - RICH INTERACTION [u, v, |u-v|, u*v] -> 4*hidden
        self.authorship_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
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
        )

    def forward(self, text_a, text_b, alpha=1.0):
        """alpha: scaling factor for gradient reversal"""
        self.domain_classifier[0].lambda_ = alpha

        # Extract features
        features_a = self.feature_extractor(text_a)
        features_b = self.feature_extractor(text_b)

        # Rich interaction features (like Siamese)
        diff = torch.abs(features_a - features_b)
        prod = features_a * features_b
        combined = torch.cat([features_a, features_b, diff, prod], dim=1)

        # Authorship prediction
        authorship_pred = self.authorship_classifier(combined)

        # Domain prediction (per-text, not per-pair)
        domain_pred_a = self.domain_classifier(features_a)
        domain_pred_b = self.domain_classifier(features_b)

        return authorship_pred, domain_pred_a, domain_pred_b

    def get_embeddings(self, x):
        return self.feature_extractor(x)

    def get_projections(self, x):
        features = self.feature_extractor(x)
        return self.projection(features)


def compute_mmd(source, target, kernel='rbf', sigma=1.0):
    """Maximum Mean Discrepancy for distribution alignment."""
    n_source = source.size(0)
    n_target = target.size(0)

    if n_source == 0 or n_target == 0:
        return torch.tensor(0.0, device=source.device)

    if kernel == 'rbf':
        ss = torch.cdist(source, source, p=2)
        k_ss = torch.exp(-ss ** 2 / (2 * sigma ** 2))
        tt = torch.cdist(target, target, p=2)
        k_tt = torch.exp(-tt ** 2 / (2 * sigma ** 2))
        st = torch.cdist(source, target, p=2)
        k_st = torch.exp(-st ** 2 / (2 * sigma ** 2))
    else:
        k_ss = source @ source.t()
        k_tt = target @ target.t()
        k_st = source @ target.t()

    mmd = k_ss.mean() + k_tt.mean() - 2 * k_st.mean()
    return mmd


def compute_center_loss(features, labels, centers, num_classes=2):
    """Center loss to reduce intra-class variation."""
    batch_size = features.size(0)
    centers_batch = centers[labels.long()]
    loss = ((features - centers_batch) ** 2).sum() / batch_size
    return loss


def update_centers(centers, features, labels, alpha=0.5):
    """Update class centers with exponential moving average."""
    for i in range(centers.size(0)):
        mask = (labels == i)
        if mask.sum() > 0:
            centers[i] = alpha * centers[i] + (1 - alpha) * features[mask].mean(0)
    return centers
