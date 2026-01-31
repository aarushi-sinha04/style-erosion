import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalSiamese(nn.Module):
    def __init__(self, input_dims):
        """
        input_dims: dict of dimensions {'char': 3000, 'pos': 1000, 'lex': 300, 'readability': 8}
        """
        super(HierarchicalSiamese, self).__init__()
        
        self.input_dims = input_dims
        COMMON_DIM = 128
        DROPOUT = 0.3
        
        # 1. View Encoders (Project all to COMMON_DIM)
        # Char (3000 -> 128)
        self.char_encoder = nn.Sequential(
            nn.Linear(input_dims['char'], 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(512, COMMON_DIM),
            nn.BatchNorm1d(COMMON_DIM), nn.ReLU()
        )
        
        # POS (1000 -> 128)
        self.pos_encoder = nn.Sequential(
            nn.Linear(input_dims['pos'], 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(256, COMMON_DIM),
            nn.BatchNorm1d(COMMON_DIM), nn.ReLU()
        )
        
        # Lexical (300 -> 128)
        self.lex_encoder = nn.Sequential(
            nn.Linear(input_dims['lex'], 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(128, COMMON_DIM),
            nn.BatchNorm1d(COMMON_DIM), nn.ReLU()
        )
        
        # Readability (8 -> 128)
        self.read_encoder = nn.Sequential(
            nn.Linear(input_dims['readability'], 64),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(64, COMMON_DIM),
            nn.BatchNorm1d(COMMON_DIM), nn.ReLU()
        )
        
        # 2. Attention Mechanism (View Attention)
        # We treat the 4 views as a sequence of length 4.
        # We want to learn which view is important.
        self.attention = nn.MultiheadAttention(embed_dim=COMMON_DIM, num_heads=4, batch_first=True)
        
        # 3. Aggregation & Similarity Head
        # We flatten the attended views: 4 * 128 = 512
        FLATTENED_DIM = 4 * COMMON_DIM
        
        self.head = nn.Sequential(
            nn.Linear(FLATTENED_DIM * 4, 512), # *4 because (u, v, |u-v|, u*v)
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1) # Logits
        )
        
    def forward_one(self, x_dict):
        """
        x_dict: dict of tensors {'char': [B, 3000], ...}
        Returns: embedding [B, 4*128], attention_weights [B, 4, 4]
        """
        # Encode
        char_emb = self.char_encoder(x_dict['char'])
        pos_emb = self.pos_encoder(x_dict['pos'])
        lex_emb = self.lex_encoder(x_dict['lex'])
        read_emb = self.read_encoder(x_dict['readability'])
        
        # Stack -> [Batch, 4, COMMON_DIM]
        # Order: Char, Pos, Lex, Read
        view_stack = torch.stack([char_emb, pos_emb, lex_emb, read_emb], dim=1)
        
        # Attention
        # attn_output: [Batch, 4, COMMON_DIM]
        # attn_weights: [Batch, 4, 4] (Query vs Key)
        # We use self-attention to let views contextuallize each other
        attn_output, attn_weights = self.attention(view_stack, view_stack, view_stack)
        
        # Residual connection + Norm? Or just use attended?
        # Let's use Residual + Norm usually, but simple is fine for now.
        # Just Flatten.
        
        embeddings = attn_output.reshape(attn_output.size(0), -1) # [Batch, 512]
        
        return embeddings, attn_weights

    def forward(self, x1_dict, x2_dict):
        u, _ = self.forward_one(x1_dict)
        v, _ = self.forward_one(x2_dict)
        
        # Interaction
        diff = torch.abs(u - v)
        prod = u * v
        
        combined = torch.cat([u, v, diff, prod], dim=1)
        logits = self.head(combined)
        
        return logits
